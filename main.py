import requests
import datetime
import pytz
import pandas as pd
import numpy as np
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import traceback

# --- CONFIGURATION ---
TOKEN = "8212480058:AAHrq5yjlAzrnOlAla0IE42V2Z2w4Y05s80"
CHAT_ID = "1091559539"
SYMBOLS = ["BTCUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "ETHUSDT", "DASHUSDT", "BCHUSDT", "FILUSDT", "LTCUSDT", "YFIUSDT", "ZECUSDT"]
TIMEZONE = pytz.timezone("Europe/Paris")
SIGNAL_COOLDOWN_MINUTES = 5
CONFIDENCE_THRESHOLD = 60

# --- PARAMÈTRES INDICATEURS ---
RSI_PERIOD = 14
EMA_PERIOD = 9
EMA_TREND = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
CCI_PERIOD = 20
VOLUME_PERIOD = 20
BB_PERIOD = 20
STOCH_PERIOD = 14
RANGE_THRESHOLD = 0.15

# --- VARIABLES GLOBALES ---
app = Application.builder().token(TOKEN).build()
is_running = False
last_sent_signals = {}
active_signals = {}
last_alert_time = datetime.datetime.now(TIMEZONE)

# --- FONCTIONS ESSENTIELLES ---
def get_ohlcv(symbol, interval):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(TIMEZONE)
    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float
    })
    return df

def calculate_indicators(df):
    df["EMA"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=EMA_TREND, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    exp1 = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.fabs(x - x.mean()).mean())
    df["CCI"] = (tp - sma) / (0.015 * mad)
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)
    atr = tr.rolling(ADX_PERIOD).mean()
    up_move = df["high"].diff()
    down_move = df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(ADX_PERIOD).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(ADX_PERIOD).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["ADX"] = dx.rolling(ADX_PERIOD).mean()
    df["BB_MA"] = df["close"].rolling(BB_PERIOD).mean()
    df["BB_STD"] = df["close"].rolling(BB_PERIOD).std()
    df["BB_upper"] = df["BB_MA"] + 2 * df["BB_STD"]
    df["BB_lower"] = df["BB_MA"] - 2 * df["BB_STD"]
    df["STOCH"] = ((df["close"] - df["low"].rolling(STOCH_PERIOD).min()) /
                   (df["high"].rolling(STOCH_PERIOD).max() - df["low"].rolling(STOCH_PERIOD).min())) * 100
    return df

def check_signal(df):
    if df["RSI"].iloc[-1] < 30 and df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]:
        return "CALL"
    elif df["RSI"].iloc[-1] > 70 and df["MACD"].iloc[-1] < df["MACD_Signal"].iloc[-1]:
        return "PUT"
    else:
        return None

def estimate_confidence(df):
    score = 0
    reasons = []

    if df["RSI"].iloc[-1] < 30 or df["RSI"].iloc[-1] > 70:
        score += 15
        reasons.append("RSI extrême")

    if (df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]) or (df["MACD"].iloc[-1] < df["MACD_Signal"].iloc[-1]):
        score += 15
        reasons.append("Croisement MACD")

    if df["ADX"].iloc[-1] > 20:
        score += 10
        reasons.append("Tendance forte (ADX)")

    if df["volume"].iloc[-1] > df["volume"].rolling(VOLUME_PERIOD).mean().iloc[-1]:
        score += 10
        reasons.append("Volume élevé")

    if df["CCI"].iloc[-1] > 100 or df["CCI"].iloc[-1] < -100:
        score += 10
        reasons.append("CCI extrême")

    if df["close"].iloc[-1] > df["EMA200"].iloc[-1] and df["EMA"].iloc[-1] > df["EMA200"].iloc[-1]:
        score += 10
        reasons.append("Au-dessus de l'EMA200")

    return min(score, 100), reasons

# --- COMMANDES TELEGRAM ---
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_running
    query = update.callback_query
    await query.answer()
    if query.data == "stop":
        is_running = False
        await query.edit_message_text("🛑 Bot arrêté avec succès.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("🛑 Stop", callback_data="stop")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📊 Bot démarré. Analyse des marchés en cours...",
        reply_markup=reply_markup
    )
    asyncio.create_task(safe_monitoring_loop())

# --- RÉSUMÉ JOURNALIER + TOP PAIRES VOLATILES ---
async def daily_summary():
    try:
        summary = "📅 *Résumé Journalier - Performances du Marché*\n\n"
        top_pairs = []
        for symbol in SYMBOLS:
            df = get_ohlcv(symbol, "15m")
            df = calculate_indicators(df)
            last_volatility = (df["high"] - df["low"]).mean()
            top_pairs.append((symbol, last_volatility))

        top_pairs.sort(key=lambda x: x[1], reverse=True)
        summary += "🔥 *Top paires les plus volatiles :*\n"
        for s, vol in top_pairs[:5]:
            summary += f"- `{s}` : Volatilité moyenne ~ {vol:.4f}\n"

        await app.bot.send_message(chat_id=CHAT_ID, text=summary, parse_mode="Markdown")
    except Exception as e:
        print(f"[ERREUR Résumé Journalier] {e}")

# --- MAIN ---
if __name__ == "__main__":
    async def main():
        scheduler = AsyncIOScheduler(timezone=TIMEZONE)
        scheduler.add_job(daily_summary, trigger='cron', hour=23, minute=59)
        scheduler.start()

        await app.initialize()
        await app.start()

        # Bouton STOP ajouté ici 👇
        keyboard = [[InlineKeyboardButton("🛑 Stop", callback_data="stop")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await app.bot.send_message(
            chat_id=CHAT_ID,
            text="✅ Bot lancé avec succès et prêt à analyser les marchés !",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

        app.add_handler(CommandHandler("start", start))
        app.add_handler(CallbackQueryHandler(button))
        app.add_handler(CommandHandler("set_threshold", set_threshold))

        await app.run_polling()

    asyncio.run(main())