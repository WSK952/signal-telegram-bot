import requests
import time
import datetime
import pytz
import pandas as pd
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio

# --- CONFIG ---
TOKEN = "8450398342:AAEhPlH-lrECa2moq_4oSOKDjSmMpGmeaRA"
CHAT_ID = "1091559539"
SYMBOLS = ["BTCUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "ETHUSDT", "DASHUSDT", "BCHUSDT", "FILUSDT", "LTCUSDT", "YFIUSDT", "ZECUSDT"]
INTERVALS = ["1m", "5m", "15m"]
TIMEZONE = pytz.timezone("Europe/Paris")

RSI_PERIOD = 14
EMA_PERIOD = 9
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
CCI_PERIOD = 20

is_running = False
last_sent_signals = {}
active_signals = {}
app = Application.builder().token(TOKEN).build()

def get_ohlcv(symbol, interval):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms').dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    return df[["timestamp", "close", "high", "low"]]

def calculate_indicators(df):
    df["EMA"] = df["close"].ewm(span=EMA_PERIOD).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    exp1 = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["CCI"] = (df["close"] - (df["high"] + df["low"] + df["close"]) / 3).rolling(CCI_PERIOD).mean()
    df["ADX"] = abs(df["high"] - df["low"]).rolling(ADX_PERIOD).mean()
    return df

def estimate_confidence(df):
    score = 0
    if df["RSI"].iloc[-1] < 30 or df["RSI"].iloc[-1] > 70:
        score += 1
    if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]:
        score += 1
    if df["CCI"].iloc[-1] > 100 or df["CCI"].iloc[-1] < -100:
        score += 1
    if df["ADX"].iloc[-1] > 25:
        score += 1
    return int(score * 25)  # Sur 100%

def check_signal(df):
    rsi = df["RSI"].iloc[-1]
    ema = df["EMA"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_Signal"].iloc[-1]
    close = df["close"].iloc[-1]
    if rsi < 30 and close > ema and macd > macd_signal:
        return "CALL ð"
    elif rsi > 70 and close < ema and macd < macd_signal:
        return "PUT ð"
    return None

async def send_signal(pair, signal_type, df):
    confidence = estimate_confidence(df)
    next_time = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)
    close = df["close"].iloc[-1]
    active_signals[pair] = {"type": signal_type, "price": close, "time": next_time}

    message = (
        f"ð¨ *Signal dÃ©tectÃ©* : {signal_type}
"
        f"ð *Paire* : `{pair}`
"
        f"ð *Trade Ã  placer vers* : {next_time.strftime('%H:%M:%S')}
"
        f"ð RSI : {df['RSI'].iloc[-1]:.2f}
"
        f"ð EMA : {df['EMA'].iloc[-1]:.2f}
"
        f"ð MACD : {df['MACD'].iloc[-1]:.4f}
"
        f"âï¸ MACD Signal : {df['MACD_Signal'].iloc[-1]:.4f}
"
        f"ð CCI : {df['CCI'].iloc[-1]:.2f}
"
        f"ð ADX : {df['ADX'].iloc[-1]:.2f}
"
        f"ð *Taux de fiabilitÃ© estimÃ©* : {confidence}%"
    )
    await app.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")

async def monitoring_loop():
    global is_running
    if is_running:
        return
    is_running = True

    await app.bot.send_message(chat_id=CHAT_ID, text="ð¢ Bot lancÃ© et en cours dâanalyse.")

    try:
        while is_running:
            for symbol in SYMBOLS:
                try:
                    combined_df = None
                    for interval in INTERVALS:
                        df = get_ohlcv(symbol, interval)
                        df = calculate_indicators(df)
                        if interval == "1m":
                            signal = check_signal(df)
                            if signal:
                                timestamp = df["timestamp"].iloc[-1].strftime('%Y-%m-%d %H:%M')
                                last_key = f"{symbol}_{signal}_{timestamp}"
                                if last_key != last_sent_signals.get(symbol):
                                    await send_signal(symbol, signal, df)
                                    last_sent_signals[symbol] = last_key
                        if combined_df is None:
                            combined_df = df
                except Exception as e:
                    print(f"Erreur pour {symbol} :", e)

            # VÃ©rifie les rÃ©sultats des signaux prÃ©cÃ©dents
            for pair, data in list(active_signals.items()):
                now = datetime.datetime.now(TIMEZONE)
                if now >= data["time"] + pd.Timedelta(minutes=1):
                    df = get_ohlcv(pair, "1m")
                    last_close = df["close"].iloc[-1]
                    result = "â *Signal gagnant*" if (
                        (data["type"].startswith("CALL") and last_close > data["price"]) or
                        (data["type"].startswith("PUT") and last_close < data["price"])
                    ) else "â *Signal perdant*"
                    await app.bot.send_message(chat_id=CHAT_ID, text=f"ð RÃ©sultat du signal {pair} : {result}", parse_mode="Markdown")
                    del active_signals[pair]

            await asyncio.sleep(60)
    finally:
        is_running = False

# Commandes Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("ð Stop", callback_data="stop")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("ð¢ Bot dÃ©marrÃ© et en cours dâanalyse des marchÃ©s.", reply_markup=reply_markup)
    await monitoring_loop()

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_running
    query = update.callback_query
    await query.answer()
    if query.data == "stop":
        is_running = False
        await query.edit_message_text("ð Bot arrÃªtÃ© avec succÃ¨s.")

app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(button))

if __name__ == "__main__":
    app.run_polling()