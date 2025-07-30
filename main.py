import requests
import time
import datetime
import pytz
import pandas as pd
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio

# --- CONFIGURATION ---
TOKEN = "8450398342:AAEhPlH-lrECa2moq_4oSOKDjSmMpGmeaRA"
CHAT_ID = "1091559539"
SYMBOLS = ["BTCUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "ETHUSDT", "DASHUSDT", "BCHUSDT", "FILUSDT", "LTCUSDT", "YFIUSDT", "ZECUSDT"]
TIMEZONE = pytz.timezone("Europe/Paris")
INTERVALS = ["1m", "5m", "15m"]

RSI_PERIOD = 14
EMA_PERIOD = 9
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
CCI_PERIOD = 20

app = Application.builder().token(TOKEN).build()
is_running = False
last_sent_signals = {}
active_signals = {}

# --- INDICATEURS ---
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
    return int(score * 25)

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

# --- MESSAGES ---
async def send_main_signal(symbol, signal, df, confidence):
    close = df["close"].iloc[-1]
    execution_time = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)
    active_signals[symbol] = {
        "type": signal,
        "price": close,
        "time": execution_time,
        "initial_confidence": confidence
    }

    message = (
        f"ð¨ *Signal dÃ©tectÃ©* : {signal}

"
        f"ð *Paire* : `{symbol}`
"
        f"ð *Heure prÃ©vue du trade* : {execution_time.strftime('%H:%M:%S')}

"
        f"ð RSI : {df['RSI'].iloc[-1]:.2f}
"
        f"ð EMA : {df['EMA'].iloc[-1]:.2f}
"
        f"ð MACD : {df['MACD'].iloc[-1]:.4f}
"
        f"âï¸ MACD Signal : {df['MACD_Signal'].iloc[-1]:.4f}
"
        f"ð CCI : {df['CCI'].iloc[-1]:.2f}
"
        f"ð ADX : {df['ADX'].iloc[-1]:.2f}

"
        f"ð§  *FiabilitÃ© estimÃ©e* : {confidence}%"
    )
    await app.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")

async def send_m1_confirmation(symbol):
    data = active_signals[symbol]
    df = get_ohlcv(symbol, "1m")
    df = calculate_indicators(df)
    confidence = estimate_confidence(df)
    note = "â *M1 CONFIRME LE SIGNAL*" if confidence >= data["initial_confidence"] else "â ï¸ *M1 NE CONFIRME PAS LE SIGNAL*"
    conseil = "â Je te conseille de suivre ce signal." if confidence >= 75 else "â Je te dÃ©conseille de suivre ce signal."
    delta = confidence - data["initial_confidence"]

    message = (
        f"ð *Mise Ã  jour M1 pour* `{symbol}`
"
        f"{note}
"
        f"ð¯ Nouvelle fiabilitÃ© : {confidence}% ({'+' if delta >= 0 else ''}{delta}%)
"
        f"ð¬ *Conseil* : {conseil}"
    )
    await app.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")

async def send_trade_result(symbol):
    data = active_signals[symbol]
    df = get_ohlcv(symbol, "1m")
    last_close = df["close"].iloc[-1]
    result = "â *Signal gagnant*" if (
        (data["type"].startswith("CALL") and last_close > data["price"]) or
        (data["type"].startswith("PUT") and last_close < data["price"])
    ) else "â *Signal perdant*"
    await app.bot.send_message(chat_id=CHAT_ID, text=f"ð RÃ©sultat pour `{symbol}` : {result}", parse_mode="Markdown")
    del active_signals[symbol]

# --- MONITORING ---
async def monitoring_loop():
    global is_running
    if is_running:
        return
    is_running = True
    await app.bot.send_message(chat_id=CHAT_ID, text="â Bot lancÃ© et en cours dâanalyse.")

    try:
        while is_running:
            now = datetime.datetime.now(TIMEZONE)
            for symbol in SYMBOLS:
                try:
                    df_m15 = calculate_indicators(get_ohlcv(symbol, "15m"))
                    df_m5 = calculate_indicators(get_ohlcv(symbol, "5m"))
                    signal_m15 = check_signal(df_m15)
                    signal_m5 = check_signal(df_m5)
                    if signal_m15 and signal_m15 == signal_m5:
                        timestamp = df_m15["timestamp"].iloc[-1].strftime('%Y-%m-%d %H:%M')
                        last_key = f"{symbol}_{signal_m15}_{timestamp}"
                        if last_key != last_sent_signals.get(symbol):
                            confidence = int((estimate_confidence(df_m15) + estimate_confidence(df_m5)) / 2)
                            await send_main_signal(symbol, signal_m15, df_m15, confidence)
                            last_sent_signals[symbol] = last_key

                    if symbol in active_signals:
                        time_diff = (active_signals[symbol]["time"] - now).total_seconds()
                        if 50 <= time_diff <= 70:
                            await send_m1_confirmation(symbol)
                        elif now >= active_signals[symbol]["time"] + pd.Timedelta(minutes=1):
                            await send_trade_result(symbol)
                except Exception as e:
                    print(f"Erreur {symbol} :", e)
            await asyncio.sleep(30)
    finally:
        is_running = False

# --- COMMANDES TELEGRAM ---
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
