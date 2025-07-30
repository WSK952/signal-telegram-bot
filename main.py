
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
RSI_PERIOD = 14
EMA_PERIOD = 9
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
TIMEZONE = pytz.timezone("Europe/Paris")

is_running = False
last_sent_signals = {}
app = Application.builder().token(TOKEN).build()

def get_ohlcv(symbol, interval):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms').dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    return df[["timestamp", "close"]]

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
    df["CCI"] = (df["close"] - df["close"].rolling(20).mean()) / (0.015 * df["close"].rolling(20).std())
    df["STOCH"] = ((df["close"] - df["close"].rolling(14).min()) / (df["close"].rolling(14).max() - df["close"].rolling(14).min())) * 100
    df["ADX"] = df["close"].diff().abs().rolling(14).mean()
    df["BB_MIDDLE"] = df["close"].rolling(20).mean()
    df["BB_UPPER"] = df["BB_MIDDLE"] + 2 * df["close"].rolling(20).std()
    df["BB_LOWER"] = df["BB_MIDDLE"] - 2 * df["close"].rolling(20).std()
    return df

def estimate_confidence(indicators):
    score = 0
    if indicators["RSI"] < 30 or indicators["RSI"] > 70:
        score += 1
    if (indicators["MACD"] - indicators["MACD_Signal"]) > 0:
        score += 1
    if indicators["CCI"] < -100 or indicators["CCI"] > 100:
        score += 1
    if indicators["STOCH"] < 20 or indicators["STOCH"] > 80:
        score += 1
    if indicators["close"] > indicators["EMA"]:
        score += 1
    if indicators["close"] > indicators["BB_UPPER"] or indicators["close"] < indicators["BB_LOWER"]:
        score += 1
    return int((score / 6) * 100)

def check_signal(multiframe_data):
    call_conditions = 0
    put_conditions = 0
    for df in multiframe_data.values():
        indicators = {
            "RSI": df["RSI"].iloc[-1],
            "EMA": df["EMA"].iloc[-1],
            "MACD": df["MACD"].iloc[-1],
            "MACD_Signal": df["MACD_Signal"].iloc[-1],
            "CCI": df["CCI"].iloc[-1],
            "STOCH": df["STOCH"].iloc[-1],
            "close": df["close"].iloc[-1],
            "BB_UPPER": df["BB_UPPER"].iloc[-1],
            "BB_LOWER": df["BB_LOWER"].iloc[-1]
        }
        if indicators["RSI"] < 30 and indicators["MACD"] > indicators["MACD_Signal"] and indicators["close"] > indicators["EMA"]:
            call_conditions += 1
        elif indicators["RSI"] > 70 and indicators["MACD"] < indicators["MACD_Signal"] and indicators["close"] < indicators["EMA"]:
            put_conditions += 1

    if call_conditions >= 2:
        return "CALL ð", indicators
    elif put_conditions >= 2:
        return "PUT ð", indicators
    return None, None

async def send_signal(pair, signal_type, indicators, timestamp):
    confidence = estimate_confidence(indicators)
    message = (
        f"ð¨ *Signal dÃ©tectÃ©* : {signal_type}

"
        f"ð *Paire* : `{pair}`
"
        f"ð *Place le trade Ã * : {timestamp.strftime('%H:%M:%S')} (dans quelques minutes)

"
        f"ð *RSI* : {indicators['RSI']:.2f}
"
        f"ð *EMA* : {indicators['EMA']:.2f}
"
        f"ð *MACD* : {indicators['MACD']:.4f}
"
        f"âï¸ *MACD Signal* : {indicators['MACD_Signal']:.4f}
"
        f"ð° *Close* : {indicators['close']:.2f}
"
        f"ð¯ *Taux de fiabilitÃ© estimÃ©* : {confidence}%"
    )
    await app.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")

async def monitoring_loop():
    global is_running
    if is_running:
        return
    is_running = True
    await app.bot.send_message(chat_id=CHAT_ID, text="ð¢ Bot dÃ©marrÃ© et en cours dâanalyse des marchÃ©s.")

    try:
        while is_running:
            for symbol in SYMBOLS:
                try:
                    multiframe_data = {}
                    for interval in INTERVALS:
                        df = get_ohlcv(symbol, interval)
                        df = calculate_indicators(df)
                        multiframe_data[interval] = df
                    signal, indicators = check_signal(multiframe_data)
                    if signal:
                        timestamp = list(multiframe_data.values())[0]["timestamp"].iloc[-1]
                        last_key = f"{symbol}_{signal}_{timestamp}"
                        if last_key != last_sent_signals.get(symbol):
                            await send_signal(symbol, signal, indicators, timestamp)
                            last_sent_signals[symbol] = last_key
                except Exception as e:
                    print(f"Erreur sur {symbol} :", e)
            await asyncio.sleep(60)
    finally:
        is_running = False

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
