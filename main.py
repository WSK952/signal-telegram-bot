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
INTERVAL = "1m"
RSI_PERIOD = 14
EMA_PERIOD = 9
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
TIMEZONE = pytz.timezone("Europe/Paris")

is_running = False
last_sent_signals = {}
app = Application.builder().token(TOKEN).build()

def get_ohlcv(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={INTERVAL}&limit=100"
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
    return df

def check_signal(df):
    rsi = df["RSI"].iloc[-1]
    ema = df["EMA"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_Signal"].iloc[-1]
    close = df["close"].iloc[-1]
    if rsi < 30 and close > ema and macd > macd_signal:
        return "CALL 📈"
    elif rsi > 70 and close < ema and macd < macd_signal:
        return "PUT 📉"
    return None

async def send_signal(pair, signal_type, df):
    rsi = df["RSI"].iloc[-1]
    ema = df["EMA"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_Signal"].iloc[-1]
    close = df["close"].iloc[-1]
    next_time = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)

    message = (
        f"🚨 *Signal détecté* : {signal_type}\n\n"
        f"📊 *Paire* : `{pair}`\n"
        f"🕒 *Place le trade à* : {next_time.strftime('%H:%M:%S')} (dans quelques minutes)\n\n"
        f"📉 *RSI* : {rsi:.2f}\n"
        f"📈 *EMA* : {ema:.2f}\n"
        f"📊 *MACD* : {macd:.4f}\n"
        f"⚙️ *MACD Signal* : {macd_signal:.4f}\n"
        f"💰 *Close* : {close:.2f}\n"
        f"🧠 *Fiabilité élevée*"
    )
    await app.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")

async def monitoring_loop():
    global is_running
    if is_running:
        return
    is_running = True

    last_summary = time.time()
    summary_interval = 900  # 15 minutes

    try:
        while is_running:
            all_results = []
            for symbol in SYMBOLS:
                try:
                    df = get_ohlcv(symbol)
                    df = calculate_indicators(df)
                    signal = check_signal(df)
                    if signal:
                        timestamp = df["timestamp"].iloc[-1].strftime('%Y-%m-%d %H:%M')
                        last_key = f"{symbol}_{signal}_{timestamp}"
                        if last_key != last_sent_signals.get(symbol):
                            await send_signal(symbol, signal, df)
                            last_sent_signals[symbol] = last_key
                    else:
                        rsi = df["RSI"].iloc[-1]
                        if rsi < 35:
                            all_results.append(f"⚠️ {symbol} : RSI {rsi:.2f} (potentiel CALL)")
                        elif rsi > 65:
                            all_results.append(f"⚠️ {symbol} : RSI {rsi:.2f} (potentiel PUT)")
                except Exception as e:
                    print(f"Erreur sur {symbol} :", e)

            if time.time() - last_summary > summary_interval:
                now = datetime.datetime.now(TIMEZONE).strftime("%H:%M")
                if all_results:
                    resume = f"📋 *Rapport d’analyse {now}*\n\n" + "\n".join(all_results)
                else:
                    resume = f"📋 *Rapport d’analyse {now}*\n\nAucun mouvement intéressant actuellement."
                await app.bot.send_message(chat_id=CHAT_ID, text=resume, parse_mode="Markdown")
                last_summary = time.time()

            await asyncio.sleep(60)
    finally:
        is_running = False

# --- Commandes Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("🛑 Stop", callback_data="stop")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("🟢 Bot démarré et en cours d’analyse des marchés.", reply_markup=reply_markup)
    await monitoring_loop()

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_running
    query = update.callback_query
    await query.answer()
    if query.data == "stop":
        is_running = False
        await query.edit_message_text("🛑 Bot arrêté avec succès.")

app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(button))

if __name__ == "__main__":
    app.run_polling()