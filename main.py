# ✅ main.py - Bot de signaux amélioré (v31.07.2025-00h33)
import requests
import time
import datetime
import pytz
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio

# CONFIG
TOKEN = "8450398342:AAEhPlH-lrECa2moq_4oSOKDjSmMpGmeaRA"
CHAT_ID = "1091559539"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT"]
INTERVALS = ["1m", "5m", "10m", "15m"]
TIMEZONE = pytz.timezone("Europe/Paris")
RSI_PERIOD = 14
EMA_PERIOD = 9
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

is_running = False
last_sent_signals = {}
tracked_signals = {}
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
    return df

def check_signal_mtf(dataframes):
    votes = {"CALL": 0, "PUT": 0}
    for df in dataframes:
        rsi = df["RSI"].iloc[-1]
        ema = df["EMA"].iloc[-1]
        macd = df["MACD"].iloc[-1]
        macd_signal = df["MACD_Signal"].iloc[-1]
        close = df["close"].iloc[-1]
        if rsi < 30 and close > ema and macd > macd_signal:
            votes["CALL"] += 1
        elif rsi > 70 and close < ema and macd < macd_signal:
            votes["PUT"] += 1
    if votes["CALL"] >= 3:
        return "CALL 📈", int((votes["CALL"] / len(dataframes)) * 100)
    elif votes["PUT"] >= 3:
        return "PUT 📉", int((votes["PUT"] / len(dataframes)) * 100)
    return None, 0

async def send_signal(pair, signal_type, confidence, df):
    close = df["close"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    ema = df["EMA"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_Signal"].iloc[-1]
    current_time = df["timestamp"].iloc[-1]

    message = (
        f"🚨 *Signal détecté* : {signal_type}\n"
        f"📊 *Paire* : `{pair}`\n"
        f"🕒 *Heure de détection* : {current_time.strftime('%H:%M:%S')}\n"
        f"✅ *Fiabilité estimée* : {confidence}%\n\n"
        f"📉 RSI : {rsi:.2f}\n"
        f"📈 EMA : {ema:.2f}\n"
        f"📊 MACD : {macd:.4f}\n"
        f"⚙️ MACD Signal : {macd_signal:.4f}\n"
        f"💰 Close : {close:.2f}\n"
    )
    await app.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
    tracked_signals[pair] = {"time": current_time, "type": signal_type, "price": close}

async def verify_signals():
    for symbol, info in list(tracked_signals.items()):
        try:
            df = get_ohlcv(symbol, "1m")
            close = df["close"].iloc[-1]
            entry = info["price"]
            result = "✅ Signal VALIDÉ" if (
                info["type"].startswith("CALL") and close > entry
                or info["type"].startswith("PUT") and close < entry
            ) else "❌ Signal NON VALIDÉ"
            msg = f"{result} sur `{symbol}` ({info['type']})\nPrix entrée : {entry} | Prix actuel : {close:.2f}"
            await app.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        except:
            continue
        del tracked_signals[symbol]

async def monitoring_loop():
    global is_running
    if is_running: return
    is_running = True
    await app.bot.send_message(chat_id=CHAT_ID, text="🟢 Bot actif. Analyse multi-timeframe en cours...")

    last_summary = time.time()
    summary_interval = 900  # 15 min

    try:
        while is_running:
            await verify_signals()
            all_results = []
            for symbol in SYMBOLS:
                try:
                    dfs = [calculate_indicators(get_ohlcv(symbol, interval)) for interval in INTERVALS]
                    signal, confidence = check_signal_mtf(dfs)
                    if signal and confidence >= 75:
                        key = f"{symbol}_{signal}_{dfs[0]['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}"
                        if key != last_sent_signals.get(symbol):
                            await send_signal(symbol, signal, confidence, dfs[0])
                            last_sent_signals[symbol] = key
                    else:
                        rsi = dfs[0]["RSI"].iloc[-1]
                        if rsi < 35:
                            all_results.append(f"⚠️ {symbol} RSI {rsi:.2f} (potentiel CALL)")
                        elif rsi > 65:
                            all_results.append(f"⚠️ {symbol} RSI {rsi:.2f} (potentiel PUT)")
                except Exception as e:
                    print(f"[ERREUR] {symbol} ->", e)

            if time.time() - last_summary > summary_interval:
                now = datetime.datetime.now(TIMEZONE).strftime("%H:%M")
                msg = "\n".join(all_results) if all_results else "Aucun mouvement intéressant."
                await app.bot.send_message(chat_id=CHAT_ID, text=f"📋 *Rapport {now}*\n\n{msg}", parse_mode="Markdown")
                last_summary = time.time()

            await asyncio.sleep(60)
    finally:
        is_running = False

# --- Commandes Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("🛑 Stop", callback_data="stop")]]
    await update.message.reply_text("✅ Bot lancé avec analyse multi-timeframe (1m/5m/10m/15m)", reply_markup=InlineKeyboardMarkup(keyboard))
    await monitoring_loop()

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_running
    query = update.callback_query
    await query.answer()
    if query.data == "stop":
        is_running = False
        await query.edit_message_text("🛑 Bot arrêté.")

app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(button))

if __name__ == "__main__":
    app.run_polling()