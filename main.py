import requests
import time
import datetime
import pytz
import pandas as pd
from telegram import Bot
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
bot = Bot(token=TOKEN)

# Flag pour éviter les doubles boucles
is_running = False

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
    timestamp = df["timestamp"].iloc[-1].strftime("%H:%M:%S")
    rsi = df["RSI"].iloc[-1]
    ema = df["EMA"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_Signal"].iloc[-1]
    close = df["close"].iloc[-1]
    message = f"""📢 Signal détecté : {signal_type}
🌐 Pair : {pair}
📉 RSI : {rsi:.2f}
📈 EMA : {ema:.2f}
📊 MACD : {macd:.4f} / Signal : {macd_signal:.4f}
💰 Close : {close:.2f}
🕒 Heure : {timestamp}
📆 Durée : 60s"""
    await bot.send_message(chat_id=CHAT_ID, text=message)

async def monitoring_loop():
    global is_running
    if is_running:
        print("⛔️ Boucle déjà active. Ignorée.")
        return
    is_running = True

    await bot.send_message(chat_id=CHAT_ID, text="✅ Bot lancé avec succès et prêt à analyser les marchés !")
    last_summary = time.time()
    summary_interval = 600  # 10 minutes

    try:
        while True:
            all_results = []
            for symbol in SYMBOLS:
                try:
                    df = get_ohlcv(symbol)
                    df = calculate_indicators(df)
                    signal = check_signal(df)
                    if signal:
                        await send_signal(symbol, signal, df)
                    else:
                        rsi = df["RSI"].iloc[-1]
                        if rsi < 35:
                            all_results.append(f"⚠️ {symbol} : RSI {rsi:.2f} (possible CALL en approche)")
                        elif rsi > 65:
                            all_results.append(f"⚠️ {symbol} : RSI {rsi:.2f} (possible PUT en approche)")
                except Exception as e:
                    print(f"Erreur sur {symbol} :", e)

            if time.time() - last_summary > summary_interval:
                now = datetime.datetime.now(TIMEZONE).strftime("%H:%M")
                if all_results:
                    resume = f"📊 Rapport d’analyse {now}\n\n" + "\n".join(all_results)
                else:
                    resume = f"📊 Rapport d’analyse {now}\n\nAucun mouvement intéressant détecté sur les paires surveillées."
                await bot.send_message(chat_id=CHAT_ID, text=resume)
                last_summary = time.time()

            await asyncio.sleep(60)
    finally:
        is_running = False

if __name__ == "__main__":
    asyncio.run(monitoring_loop())