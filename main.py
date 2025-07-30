import requests
import time
import datetime
import pytz
import pandas as pd
import numpy as np
from telegram import Bot
import os

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

# --- Binance OHLCV ---
def get_ohlcv(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={INTERVAL}&limit=100"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms').dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    return df[["timestamp", "close"]]

# --- Indicateurs ---
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

# --- Signal ---
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
    else:
        return None

# --- Envoyer signal ---
def send_signal(pair, signal_type, df):
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

    bot.send_message(chat_id=CHAT_ID, text=message)

# --- Boucle principale ---
def run():
    bot.send_message(chat_id=CHAT_ID, text="✅ Bot de signaux lancé avec succès !")
    while True:
        for symbol in SYMBOLS:
            try:
                df = get_ohlcv(symbol)
                df = calculate_indicators(df)
                signal = check_signal(df)
                if signal:
                    send_signal(symbol, signal, df)
            except Exception as e:
                print(f"Erreur sur {symbol} :", e)
        time.sleep(60)

import asyncio

async def main():
    await bot.send_message(chat_id=CHAT_ID, text="✅ Bot de signaux lancé avec succès !")
    run()

if __name__ == "__main__":
    asyncio.run(main())