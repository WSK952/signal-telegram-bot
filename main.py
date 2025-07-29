import requests
import time
import datetime
import pytz
import pandas as pd
import logging
from telegram import Bot
import os

# CONFIGURATION
TOKEN = "8450398342:AAEhPlH-lrECa2moq_4oSOKDjSmMpGmeaRA"
CHAT_ID = "@Signalwskbot"
SYMBOLS = ["ETH/USDT", "BTC/USDT", "BNB/USDT", "XRP/USDT"]
RSI_PERIOD = 14
EMA_PERIOD = 9
SIGNAL_DURATION = 60  # secondes
TIMEZONE = pytz.timezone("Europe/Paris")

# Initialisation
bot = Bot(token=TOKEN)
log_file = os.path.join(os.path.dirname(__file__), "signals_history.csv")

def get_ohlcv(symbol):
    now = datetime.datetime.now(TIMEZONE)
    timestamps = [now - datetime.timedelta(minutes=i) for i in range(100)][::-1]
    prices = [100 + i*0.1 for i in range(100)]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": prices
    })
    return df

def calculate_indicators(df):
    df["EMA"] = df["close"].ewm(span=EMA_PERIOD).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def send_signal(pair, rsi, ema, close, action, timestamp):
    message = (
        f"📢 Signal détecté : {action}\n"
        f"🌐 Pair : {pair}\n"
        f"📉 RSI : {rsi:.2f}\n"
        f"📈 EMA : {ema:.2f}\n"
        f"💰 Close : {close:.2f}\n"
        f"🕒 Heure : {timestamp.strftime('%H:%M:%S')}\n"
        f"📆 Durée : {SIGNAL_DURATION}s"
    )
    bot.send_message(chat_id=CHAT_ID, text=message)
    log_result(pair, action, rsi, ema, close, timestamp)

def log_result(pair, action, rsi, ema, close, timestamp):
    result = {
        "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "pair": pair,
        "rsi": round(rsi, 2),
        "ema": round(ema, 2),
        "close": round(close, 2),
        "signal": action
    }
    df = pd.DataFrame([result])
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, mode='a', header=False, index=False)

def run_bot():
    while True:
        now = datetime.datetime.now(TIMEZONE)
        for pair in SYMBOLS:
            df = get_ohlcv(pair)
            df = calculate_indicators(df)
            rsi = df["RSI"].iloc[-1]
            ema = df["EMA"].iloc[-1]
            close = df["close"].iloc[-1]
            action = None
            if rsi < 30 and close > ema:
                action = "CALL 📈"
            elif rsi > 70 and close < ema:
                action = "PUT 📉"
            if action:
                send_signal(pair, rsi, ema, close, action, now)
        time.sleep(60)

if __name__ == "__main__":
    run_bot()
