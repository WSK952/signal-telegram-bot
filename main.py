import requests
import time
import datetime
import pytz
import pandas as pd
import numpy as np
from telegram import Bot

# === CONFIGURATION ===
TOKEN = "8450398342:AAEhPlH-lrECa2moq_4oSOKDjSmMpGmeaRA"
CHAT_ID = "@Signalwskbot"
TIMEZONE = pytz.timezone("Europe/Paris")
SYMBOLS = [
    "BTCUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "ETHUSDT",
    "DASHUSDT", "BCHUSDT", "FILUSDT", "LTCUSDT", "YFIUSDT", "ZECUSDT"
]
INTERVAL = "1m"
RSI_PERIOD = 14
EMA_PERIOD = 9
SIGNAL_DURATION = 60  # secondes
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

bot = Bot(token=TOKEN)

def get_klines(symbol, interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    return df[["timestamp", "close"]]

def calculate_indicators(df):
    df["EMA"] = df["close"].ewm(span=EMA_PERIOD).mean()

    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(RSI_PERIOD).mean()
    avg_loss = pd.Series(loss).rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df["close"].ewm(span=MACD_FAST).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=MACD_SIGNAL).mean()

    return df

def send_signal(symbol, rsi, ema, macd, macd_signal, close, action, timestamp):
    msg = (
        f"📢 Signal détecté : {action}\n"
        f"📊 Pair : {symbol}\n"
        f"📉 RSI : {rsi:.2f}\n"
        f"📈 EMA : {ema:.2f}\n"
        f"📊 MACD : {macd:.2f}\n"
        f"📊 MACD Signal : {macd_signal:.2f}\n"
        f"💰 Close : {close:.2f}\n"
        f"🕒 Heure : {timestamp.strftime('%H:%M:%S')}\n"
        f"📆 Durée : {SIGNAL_DURATION}s"
    )
    bot.send_message(chat_id=CHAT_ID, text=msg)

def run_bot():
    while True:
        now = datetime.datetime.now(TIMEZONE)
        for symbol in SYMBOLS:
            try:
                df = get_klines(symbol, INTERVAL)
                df = calculate_indicators(df)
                latest = df.iloc[-1]
                rsi = latest["RSI"]
                ema = latest["EMA"]
                macd = latest["MACD"]
                macd_signal = latest["MACD_SIGNAL"]
                close = latest["close"]
                action = None

                # CONDITIONS FIABLES
                if rsi < 30 and macd > macd_signal and close > ema:
                    action = "CALL 📈"
                elif rsi > 70 and macd < macd_signal and close < ema:
                    action = "PUT 📉"

                if action:
                    send_signal(symbol, rsi, ema, macd, macd_signal, close, action, now)

            except Exception as e:
                print(f"Erreur avec {symbol} : {e}")

        time.sleep(SIGNAL_DURATION)

if __name__ == "__main__":
    run_bot()
