import logging
import asyncio
import datetime
from telegram import Bot
from telegram.ext import Application, CommandHandler
import requests
import pandas as pd
import numpy as np
import os

# Configuration
TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # à remplir dans Railway
PAIR_LIST = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LINKUSDT', 'DASHUSDT', 'BCHUSDT', 'FILUSDT', 'LTCUSDT', 'YFIUSDT', 'ZECUSDT']
INTERVAL = '1m'
LIMIT = 100

# Log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Récupération des données OHLCV depuis Binance
def get_ohlcv(pair):
    url = f'https://api.binance.com/api/v3/klines?symbol={pair}&interval={INTERVAL}&limit={LIMIT}'
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','close_time','quote_asset_volume','num_trades','taker_buy_base','taker_buy_quote','ignore'])
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        return df
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données : {e}")
        return None

# Calcul des indicateurs RSI, EMA et MACD
def compute_indicators(df):
    df['ema'] = df['close'].ewm(span=14).mean()
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9).mean()
    return df

# Vérifie si un signal fiable existe
def check_signal(df):
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    if last['rsi'] < 30 and last['macd'] > last['signal']:
        return "CALL"
    elif last['rsi'] > 70 and last['macd'] < last['signal']:
        return "PUT"
    return None

# Envoie le message Telegram
async def send_signal(application):
    while True:
        for pair in PAIR_LIST:
            df = get_ohlcv(pair)
            if df is not None:
                df = compute_indicators(df)
                signal = check_signal(df)
                if signal:
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    msg = f"✅ Signal {signal} détecté pour {pair} à {now}"
                    await application.bot.send_message(chat_id=CHAT_ID, text=msg)
        await asyncio.sleep(60)

# Commande de test
async def start(update, context):
    await update.message.reply_text("🤖 Bot opérationnel.")

# Lancement du bot
async def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    asyncio.create_task(send_signal(application))
    await application.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
