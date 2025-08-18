# --- IMPORTATIONS & CONFIGURATION GLOBALE ---
import os
import pytz
import asyncio
import logging
import datetime as dt
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- SUIVI JOURNALIER DES PERFORMANCES ---
from collections import defaultdict

daily_stats = {
    "total_signals": 0,
    "wins": 0,
    "losses": 0,
    "directions": [],
}
# --- PARAMÈTRES DU BOT TELEGRAM ---
TOKEN = "8450398342:AAEhPlH-lrECa2moq_4oSOKDjSmMpGmeaRA"
CHAT_ID = "1091559539"
TIMEZONE = pytz.timezone("Europe/Paris")

# --- PARAMÈTRES DE L'ANALYSE ---
PAIR = "ETHUSDT"
INTERVALS = ["1m", "5m"]  # Analyse sur M1 + M5
LIMIT = 200  # Nombre de bougies à récupérer

# --- INITIALISATION DES LOGS ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# --- INITIALISATION DU BOT TELEGRAM ---
application = Application.builder().token(TOKEN).build()

# --- SUIVI JOURNALIER DES PERFORMANCES ---
from collections import defaultdict

daily_stats = {
    "total_signals": 0,
    "wins": 0,
    "losses": 0,
    "directions": [],
}
# --- 📈 RÉCUPÉRATION DES DONNÉES DE MARCHÉ (Binance) ---
def get_ohlcv(pair="ETHUSDT", interval="1m", limit=200):
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": pair,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        return df

    except Exception as e:
        print(f"Erreur lors du téléchargement des données : {e}")
        return pd.DataFrame()

# --- 🔧 CALCUL DES INDICATEURS TECHNIQUES ---
def calculate_indicators(df):
    if df.empty:
        return df

    df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["BB_MA"] = df["close"].rolling(window=20).mean()
    df["BB_STD"] = df["close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_MA"] + 2 * df["BB_STD"]
    df["BB_lower"] = df["BB_MA"] - 2 * df["BB_STD"]

    stoch_rsi_period = 14
    rsi_min = df["RSI"].rolling(stoch_rsi_period).min()
    rsi_max = df["RSI"].rolling(stoch_rsi_period).max()
    df["StochRSI"] = 100 * (df["RSI"] - rsi_min) / (rsi_max - rsi_min)

    return df

# --- 🧠 DÉTECTION DU SIGNAL TRADING AVEC SCORE DE FIABILITÉ ---
def detect_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    confirmations = []
    explanations = []

    # RSI
    if last["RSI"] < 30:
        confirmations.append("CALL")
        explanations.append("RSI < 30 (survendu)")
    elif last["RSI"] > 70:
        confirmations.append("PUT")
        explanations.append("RSI > 70 (suracheté)")

    # MACD
    if last["MACD"] > last["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        confirmations.append("CALL")
        explanations.append("MACD croisement haussier")
    elif last["MACD"] < last["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        confirmations.append("PUT")
        explanations.append("MACD croisement baissier")

    # EMA Trend
    if last["EMA9"] > last["EMA200"]:
        confirmations.append("CALL")
        explanations.append("Tendance haussière (EMA9 > EMA200)")
    elif last["EMA9"] < last["EMA200"]:
        confirmations.append("PUT")
        explanations.append("Tendance baissière (EMA9 < EMA200)")

    # Stochastic RSI
    if last["StochRSI"] < 20:
        confirmations.append("CALL")
        explanations.append("StochRSI < 20")
    elif last["StochRSI"] > 80:
        confirmations.append("PUT")
        explanations.append("StochRSI > 80")

    # Résultat global
    call_count = confirmations.count("CALL")
    put_count = confirmations.count("PUT")

    if call_count > put_count:
        signal = "CALL"
    elif put_count > call_count:
        signal = "PUT"
    else:
        signal = None

    # Fiabilité
    total = len(explanations)
    confirm = max(call_count, put_count)
    score = round((confirm / total) * 100) if total > 0 else 0

    return signal, score, explanations

# --- 📤 ENVOI DU SIGNAL FORMATÉ SUR TELEGRAM ---
async def send_signal_telegram(signal, confidence, explanations):
    now = dt.datetime.now(TIMEZONE)
    trade_time = (now + dt.timedelta(minutes=1)).strftime("%H:%M")
    context_text = "\n".join([f"🔹 {e}" for e in explanations])

    message = f"""
📊 *NOUVEAU SIGNAL - ETH/USDT*
━━━━━━━━━━━━━━━━━━━━━━━
🕐 *Heure de trade* : {trade_time}
📈 *Direction* : {signal}
⏱ *Durée* : 60 secondes
✅ *Fiabilité* : {confidence}%

📌 *Analyse technique :*
{context_text}
━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Place l’ordre *quelques secondes avant {trade_time} pile*.
    """.strip()

    keyboard = [[InlineKeyboardButton("🛑 STOP", callback_data="stop")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await application.bot.send_message(
        chat_id=CHAT_ID,
        text=message,
        parse_mode="Markdown",
        reply_markup=reply_markup
    )

# --- ✅ RÉSULTAT DU TRADE APRÈS 60 SECONDES ---
async def send_result_after_trade(signal, entry_price):
    await asyncio.sleep(60)

    latest_data = get_ohlcv(PAIR, "1m", limit=1)
    if latest_data.empty:
        return

    exit_price = latest_data["close"].iloc[-1]
    result = None

    if signal == "CALL":
        result = "✅ GAGNÉ" if exit_price > entry_price else "❌ PERDU"
    elif signal == "PUT":
        result = "✅ GAGNÉ" if exit_price < entry_price else "❌ PERDU"

    variation = round((exit_price - entry_price) / entry_price * 100, 3)
    direction = "↗️" if variation > 0 else "↘️"

    reason = ""
    if result == "❌ PERDU":
        reason = "🔎 *Le marché s’est retourné après le signal.*\nPeut-être une mèche ou un volume trop faible."
    elif result == "✅ GAGNÉ":
        reason = "📈 *Le signal a bien été confirmé par le marché.*"

    msg = f"""
🎯 *RÉSULTAT DU SIGNAL*
━━━━━━━━━━━━━━━━━━━━━━━
📊 Résultat : {result}
📈 Direction : {signal}
💸 Entrée : `{entry_price:.4f}` | Sortie : `{exit_price:.4f}` {direction}
📊 Variation : `{variation}%`

📌 *Analyse rapide :*
{reason}
    """.strip()

    await application.bot.send_message(
        chat_id=CHAT_ID,
        text=msg,
        parse_mode="Markdown"
    ))

    # Mémorisation du résultat
    daily_stats["total_signals"] += 1
    if result == "✅ GAGNÉ":
        daily_stats["wins"] += 1
    elif result == "❌ PERDU":
        daily_stats["losses"] += 1
    daily_stats["directions"].append(signal)

    await application.bot.send_message(
        chat_id=CHAT_ID,
        text=msg,
        parse_mode="Markdown"
    )
# --- 🔄 ANALYSE AUTOMATIQUE & ENVOI DE SIGNAL ---
async def monitor_market():
    while True:
        try:
            df = get_ohlcv(PAIR, "1m", LIMIT)
            if df.empty:
                await asyncio.sleep(15)
                continue

            df = calculate_indicators(df)
            signal, confidence, explanations = detect_signal(df)

            if signal and confidence >= 20:
                await send_signal_telegram(signal, confidence, explanations)
                entry_price = df["close"].iloc[-1]
                await send_result_after_trade(signal, entry_price)
            else:
                now = dt.datetime.now(TIMEZONE).strftime("%H:%M:%S")
                print(f"[{now}] Aucun signal fiable détecté.")

        except Exception as e:
            print(f"Erreur dans monitor_market() : {e}")

        await asyncio.sleep(15)
    # Mémorisation du résultat
    daily_stats["total_signals"] += 1
    if result == "✅ GAGNÉ":
        daily_stats["wins"] += 1
    elif result == "❌ PERDU":
        daily_stats["losses"] += 1
    daily_stats["directions"].append(signal)

    await application.bot.send_message(
        chat_id=CHAT_ID,
        text=msg,
        parse_mode="Markdown"
    )
# --- 🕐 MESSAGE DE RAPPORT PÉRIODIQUE SANS SIGNAL ---
async def send_no_signal_report(df):
    if df.empty:
        return

    now = dt.datetime.now(TIMEZONE).strftime("%H:%M")
    last = df.iloc[-1]
    rsi = last["RSI"]
    ema9 = last["EMA9"]
    ema200 = last["EMA200"]
    tendance = "↗️ Haussière" if ema9 > ema200 else "↘️ Baissière"

    msg = f"""
🕐 *RAPPORT PÉRIODIQUE — {now}*
━━━━━━━━━━━━━━━━━━━━━━━
🔍 Aucune opportunité fiable détectée.
📊 RSI actuel : `{round(rsi, 2)}`
📈 Tendance : {tendance}

💤 Le bot continue de surveiller le marché ETH/USDT...
    """.strip()

    await application.bot.send_message(
        chat_id=CHAT_ID,
        text=msg,
        parse_mode="Markdown"
    )
    
# --- 📡 RÉCUPÉRATION DES ACTUS TWITTER ---
import subprocess

def get_latest_tweets(keyword="ethereum", limit=10):
    try:
        command = f'snscrape --max-results {limit} twitter-search "{keyword}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        tweets = result.stdout.strip().split("\n")
        return tweets[-limit:]  # Retourne les derniers tweets utiles
    except Exception as e:
        print(f"[TWITTER] Erreur récupération tweets : {e}")
        return []
tweets = get_latest_tweets("ethusdt")
for t in tweets:
    print(t)

# --- 💬 ANALYSE DE SENTIMENT D’UN TEXTE ---
from textblob import TextBlob

def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Va de -1 (très négatif) à +1 (très positif)
        if polarity > 0.1:
            return "positif"
        elif polarity < -0.1:
            return "négatif"
        else:
            return "neutre"
    except Exception as e:
        print(f"[SENTIMENT] Erreur : {e}")
        return "neutre"
tweets = get_latest_tweets("ethusdt", 5)
for t in tweets:
    sentiment = analyze_sentiment(t)
    print(f"> {sentiment} : {t[:100]}")

# --- 🧪 ÉVALUATION DU SENTIMENT GLOBAL DES TWEETS ETH ---
def evaluate_market_sentiment():
    tweets = get_latest_tweets("ethusdt", 10)
    sentiments = {"positif": 0, "négatif": 0, "neutre": 0}

    for t in tweets:
        result = analyze_sentiment(t)
        sentiments[result] += 1

    total = sum(sentiments.values())
    if total == 0:
        return "neutre"

    dominant = max(sentiments, key=sentiments.get)
    return dominant

    # --- Ajustement du score selon l’actualité Twitter ---
    sentiment = evaluate_market_sentiment()
    if sentiment == "positif":
        score += 5
        explanations.append("💬 Sentiment Twitter global positif")
    elif sentiment == "négatif":
        score -= 5
        explanations.append("💬 Sentiment Twitter global négatif")
    else:
        explanations.append("💬 Sentiment Twitter neutre")
# --- 📊 ENVOI DU RÉSUMÉ JOURNALIER À 23h59 ---
async def send_daily_summary():
    total = daily_stats["total_signals"]
    wins = daily_stats["wins"]
    losses = daily_stats["losses"]
    directions = daily_stats["directions"]

    winrate = round((wins / total) * 100, 2) if total > 0 else 0
    tendance = max(set(directions), key=directions.count) if directions else "Aucune"

    msg = f"""
📆 *RÉSUMÉ DE LA JOURNÉE — {dt.datetime.now(TIMEZONE).strftime("%d/%m/%Y")}*
━━━━━━━━━━━━━━━━━━━━━━━
📈 Total signaux : `{total}`
✅ Gagnés : `{wins}` | ❌ Perdus : `{losses}`
🎯 Taux de réussite : `{winrate}%`
📊 Tendance dominante : *{tendance}*

💤 Bonne nuit ! Le bot reprendra demain matin automatiquement.
    """.strip()

    # Réinitialisation des stats pour demain
    for key in daily_stats:
        daily_stats[key] = 0 if key != "directions" else []

    await application.bot.send_message(
        chat_id=CHAT_ID,
        text=msg,
        parse_mode="Markdown"
    )
# --- 🔁 LANCEMENT DU BOT & TÂCHES PLANIFIÉES ---
if __name__ == "__main__":
    scheduler = AsyncIOScheduler()

    async def periodic_report():
        df = get_ohlcv(PAIR, "1m", LIMIT)
        df = calculate_indicators(df)
        await send_no_signal_report(df)

    scheduler.add_job(periodic_report, "interval", minutes=30)
        scheduler.add_job(send_daily_summary, "cron", hour=23, minute=59)
    scheduler.start()

    application.add_handler(CommandHandler("start", lambda update, context: update.message.reply_text("🤖 Bot actif et en surveillance...")))

    application.run_task(monitor_market())
    application.run_polling()

