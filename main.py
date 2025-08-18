# --- 📦 IMPORTATIONS ---
import os
import pytz
import asyncio
import logging
import datetime as dt
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import subprocess
from textblob import TextBlob
from collections import defaultdict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- ⚙️ CONFIGURATION GLOBALE ---
TOKEN = "8212480058:AAHrq5yjlAzrnOlAla0IE42V2Z2w4Y05s80"
CHAT_ID = "1091559539"
TIMEZONE = pytz.timezone("Europe/Paris")
PAIR = "ETHUSDT"
INTERVALS = ["1m", "5m"]
LIMIT = 200

# --- 📊 INIT LOGS & TELEGRAM BOT ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
application = Application.builder().token(TOKEN).build()

# --- 📈 STATISTIQUES JOURNALIÈRES ---
daily_stats = {
    "total_signals": 0,
    "wins": 0,
    "losses": 0,
    "directions": [],
}

# --- 🧠 MÉMOIRE DES ERREURS (APPRENTISSAGE) ---
error_memory = []

# --- 📉 RÉCUPÉRATION DES DONNÉES DE MARCHÉ (Binance) ---
def get_ohlcv(pair="ETHUSDT", interval="1m", limit=200):
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": pair, "interval": interval, "limit": limit}
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
        print(f"[❌] Erreur téléchargement données : {e}")
        return pd.DataFrame()

# --- 📊 CALCUL DES INDICATEURS TECHNIQUES ---
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

# --- 🧠 DÉTECTION DES SIGNAUX + SCORE DE FIABILITÉ ---
def detect_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    confirmations = []
    explanations = []

    if last["RSI"] < 30:
        confirmations.append("CALL")
        explanations.append("RSI < 30 (survendu)")
    elif last["RSI"] > 70:
        confirmations.append("PUT")
        explanations.append("RSI > 70 (suracheté)")

    if last["MACD"] > last["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        confirmations.append("CALL")
        explanations.append("MACD croisement haussier")
    elif last["MACD"] < last["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        confirmations.append("PUT")
        explanations.append("MACD croisement baissier")

    if last["EMA9"] > last["EMA200"]:
        confirmations.append("CALL")
        explanations.append("Tendance haussière (EMA9 > EMA200)")
    elif last["EMA9"] < last["EMA200"]:
        confirmations.append("PUT")
        explanations.append("Tendance baissière (EMA9 < EMA200)")

    if last["StochRSI"] < 20:
        confirmations.append("CALL")
        explanations.append("StochRSI < 20")
    elif last["StochRSI"] > 80:
        confirmations.append("PUT")
        explanations.append("StochRSI > 80")

    call_count = confirmations.count("CALL")
    put_count = confirmations.count("PUT")

    if call_count > put_count:
        signal = "CALL"
    elif put_count > call_count:
        signal = "PUT"
    else:
        signal = None

    total = len(explanations)
    confirm = max(call_count, put_count)
    score = round((confirm / total) * 100) if total > 0 else 0

    return signal, score, explanations

# --- 📤 ENVOI DU SIGNAL SUR TELEGRAM ---
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

# --- ✅ ENVOI DU RÉSULTAT APRÈS 60s ---
async def send_result_after_trade(signal, entry_price):
    await asyncio.sleep(60)
    df = get_ohlcv(PAIR, "1m", 2)
    if df.empty:
        return

    last = df.iloc[-1]
    exit_price = last["close"]
    result = None

    if signal == "CALL":
        result = "✅ GAGNÉ" if exit_price > entry_price else "❌ PERDU"
    elif signal == "PUT":
        result = "✅ GAGNÉ" if exit_price < entry_price else "❌ PERDU"

    variation = round((exit_price - entry_price) / entry_price * 100, 3)
    direction = "↗️" if variation > 0 else "↘️"

    if result == "❌ PERDU":
        if abs(variation) < 0.1:
            reason = "🔸 Variation trop faible ➜ marché plat."
            error_memory.append("range_market")
        elif last["volume"] < df["volume"].rolling(20).mean().iloc[-1]:
            reason = "🔸 Volume faible ➜ entrée peu fiable."
            error_memory.append("low_volume")
        elif last["RSI"] > 60 and signal == "CALL":
            reason = "🔸 RSI trop élevé ➜ zone de retournement probable."
            error_memory.append("rsi_high_on_call")
        else:
            reason = "🔎 Le marché s’est retourné brutalement."
            error_memory.append("unclassified_error")
    else:
        reason = "📈 Le signal a été confirmé par le marché."

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
    )

    daily_stats["total_signals"] += 1
    if result == "✅ GAGNÉ":
        daily_stats["wins"] += 1
    elif result == "❌ PERDU":
        daily_stats["losses"] += 1
    daily_stats["directions"].append(signal)

# --- 🧠 AUTO-CORRECTION SIMPLIFIÉE DES ERREURS ---
def adjust_strategy_based_on_errors():
    if not error_memory:
        return

    stats = pd.Series(error_memory).value_counts()
    print("🔁 Auto-correction basée sur les erreurs les plus fréquentes :")
    print(stats)

    if stats.get("range_market", 0) >= 2:
        print("⚠️ Trop de marché plat ➜ éviter les signaux faibles.")
    if stats.get("low_volume", 0) >= 2:
        print("⚠️ Trop de volume faible ➜ filtrer les signaux par volume.")
        
# --- 🔄 MONITORING AUTOMATIQUE DU MARCHÉ ---
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

# --- 🕐 RAPPORT PÉRIODIQUE SANS SIGNAL ---
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
def get_latest_tweets(keyword="ethusdt", limit=10):
    try:
        command = f'snscrape --max-results {limit} twitter-search "{keyword}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        tweets = result.stdout.strip().split("\n")
        return tweets[-limit:]
    except Exception as e:
        print(f"[TWITTER] Erreur récupération tweets : {e}")
        return []

# --- 💬 ANALYSE DE SENTIMENT D’UN TWEET ---
def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "positif"
        elif polarity < -0.1:
            return "négatif"
        else:
            return "neutre"
    except Exception as e:
        print(f"[SENTIMENT] Erreur : {e}")
        return "neutre"

# --- 🧪 ÉVALUATION DU SENTIMENT GLOBAL DU MARCHÉ ---
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

# --- 📊 RÉSUMÉ JOURNALIER À 23H59 ---
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

    chart_path = generate_performance_chart()

    await application.bot.send_photo(chat_id=CHAT_ID, photo=open(chart_path, "rb"))
    await application.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

    # Réinitialisation pour le lendemain
    for key in daily_stats:
        daily_stats[key] = 0 if key != "directions" else []

# --- 📈 GRAPHIQUE DE SUIVI DES PERFORMANCES ---
def generate_performance_chart():
    total = daily_stats["total_signals"]
    wins = daily_stats["wins"]
    losses = daily_stats["losses"]

    labels = ["Gagnés", "Perdus"]
    values = [wins, losses]
    colors = ["green", "red"]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=colors)
    ax.set_title("Performance quotidienne des signaux")
    ax.set_ylabel("Nombre de trades")

    filename = "performance.png"
    plt.savefig(filename)
    plt.close()
    return filename

# --- 🟢 COMMANDE /start ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("🛑 STOP", callback_data="stop")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "✅ Bot lancé avec succès et prêt à analyser les marchés !\nClique sur le bouton ci-dessous pour l'arrêter si nécessaire.",
        reply_markup=reply_markup
    )

# --- 🔘 BOUTON 🛑 STOP ---
async def handle_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    await update.callback_query.edit_message_text("⛔️ Bot stoppé manuellement.")
    os._exit(0)

# --- 🔍 COMMANDE /verifie ---
async def verifie_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = get_ohlcv(PAIR, "1m", LIMIT)
    if df.empty:
        await update.message.reply_text("Erreur : impossible de récupérer les données.")
        return

    df = calculate_indicators(df)
    signal, confidence, explanations = detect_signal(df)

    if signal:
        text = f"""
✅ Signal détecté maintenant :
Direction : {signal}
Fiabilité : {confidence}%
Contexte :
{chr(10).join([f"- {e}" for e in explanations])}
        """.strip()
    else:
        text = "❌ Aucun signal détecté actuellement."

    await update.message.reply_text(text)

# --- 📚 COMMANDE /historique ---
async def historique_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📚 Fonction historique bientôt disponible (version simplifiée en cours de dev).")

# --- 📌 ENREGISTREMENT DES HANDLERS ---
application.add_handler(CommandHandler("start", start_command))
application.add_handler(CommandHandler("verifie", verifie_command))
application.add_handler(CommandHandler("historique", historique_command))
application.add_handler(CallbackQueryHandler(handle_stop))

# --- ⏰ PLANIFICATION DES TÂCHES AVEC APSCHEDULER + LANCEMENT DU BOT ---
if __name__ == "__main__":
    async def main():
        scheduler = AsyncIOScheduler()

        # --- Rapport toutes les 30 minutes ---
        async def periodic_report():
            df = get_ohlcv(PAIR, "1m", LIMIT)
            df = calculate_indicators(df)
            await send_no_signal_report(df)

        # --- Ajouter les tâches planifiées ---
        scheduler.add_job(periodic_report, "interval", minutes=30)
        scheduler.add_job(send_daily_summary, "cron", hour=23, minute=59)
        scheduler.start()

        # --- Démarrer le monitoring + le bot en parallèle ---
        asyncio.create_task(monitor_market())

        await application.initialize()
        await application.start()
        await application.bot.send_message(
    chat_id=CHAT_ID,
    text="🚀 Bot lancé avec succès et prêt à détecter les signaux sur ETH/USDT.",
)
        await application.updater.start_polling()
        await application.updater.idle()

    asyncio.run(main())