import requests
import datetime
import pytz
import pandas as pd
import numpy as np
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import traceback

# --- CONFIGURATION ---
TOKEN = "8212480058:AAHrq5yjlAzrnOlAla0IE42V2Z2w4Y05s80"
CHAT_ID = "1091559539"
SYMBOLS = ["BTCUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "ETHUSDT", "DASHUSDT", "BCHUSDT", "FILUSDT", "LTCUSDT", "YFIUSDT", "ZECUSDT"]
TIMEZONE = pytz.timezone("Europe/Paris")
SIGNAL_COOLDOWN_MINUTES = 5
CONFIDENCE_THRESHOLD = 60

# --- PARAMÈTRES INDICATEURS ---
RSI_PERIOD = 14
EMA_PERIOD = 9
EMA_TREND = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
CCI_PERIOD = 20
VOLUME_PERIOD = 20
BB_PERIOD = 20
STOCH_PERIOD = 14
RANGE_THRESHOLD = 0.15

# --- VARIABLES GLOBALES ---
app = Application.builder().token(TOKEN).build()
is_running = False
last_sent_signals = {}
active_signals = {}
last_alert_time = datetime.datetime.now(TIMEZONE)
# --- ANALYSE MULTI-TIMEFRAME ---
def get_multi_timeframe_data(symbol):
    data = {}
    for tf in ["1m", "5m", "15m"]:
        df = get_ohlcv(symbol, tf)
        df = calculate_indicators(df)
        data[tf] = df
    return data

def combine_signal_mtf(data):
    sig_5m = check_signal(data["5m"])
    sig_15m = check_signal(data["15m"])
    if sig_5m and sig_15m and sig_5m == sig_15m:
        return sig_5m
    return None

# --- FILTRE SUPPORT / RÉSISTANCE ---
def is_near_support_resistance(df):
    last_close = df["close"].iloc[-1]
    recent_high = df["high"].rolling(20).max().iloc[-1]
    recent_low = df["low"].rolling(20).min().iloc[-1]
    seuil = (recent_high - recent_low) * 0.05
    near_resistance = abs(last_close - recent_high) < seuil
    near_support = abs(last_close - recent_low) < seuil
    return near_resistance or near_support

# --- MESSAGE DE CONFIRMATION OU D'AVERTISSEMENT ---
async def confirm_with_m1(symbol, main_signal, active_df):
    df_m1 = get_ohlcv(symbol, "1m")
    df_m1 = calculate_indicators(df_m1)
    confirm_signal = check_signal(df_m1)
    confidence, reasons = estimate_confidence(df_m1)

    if confirm_signal == main_signal:
        await app.bot.send_message(
            chat_id=CHAT_ID,
            text=f"🧠 *Confirmation M1 reçue pour* `{symbol}`\n✅ Signal confirmé.\n🔒 Fiabilité ajustée : *{confidence}%*",
            parse_mode="Markdown"
        )
    else:
        await app.bot.send_message(
            chat_id=CHAT_ID,
            text=f"⚠️ *Attention M1 n'a pas confirmé le signal sur* `{symbol}`\n❌ Fiabilité diminuée : *{confidence}%*\n💡 Réfléchis avant d’entrer en position.",
            parse_mode="Markdown"
        )
        # --- SIGNAL + ANTI-DOUBLON + TIMER ---
from collections import deque
signal_history = {}

async def send_signal(symbol, signal, df, confidence, reasons):
    now = datetime.datetime.now(TIMEZONE)
    if symbol not in signal_history:
        signal_history[symbol] = deque(maxlen=10)
    for entry in signal_history[symbol]:
        if entry["type"] == signal and (now - entry["time"]).total_seconds() < 300:
            return  # Signal déjà envoyé récemment
    signal_history[symbol].append({"type": signal, "time": now})

    next_time = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)
    delay = (next_time - now).seconds
    timer_msg = f"⏳ *Place ton trade dans* : {delay} sec"

    reason_txt = "\n".join([f"- {r}" for r in reasons])
    close = df["close"].iloc[-1]
    active_signals[symbol] = {"type": signal, "price": close, "time": next_time}

    msg = (
        f"🚨 *Signal détecté* : {signal}\n"
        f"📊 *Paire* : `{symbol}`\n"
        f"⏰ *Trade à* : {next_time.strftime('%H:%M:%S')}\n"
        f"{timer_msg}\n\n"
        f"📉 RSI : {df['RSI'].iloc[-1]:.2f} | EMA : {df['EMA'].iloc[-1]:.2f} | EMA200 : {df['EMA200'].iloc[-1]:.2f}\n"
        f"📈 MACD : {df['MACD'].iloc[-1]:.4f} | 🔁 Signal : {df['MACD_Signal'].iloc[-1]:.4f}\n"
        f"📏 CCI : {df['CCI'].iloc[-1]:.2f} | ADX : {df['ADX'].iloc[-1]:.2f} | 🔊 Volume : {df['volume'].iloc[-1]:.2f}\n\n"
        f"🧠 *Fiabilité estimée* : {confidence}%\n"
        f"✅ *Critères validés* :\n{reason_txt}"
    )
    await app.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    # --- CONFIRMATION PAR M1 AVANT TRADE ---
async def confirm_signal_with_m1(symbol):
    if symbol not in active_signals:
        return
    df_m1 = get_ohlcv(symbol, "1m")
    df_m1 = calculate_indicators(df_m1)
    signal_type = active_signals[symbol]["type"]
    confidence_m1, reasons_m1 = estimate_confidence(df_m1)
    confirmation = None

    if signal_type.startswith("CALL") and df_m1["RSI"].iloc[-1] < 30 and df_m1["MACD"].iloc[-1] > df_m1["MACD_Signal"].iloc[-1]:
        confirmation = True
    elif signal_type.startswith("PUT") and df_m1["RSI"].iloc[-1] > 70 and df_m1["MACD"].iloc[-1] < df_m1["MACD_Signal"].iloc[-1]:
        confirmation = True
    else:
        confirmation = False

    if confirmation:
        msg = (
            f"🔁 *Confirmation M1* pour `{symbol}`\n"
            f"✅ M1 confirme le signal {signal_type}\n"
            f"📊 Nouvelle fiabilité : {confidence_m1}%\n"
            f"🧠 Raisons :\n" + "\n".join([f"- {r}" for r in reasons_m1])
        )
    else:
        msg = (
            f"⚠️ *Alerte* : `{symbol}`\n"
            f"❌ M1 *ne confirme pas* le signal {signal_type}\n"
            f"📉 Fiabilité revue à : {confidence_m1}%\n"
            f"⚠️ Nous vous déconseillons de suivre ce signal."
        )
    await app.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    # --- ENVOI DE RÉSUMÉ / AUCUN SIGNAL ---
last_report_time = datetime.datetime.now(TIMEZONE)

async def send_periodic_report():
    global last_report_time
    now = datetime.datetime.now(TIMEZONE)
    if (now - last_report_time).seconds >= 600:  # Toutes les 10 minutes
        summary = "🔎 *Analyse périodique :*\nAucun signal fiable détecté sur les dernières paires analysées.\n"
        await app.bot.send_message(chat_id=CHAT_ID, text=summary, parse_mode="Markdown")
        last_report_time = now

# --- AUTO-RESTART SI CRASH ---
async def safe_monitoring_loop():
    while True:
        try:
            await monitoring_loop()
        except Exception as e:
            await app.bot.send_message(chat_id=CHAT_ID, text=f"🔁 *Redémarrage automatique après erreur :* {str(e)}")
            await asyncio.sleep(5)

# --- SET THRESHOLD (pour futur amélioration) ---
THRESHOLD = 60

async def set_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global THRESHOLD
    try:
        new_val = int(context.args[0])
        THRESHOLD = new_val
        await update.message.reply_text(f"🔧 Nouveau seuil de fiabilité : {THRESHOLD}%")
    except:
        await update.message.reply_text("❌ Format invalide. Utilisez : /set_threshold 70")

# --- AUTO-CLEAN DES VIEUX SIGNAUX ---
def clean_old_signals():
    now = datetime.datetime.now(TIMEZONE)
    for symbol in list(active_signals.keys()):
        if (now - active_signals[symbol]["time"]).seconds > 180:
            del active_signals[symbol]
            # --- BOUCLE PRINCIPALE MONITORING ---
async def monitoring_loop():
    global is_running
    if is_running:
        return
    is_running = True
    await app.bot.send_message(chat_id=CHAT_ID, text="✅ Bot lancé et prêt à analyser les marchés.")
    try:
        while is_running:
            for symbol in SYMBOLS:
                try:
                    m1 = get_ohlcv(symbol, "1m")
                    m5 = get_ohlcv(symbol, "5m")
                    m15 = get_ohlcv(symbol, "15m")

                    m1 = calculate_indicators(m1)
                    m5 = calculate_indicators(m5)
                    m15 = calculate_indicators(m15)

                    base_signal = check_signal(m5)
                    confirm_signal = check_signal(m15)

                    if base_signal and confirm_signal and base_signal == confirm_signal:
                        key = f"{symbol}_{base_signal}_{m5['timestamp'].iloc[-1]}"
                        if key != last_sent_signals.get(symbol):
                            confidence, reasons = estimate_confidence(m5)
                            if confidence >= THRESHOLD:
                                await send_signal(symbol, base_signal, m5, confidence, reasons)
                                last_sent_signals[symbol] = key

                    # Confirmation M1 avant le trade
                    if symbol in active_signals:
                        now = datetime.datetime.now(TIMEZONE)
                        exec_time = active_signals[symbol]["time"]
                        if now >= exec_time - datetime.timedelta(seconds=60) and not active_signals[symbol].get("confirmed"):
                            m1_signal = check_signal(m1)
                            if m1_signal == active_signals[symbol]["type"]:
                                await app.bot.send_message(chat_id=CHAT_ID, text=f"✅ M1 confirme le signal `{symbol}`. Confiance renforcée.", parse_mode="Markdown")
                            else:
                                await app.bot.send_message(chat_id=CHAT_ID, text=f"⚠️ M1 ne confirme PAS le signal `{symbol}`. Précaution recommandée.", parse_mode="Markdown")
                            active_signals[symbol]["confirmed"] = True

                        # Retour 60s après exécution
                        if now >= exec_time + datetime.timedelta(seconds=60):
                            await send_result(symbol)

                except Exception as e:
                    print(f"[ERREUR] {symbol} : {e}")

            clean_old_signals()
            await send_periodic_report()
            await asyncio.sleep(10)

    finally:
        is_running = False

# --- COMMANDES TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("🛑 Stop", callback_data="stop")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("📊 Bot démarré. Analyse des marchés en cours...", reply_markup=reply_markup)

    # Lance la boucle de monitoring sans bloquer l'affichage du message
    asyncio.create_task(safe_monitoring_loop())

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_running
    query = update.callback_query
    await query.answer()
    if query.data == "stop":
        is_running = False
        await query.edit_message_text("🛑 Bot arrêté avec succès.")

# --- INIT ---
app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(button))
app.add_handler(CommandHandler("set_threshold", set_threshold))
# --- RÉSUMÉ JOURNALIER + TOP PAIRES VOLATILES ---
async def daily_summary():
    try:
        summary = "📅 *Résumé Journalier - Performances du Marché*\n\n"
        top_pairs = []
        for symbol in SYMBOLS:
            df = get_ohlcv(symbol, "15m")
            df = calculate_indicators(df)
            last_volatility = (df["high"] - df["low"]).mean()
            top_pairs.append((symbol, last_volatility))

        top_pairs.sort(key=lambda x: x[1], reverse=True)
        summary += "🔥 *Top paires les plus volatiles :*\n"
        for s, vol in top_pairs[:5]:
            summary += f"- `{s}` : Volatilité moyenne ~ {vol:.4f}\n"

        await app.bot.send_message(chat_id=CHAT_ID, text=summary, parse_mode="Markdown")
    except Exception as e:
        print(f"[ERREUR Résumé Journalier] {e}")
if __name__ == "__main__":
    async def main():
        scheduler = AsyncIOScheduler(timezone=TIMEZONE)
        scheduler.add_job(daily_summary, trigger='cron', hour=23, minute=59)
        scheduler.start()

        await app.initialize()
        await app.start()
        await app.bot.send_message(chat_id=CHAT_ID, text="✅ Bot lancé avec succès et prêt à analyser les marchés !")
        await app.run_polling()

    asyncio.run(main())