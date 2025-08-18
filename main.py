import os
import pytz
import asyncio
import logging
import datetime as dt
import pandas as pd
import numpy as np
import requests

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from apscheduler.schedulers.asyncio import AsyncIOScheduler

# === CONFIGURATIONS ===
TOKEN = "8450398342:AAEhPlH-lrECa2moq_4oSOKDjSmMpGmeaRA"
CHAT_ID = "1091559539"
TIMEZONE = pytz.timezone("Europe/Paris")

PAIR = "ETHUSDT"
INTERVALS = ["1m", "5m"]  # On travaille M1 + M5
LIMIT = 200  # nombre de bougies récupérées

# === LOGGING ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# === INITIALISATION DU BOT TELEGRAM ===
app = Application.builder().token(TOKEN).build()