"""
Threshold-based alerting. Fires a Telegram message and/or an email
when called. Both are optional — configure whichever you have credentials for.
"""
import sys
import os
import smtplib
from email.mime.text import MIMEText
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    ALERT_EMAIL_FROM, ALERT_EMAIL_TO, ALERT_EMAIL_APP_PASSWORD,
)


def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        return resp.ok
    except Exception as e:
        print(f"Telegram alert failed: {e}")
        return False


def send_email_alert(message: str, subject: str = "Pearls AQI Alert"):
    if not (ALERT_EMAIL_FROM and ALERT_EMAIL_TO and ALERT_EMAIL_APP_PASSWORD):
        return False
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ALERT_EMAIL_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(ALERT_EMAIL_FROM, ALERT_EMAIL_APP_PASSWORD)
            server.sendmail(ALERT_EMAIL_FROM, [ALERT_EMAIL_TO], msg.as_string())
        return True
    except Exception as e:
        print(f"Email alert failed: {e}")
        return False


def send_alert(message: str):
    print(f"ALERT: {message}")
    sent_telegram = send_telegram_alert(message)
    sent_email = send_email_alert(message)
    if not (sent_telegram or sent_email):
        print("No alert channel configured (set TELEGRAM_* or ALERT_EMAIL_* in .env).")
