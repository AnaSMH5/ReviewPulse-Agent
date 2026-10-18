"""Avisar al gerente y publicar el resumen en el sistema de reputación.

SEGURO POR DEFECTO: si no configuras nada, NO se envía nada. El aviso se imprime en pantalla
y se guarda en history/outbox/ (modo "dry_run"). Para enviar de verdad, define variables de
entorno (nunca escribas contraseñas ni URLs secretas en el código):

    REVIEWPULSE_WEBHOOK        URL de un webhook (Slack, Teams...). Recibe {"text": "..."}
    REVIEWPULSE_SMTP_HOST      \\
    REVIEWPULSE_SMTP_PORT       |  correo (el puerto por defecto es 587)
    REVIEWPULSE_SMTP_USER       |
    REVIEWPULSE_SMTP_PASSWORD   |
    REVIEWPULSE_ALERT_FROM      |
    REVIEWPULSE_ALERT_TO       /   uno o varios destinatarios separados por coma
    REVIEWPULSE_ENDPOINT       URL del sistema de reputación (recibe el resumen en JSON)
    REVIEWPULSE_TOKEN          token opcional para ese sistema

Si un envío falla, el agente NO se detiene: el error queda en el resultado y en el registro de acciones.
"""
import json
import os
import smtplib
import time
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import urlparse

from reviewpulse import config
from reviewpulse.report import to_plain


def _post_json(url: str, payload: dict, headers: dict = None, timeout: int = 15) -> int:
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("La URL debe empezar con http:// o https://")
    req = urllib.request.Request(
        url, data=json.dumps(to_plain(payload), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _send_email(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"], msg["From"] = subject, os.environ["REVIEWPULSE_ALERT_FROM"]
    msg["To"] = os.environ["REVIEWPULSE_ALERT_TO"]
    msg.set_content(body)
    with smtplib.SMTP(os.environ["REVIEWPULSE_SMTP_HOST"], int(os.getenv("REVIEWPULSE_SMTP_PORT", "587")), timeout=15) as s:
        s.starttls()
        if os.getenv("REVIEWPULSE_SMTP_USER"):
            s.login(os.environ["REVIEWPULSE_SMTP_USER"], os.environ["REVIEWPULSE_SMTP_PASSWORD"])
        s.send_message(msg)


def notify_manager(report: dict, history_dir=None) -> dict:
    """Envía el mensaje corto del reporte por los canales configurados.

    Devuelve {"mode": "sent" | "dry_run" | "error", "channels": [...], "errors": [...]}.
    """
    msg, week = report["short"], report["week"]
    channels, errors = [], []
    webhook = os.getenv("REVIEWPULSE_WEBHOOK")
    email_ok = bool(os.getenv("REVIEWPULSE_SMTP_HOST") and os.getenv("REVIEWPULSE_ALERT_TO"))

    if webhook:
        try:
            _post_json(webhook, {"text": msg})
            channels.append("webhook")
        except Exception as e:                       # solo el tipo de error: el mensaje podría traer la URL secreta
            errors.append(f"webhook: {type(e).__name__}")
    if email_ok:
        try:
            _send_email(f"ReviewPulse - {config.HOTEL_NAME} - semana {week}", report["markdown"])
            channels.append("email")
        except Exception as e:
            errors.append(f"email: {type(e).__name__}")

    if not webhook and not email_ok:                 # nada configurado -> simulación
        out = Path(history_dir or config.HISTORY_DIR) / "outbox"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{week}_aviso.txt").write_text(msg, encoding="utf-8")
        print(f"[SIMULADO - no se envió nada] {msg}")
        return {"mode": "dry_run", "channels": [], "errors": []}
    return {"mode": "sent" if channels else "error", "channels": channels, "errors": errors}


def publish_summary(report: dict, max_attempts: int = 3) -> dict:
    """Publica el resumen semanal (JSON, sin textos de huéspedes) en el sistema de reputación."""
    endpoint = os.getenv("REVIEWPULSE_ENDPOINT")
    if not endpoint:
        return {"mode": "dry_run", "attempts": 0, "error": None}
    headers = {"Idempotency-Key": f"reviewpulse-{report['week']}"}   # si se reenvía, el sistema puede ignorar el duplicado
    if os.getenv("REVIEWPULSE_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['REVIEWPULSE_TOKEN']}"
    error = None
    for intento in range(1, max_attempts + 1):
        try:
            _post_json(endpoint, report["payload"], headers)
            return {"mode": "sent", "attempts": intento, "error": None}
        except Exception as e:
            error = type(e).__name__
            if intento < max_attempts:
                time.sleep(1.5 ** intento)           # espera un poco más en cada reintento
    return {"mode": "error", "attempts": max_attempts, "error": error}
