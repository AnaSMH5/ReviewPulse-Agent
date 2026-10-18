"""Guardar el historial: lo que el agente calculó, dijo e hizo cada semana.

Todo queda en la carpeta history/ (config.HISTORY_DIR):
    weekly_history.csv      una fila por semana (números globales y estado)
    topic_history.csv       una fila por semana y tema
    summaries/<semana>.md   el resumen completo en texto
    actions_log.jsonl       las acciones tomadas (avisos enviados, publicaciones...), una línea por evento

Correr dos veces la misma semana NO duplica filas: la segunda corrida reemplaza a la primera.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reviewpulse import config
from reviewpulse.report import to_plain

WEEKLY_FILE, TOPIC_FILE, ACTIONS_FILE = "weekly_history.csv", "topic_history.csv", "actions_log.jsonl"


def _dir(history_dir) -> Path:
    d = Path(history_dir or config.HISTORY_DIR)
    (d / "summaries").mkdir(parents=True, exist_ok=True)
    return d


def _replace_week(path: Path, rows: list, week: str) -> None:
    """Quita lo que hubiera de esa semana, agrega las filas nuevas y guarda."""
    nuevas = pd.DataFrame(rows)
    if path.exists():
        viejas = pd.read_csv(path, dtype={"week": str})
        nuevas = pd.concat([viejas[viejas["week"] != week], nuevas], ignore_index=True)
    nuevas = nuevas.sort_values("week")
    tmp = path.with_suffix(".csv.tmp")           # se escribe a un temporal y se renombra: nunca queda a medias
    nuevas.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def save_run(report: dict, history_dir=None) -> dict:
    """Guarda fila semanal, filas por tema y el resumen en texto."""
    d, week = _dir(history_dir), report["week"]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _replace_week(d / WEEKLY_FILE, [{**report["weekly_row"], "generated_at": stamp}], week)
    if report["topic_rows"]:
        _replace_week(d / TOPIC_FILE, report["topic_rows"], week)
    summary = d / "summaries" / f"{week}.md"
    summary.write_text(report["markdown"], encoding="utf-8")
    return {"weekly": d / WEEKLY_FILE, "topics": d / TOPIC_FILE, "summary": summary}


def load_history(history_dir=None) -> pd.DataFrame:
    """Lee el historial semanal (vacío si todavía no hay corridas)."""
    path = _dir(history_dir) / WEEKLY_FILE
    return pd.read_csv(path, dtype={"week": str}) if path.exists() else pd.DataFrame()


def log_action(week: str, event: str, detail: dict = None, history_dir=None) -> None:
    """Agrega una línea al registro de acciones. Ej: log_action('2025-03-23', 'manager_notified', {...})"""
    linea = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "week": week, "event": event, "detail": to_plain(detail or {})}
    with (_dir(history_dir) / ACTIONS_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(linea, ensure_ascii=False) + "\n")


def already_notified(week: str, history_dir=None) -> bool:
    """¿Ya se avisó al gerente de verdad (no en simulación) por esta semana? Evita avisos repetidos."""
    path = _dir(history_dir) / ACTIONS_FILE
    if not path.exists():
        return False
    for linea in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(linea)
        if e["week"] == week and e["event"] == "manager_notified" and e["detail"].get("mode") == "sent":
            return True
    return False
