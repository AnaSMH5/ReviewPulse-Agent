"""Armar el resumen semanal por tema.

Entrada : las reseñas (DataFrame) y el resultado de analysis.compare(serie, semana)
Salida  : un diccionario con todo lo necesario:
            "markdown"    -> el resumen completo, para leer (se guarda en history/summaries/)
            "short"       -> mensaje corto para avisar al gerente (sin textos de huéspedes)
            "payload"     -> datos para el sistema de reputación (sin textos de huéspedes)
            "weekly_row"  -> una fila para el historial semanal
            "topic_rows"  -> una fila por tema para el historial por tema

No hay IA aquí: el texto se arma con plantillas a partir de los números. Así es
reproducible y no inventa nada.
"""
import math
import re
from datetime import date, datetime

import numpy as np
import pandas as pd

from reviewpulse import config

# Qué hacer según el tema que más negativas acumula. Son sugerencias de partida: ajústalas.
TOPIC_ACTIONS = {
    "Atención y personal": "Revisar turnos y protocolos de atención en recepción y servicio, y contactar a los huéspedes afectados.",
    "Comida y bebidas": "Revisar variedad, calidad y tiempos del servicio de alimentos y bebidas, incluido el esquema todo incluido.",
    "Habitaciones": "Inspeccionar el estado de las habitaciones (camas, baños, aire acondicionado) y ofrecer cambio cuando haya quejas.",
    "Piscina, playa y entretenimiento": "Verificar el estado y la disponibilidad de la piscina, el acceso a la playa y las actividades.",
    "Mantenimiento e instalaciones": "Abrir órdenes de mantenimiento por los daños mencionados, priorizando climatización y zonas comunes.",
    "Limpieza e higiene": "Auditar la limpieza de habitaciones y áreas comunes y reforzar el control de plagas.",
    "Ubicación": "Reforzar en la confirmación de reserva la información de llegada y de la zona (mapa, transporte, seguridad).",
    "Check-in / Check-out y reservas": "Revisar el proceso de check-in/check-out y que lo reservado coincida con lo entregado.",
    "Precio / relación calidad-precio": "Revisar tarifas, cargos adicionales y el valor percibido frente a lo que se ofrece.",
}
DEFAULT_ACTION = "Revisar manualmente las reseñas negativas de este tema."
STATUS_LABEL = {"alerta": "ALERTA", "normal": "Normal",
                "datos_insuficientes": "Datos insuficientes"}


# ---------------------------------------------------------------- utilidades
def to_plain(obj):
    """Convierte fechas, tipos de numpy y NaN en tipos simples (para guardar como JSON o CSV)."""
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)\+?\d[\d\s().-]{7,}\d(?!\d)")


def _snippet(text, max_chars=160):
    """Recorta una reseña y enmascara correos y teléfonos. No sustituye una revisión de privacidad."""
    t = _PHONE.sub("[tel]", _EMAIL.sub("[email]", str(text)))
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= max_chars else t[: max_chars - 1] + "…"


def _first_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(
        f"El DataFrame necesita alguna de estas columnas: {names}")


def _pct(x):
    return "n/d" if x is None else f"{x:.0%}"


def _num(x):
    return "n/d" if x is None else f"{x:.2f}"


def _delta(cur, prev, scale=1.0, unit=""):
    if cur is None or prev is None:
        return "n/d"
    return f"{(cur - prev) * scale:+.{0 if scale == 100 else 2}f}{unit}"


# ---------------------------------------------------------------- por tema
def _reviews_in_weeks(df, last_week, weeks):
    """Reseñas de las `weeks` semanas que terminan en `last_week` (las mismas que usa analysis.compare)."""
    start = last_week - pd.Timedelta(weeks=weeks - 1)
    return df[(df["week_start"] >= start) & (df["week_start"] <= last_week)]


def _stats(g, pcol):
    """Cantidad, negativas, % de negativas y rating medio de un grupo de reseñas."""
    n = len(g)
    neg = int((g[pcol] <= config.NEGATIVE_MAX_RATING).sum())
    return {"n": n, "n_negative": neg, "pct_negative": neg / n if n else None,
            "mean_rating": float(g[pcol].mean()) if n else None}


def _topic_rows(cur_df, prev_df, tcol, pcol, week):
    prev = {t: _stats(g, pcol) for t, g in prev_df.groupby(tcol)}
    rows = []
    for t, g in cur_df.groupby(tcol):
        s = _stats(g, pcol)
        p = prev.get(t, {"n": 0, "n_negative": 0})
        rows.append({
            "week": week, "topic": t, "n": s["n"], "negativas": s["n_negative"],
            "pct_negativas": s["pct_negative"], "rating_medio": s["mean_rating"],
            "prev_n": p["n"], "prev_negativas": p["n_negative"],
        })
    # primero los temas con más negativas; si empatan, el que más subió
    rows.sort(key=lambda r: (-r["negativas"], -
              (r["negativas"] - r["prev_negativas"])))
    return rows


# ---------------------------------------------------------------- el reporte
def build_report(df: pd.DataFrame, result: dict, include_examples: bool = True) -> dict:
    if "polarity" in df.columns and "rating" in df.columns:
        df["_analysis_polarity"] = df["polarity"].fillna(df["rating"])
    elif "polarity" in df.columns:
        df["_analysis_polarity"] = df["polarity"]
    else:
        df["_analysis_polarity"] = df["rating"]
    if "topic" in df.columns and "main_topic" in df.columns:
        df["_analysis_topic"] = df["topic"].fillna(df["main_topic"])
    elif "topic" in df.columns:
        df["_analysis_topic"] = df["topic"]
    else:
        df["_analysis_topic"] = df["main_topic"]

    # el lunes de la semana analizada
    week_start = pd.Timestamp(result["week"])
    week = week_start.date().isoformat()
    week_end = (week_start + pd.Timedelta(days=6)).date().isoformat()
    w = config.WINDOW_WEEKS

    # Las mismas dos ventanas que usó analysis.compare()
    cur_df = _reviews_in_weeks(df, week_start, w)
    prev_df = _reviews_in_weeks(df, week_start - pd.Timedelta(weeks=w), w)
    tcol = "_analysis_topic"
    pcol = "_analysis_polarity"
    ccol = "text" if "text" in df.columns else None

    cur, prev = result["current"], result["previous"]
    status = result["status"]
    topics = _topic_rows(cur_df, prev_df, tcol, pcol, week)
    priority = [t for t in topics if t["negativas"]
                > 0][: config.TOP_TOPICS_IN_REPORT]

    # ---- texto ----
    L = [f"# Resumen semanal ReviewPulse - {config.HOTEL_NAME}",
         f"Semana del **{week}** al **{week_end}** | ventanas de {w} semanas", "",
         f"## Estado: {STATUS_LABEL.get(status, status)}", result[
        "reason"] or "Sin cambios relevantes.", "",
        f"## Tono general (últimas {w} semanas vs {w} semanas anteriores)", "",
        "| | Anterior | Actual | Cambio |", "|---|---|---|---|",
        f"| Reseñas | {prev['n']} | {cur['n']} | {cur['n'] - prev['n']:+d} |",
        f"| % negativas | {_pct(prev['pct_negative'])} | {_pct(cur['pct_negative'])} | "
        f"{_delta(cur['pct_negative'], prev['pct_negative'], 100, ' pts')} |",
        f"| Rating medio | {_num(prev['mean_rating'])} | {_num(cur['mean_rating'])} | "
        f"{_delta(cur['mean_rating'], prev['mean_rating'])} |", ""]
    if status == "datos_insuficientes":
        L += ["> Hay pocas reseñas para concluir si el hotel mejora o empeora. Los números de abajo son solo informativos.", ""]

    L += ["## Por tema (ventana actual)", ""]
    if topics:
        L += ["| Tema | Reseñas | Negativas | % neg. | Negativas ventana anterior | Nota |",
              "|---|---|---|---|---|---|"]
        for t in topics:
            nota = "poca muestra" if t["n"] < config.MIN_TOPIC_REVIEWS else ""
            L.append(
                f"| {t['topic']} | {t['n']} | {t['negativas']} | {_pct(t['pct_negativas'])} | {t['prev_negativas']} | {nota} |")
    else:
        L.append("No hubo reseñas en la ventana actual.")
    L.append("")

    actions = []
    if priority:
        L += ["## Temas prioritarios y acciones sugeridas", ""]
        for i, t in enumerate(priority, 1):
            accion = TOPIC_ACTIONS.get(t["topic"], DEFAULT_ACTION)
            actions.append({"topic": t["topic"], "action": accion})
            L.append(f"{i}. **{t['topic']}**: {t['negativas']} negativas de {t['n']} reseñas "
                     f"(ventana anterior: {t['prev_negativas']}).")
            L.append(f"   - Acción sugerida: {accion}")
            if include_examples and ccol:
                g = cur_df[cur_df[tcol] == t["topic"]].sort_values(
                    [pcol, "created_at"], ascending=[True, False])
                if len(g) and g.iloc[0][pcol] <= config.NEGATIVE_MAX_RATING:
                    L.append(f"   - Ejemplo: \"{_snippet(g.iloc[0][ccol])}\"")
        L.append("")
    L.append("_Las acciones son sugerencias; la decisión es de la gerencia._")

    # ---- mensaje corto (sin textos de huéspedes) ----
    top = ", ".join(
        f"{t['topic']} ({t['negativas']})" for t in priority) or "sin negativas"
    short = f"ReviewPulse | {config.HOTEL_NAME} | {STATUS_LABEL.get(status, status).upper()} semana del {week}: " \
        f"{result['reason']} Temas con más negativas: {top}."

    weekly_row = {
        "week": week, "status": status, "alert": bool(result["alert"]), "reason": result["reason"],
        "cur_n": cur["n"], "cur_negativas": cur["n_negative"], "cur_pct_negativas": cur["pct_negative"],
        "cur_rating_medio": cur["mean_rating"], "prev_n": prev["n"], "prev_negativas": prev["n_negative"],
        "prev_pct_negativas": prev["pct_negative"], "prev_rating_medio": prev["mean_rating"],
        "delta_pct_negativas": result.get("delta_pct_negative"), "p_value": result.get("p_value"),
    }
    payload = {"hotel": config.HOTEL_NAME, "week_start": week, "week_end": week_end, "status": status, "alert": bool(result["alert"]),
               "reason": result["reason"], "current": cur, "previous": prev,
               "topics": topics, "recommended_actions": actions}

    return to_plain({"week": week, "status": status, "alert": bool(result["alert"]), "markdown": "\n".join(L),
                     "short": short, "payload": payload, "weekly_row": weekly_row, "topic_rows": topics,
                     "actions": actions})
