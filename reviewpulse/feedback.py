"""PASO 8 - Correcciones humanas: la forma en que el agente aprende.

El ciclo:
    1. make_queue.py   reseñas nuevas -> el modelo las clasifica -> las dudosas van a data/feedback/review_queue.csv
    2. TÚ              abres review_queue.csv (Excel) y escribes el tema correcto en la columna `topic_correct`
    3. retrain.py      pasa lo corregido a corrections.csv, reentrena y SOLO reemplaza el modelo si no empeora

Los archivos usan ";" como separador y UTF-8 con BOM, para que Excel en español los abra bien.
OJO: contienen textos de huéspedes. Mantén el repositorio privado.
"""
import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from reviewpulse import config

# Los 9 temas oficiales: en `topic_correct` solo se aceptan estos (sin importar mayúsculas ni tildes)
TOPICS = [
    "Atención y personal", "Comida y bebidas", "Habitaciones", "Piscina, playa y entretenimiento",
    "Mantenimiento e instalaciones", "Limpieza e higiene", "Ubicación",
    "Check-in / Check-out y reservas", "Precio / relación calidad-precio",
]
QUEUE_FILE, CORR_FILE = "review_queue.csv", "corrections.csv"
QUEUE_COLS = ["id", "text", "rating", "topic_predicted", "topic_conf", "topic_correct"]
CORR_COLS = ["id", "text", "rating", "main_topic"]


def _norm(s) -> str:
    t = unicodedata.normalize("NFKD", str(s).lower())
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


_CANON = {_norm(t): t for t in TOPICS}


def _dir(feedback_dir) -> Path:
    d = Path(feedback_dir or config.FEEDBACK_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read(path: Path, cols: list) -> pd.DataFrame:
    """Lee un CSV sin importar cómo lo haya guardado Excel (separador ; o , y codificación UTF-8 o antigua)."""
    if not path.exists():
        return pd.DataFrame(columns=cols)
    for codificacion in ("utf-8-sig", "cp1252"):          # el "CSV" clásico de Excel usa cp1252
        try:
            texto = path.read_text(encoding=codificacion)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"No pude leer {path.name}: guárdalo como 'CSV UTF-8' desde Excel.")
    if not texto.strip():
        return pd.DataFrame(columns=cols)
    encabezado = texto.splitlines()[0]
    sep = ";" if encabezado.count(";") >= encabezado.count(",") else ","
    df = pd.read_csv(io.StringIO(texto), sep=sep, dtype={"id": str})
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"{path.name} no tiene las columnas {faltan}. ¿Lo guardaste como CSV sin borrar los encabezados?")
    return df


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def export_queue(pred: pd.DataFrame, feedback_dir=None) -> int:
    """Agrega a la cola las reseñas que el modelo marcó con needs_review (las más dudosas primero).

    `pred` es el resultado de classify.predict(). No repite reseñas que ya están en la cola
    o que ya fueron corregidas. Devuelve cuántas agregó.
    """
    d = _dir(feedback_dir)
    queue, corr = _read(d / QUEUE_FILE, QUEUE_COLS), _read(d / CORR_FILE, CORR_COLS)
    conocidas = set(queue["id"].astype(str)) | set(corr["id"].astype(str))
    nuevas = pred[pred["needs_review"] & ~pred["id"].astype(str).isin(conocidas)].sort_values("topic_conf")
    filas = pd.DataFrame({
        "id": nuevas["id"].astype(str), "text": nuevas["text"],
        "rating": nuevas["rating"] if "rating" in nuevas.columns else np.nan,
        "topic_predicted": nuevas["topic"], "topic_conf": nuevas["topic_conf"].round(3), "topic_correct": "",
    })
    _write(pd.concat([queue, filas], ignore_index=True)[QUEUE_COLS], d / QUEUE_FILE)
    return len(filas)


def apply_queue(feedback_dir=None) -> dict:
    """Pasa de la cola a corrections.csv las filas donde ya escribiste `topic_correct`.

    - Si el tema no es uno de los 9 oficiales, la fila NO se acepta: queda en la cola y se avisa.
    - Las filas sin corregir siguen pendientes en la cola.
    """
    d = _dir(feedback_dir)
    queue, corr = _read(d / QUEUE_FILE, QUEUE_COLS), _read(d / CORR_FILE, CORR_COLS)
    escrito = queue["topic_correct"].notna() & (queue["topic_correct"].astype(str).str.strip() != "")
    canon = queue["topic_correct"].map(lambda x: _CANON.get(_norm(x)) if pd.notna(x) else None)
    ok, mal = escrito & canon.notna(), escrito & canon.isna()

    nuevas = pd.DataFrame({"id": queue.loc[ok, "id"], "text": queue.loc[ok, "text"],
                           "rating": queue.loc[ok, "rating"], "main_topic": canon[ok]})
    corr = pd.concat([corr[~corr["id"].isin(nuevas["id"])], nuevas], ignore_index=True)[CORR_COLS]
    _write(corr, d / CORR_FILE)
    _write(queue[~ok][QUEUE_COLS], d / QUEUE_FILE)
    return {"added": int(ok.sum()), "pending": int((~escrito).sum()), "invalid": queue.loc[mal, "id"].tolist()}


def load_corrections(feedback_dir=None) -> pd.DataFrame:
    """Todas las correcciones acumuladas: id, text, rating (puede faltar) y main_topic."""
    corr = _read(_dir(feedback_dir) / CORR_FILE, CORR_COLS)
    corr["rating"] = pd.to_numeric(corr["rating"], errors="coerce")
    return corr
