import pandas as pd

from reviewpulse import config


def _week_start(fechas: pd.Series) -> pd.Series:
    """Semana = el lunes de esa semana."""
    dia = fechas.dt.normalize()
    return dia - pd.to_timedelta(dia.dt.weekday, unit="D")


def load_reviews(path=None) -> pd.DataFrame:
    """Carga reviews_eda.csv (ya limpio) y lo deja con los nombres de columnas del agente."""
    raw = pd.read_csv(path or config.DATA_PROCESSED / "reviews_eda.csv", encoding="utf-8-sig")
    df = pd.DataFrame({
        "id": raw["id"],
        "rating": raw["rating"],
        "text": raw["texto_completo"],                       # título + texto
        "created_at": pd.to_datetime(raw["fecha_publicacion"]),
        "week_start": pd.to_datetime(raw["semana"]),
        "main_topic": raw["tema_principal"],
        "topics": raw["temas"].str.split("|"),               # lista de temas de la reseña
    })
    return df.sort_values("created_at").reset_index(drop=True)


def load_raw_reviews(path=None) -> pd.DataFrame:
    """Carga reseñas en el formato original del sitio (sin temas)."""
    raw = pd.read_csv(path or config.DATA_RAW / "reviews.csv", encoding="utf-8-sig")
    # El sitio entrega la hora en UTC. La pasamos a hora de Colombia ANTES de calcular la semana:
    # una reseña del lunes 01:00 UTC es del domingo 20:00 en Colombia, o sea de la semana anterior.
    fecha = pd.to_datetime(raw["Created Date"], utc=True).dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)
    df = pd.DataFrame({
        "id": raw["Id"],
        "rating": raw["Rating"].astype(int),
        "text": raw["Review Title"].fillna("") + ". " + raw["Review Text"].fillna(""),
        "created_at": fecha,
        "week_start": _week_start(fecha),
    })
    return df.sort_values("created_at").reset_index(drop=True)
