import pandas as pd

from reviewpulse import config


def weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por semana, SIN huecos (las semanas sin reseñas quedan con n=0)."""
    # Si el clasificador ya puso "polarity" (predicha), la usamos; si no, el rating original.
    col = "polarity" if "polarity" in df.columns else "rating"
    g = df.groupby("week_start")[col]
    s = pd.DataFrame({
        "n": g.size(),
        "negativas": g.apply(lambda x: int((x <= config.NEGATIVE_MAX_RATING).sum())),
        "rating_medio": g.mean(),
    })
    todas = pd.date_range(s.index.min(), s.index.max(), freq="7D")
    s = s.reindex(todas)
    s[["n", "negativas"]] = s[["n", "negativas"]].fillna(0).astype(int)
    s.index.name = "semana"
    return s


def load_series(path=None) -> pd.DataFrame:
    """Lee un historial guardado (serie_semanal.csv)."""
    path = path or config.DATA_PROCESSED / "serie_semanal.csv"
    return pd.read_csv(path, parse_dates=["semana"]).set_index("semana")


def window_summary(series: pd.DataFrame, last_week: pd.Timestamp, weeks: int) -> dict:
    """Suma las `weeks` semanas que terminan en `last_week` (incluida)."""
    sub = series.loc[last_week - pd.Timedelta(weeks=weeks - 1): last_week]
    n, neg = int(sub["n"].sum()), int(sub["negativas"].sum())
    return {
        "n": n,
        "n_negative": neg,
        "pct_negative": neg / n if n else None,
        # promedio ponderado: cada semana pesa según cuántas reseñas tuvo
        "mean_rating": float((sub["n"] * sub["rating_medio"].fillna(0)).sum() / n) if n else None,
    }
