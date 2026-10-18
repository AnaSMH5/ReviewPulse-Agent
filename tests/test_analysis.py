"""Pruebas con datos inventados: así sabes que la lógica funciona sin depender de tu CSV."""
import pandas as pd

from reviewpulse.analysis import compare
from reviewpulse.metrics import weekly_series


def _reviews(inicio, ratings):
    """Una reseña por día empezando en `inicio`."""
    fechas = pd.date_range(inicio, periods=len(ratings), freq="D")
    semana = fechas.normalize() - pd.to_timedelta(fechas.weekday, unit="D")
    return pd.DataFrame({"created_at": fechas, "week_start": semana, "rating": ratings})


def test_alerta_cuando_empeora():
    antes = _reviews("2025-01-06", [5] * 90 + [1] * 10)       # 10% negativas, ~14 semanas
    ahora = _reviews("2025-07-14", [1] * 60 + [5] * 40)       # 60% negativas
    serie = weekly_series(pd.concat([antes, ahora]))
    r = compare(serie, serie.index[-1])
    assert r["alert"] is True


def test_sin_alerta_si_no_cambia():
    patron = [5, 5, 1, 5, 4, 5, 5, 4, 5, 1] * 10
    serie = weekly_series(pd.concat([_reviews("2025-01-06", patron), _reviews("2025-07-14", patron)]))
    assert compare(serie, serie.index[-1])["alert"] is False


def test_pocas_reviews_no_concluye():
    serie = weekly_series(_reviews("2025-07-07", [1, 1, 1]))
    assert compare(serie, serie.index[-1])["status"] == "datos_insuficientes"
