"""Pruebas con datos inventados: así sabes que la lógica funciona sin depender de tu CSV."""
import pandas as pd

from reviewpulse.analysis import compare


def _reviews(inicio, ratings):
    fechas = pd.date_range(inicio, periods=len(ratings), freq="D")
    return pd.DataFrame({"created_at": fechas, "rating": ratings})


def test_alerta_cuando_empeora():
    antes = _reviews("2025-01-01", [5] * 18 + [1] * 2)      # ~10% negativas
    ahora = _reviews("2025-07-01", [1] * 12 + [5] * 8)      # 60% negativas
    df = pd.concat([antes, ahora])
    r = compare(df, pd.Timestamp("2025-07-20"))
    assert r["alert"] is True


def test_sin_alerta_si_no_cambia():
    antes = _reviews("2025-01-01", [5, 5, 1, 5, 4, 5, 5, 4, 5, 1] * 2)
    ahora = _reviews("2025-07-01", [5, 5, 1, 5, 4, 5, 5, 4, 5, 1] * 2)
    r = compare(pd.concat([antes, ahora]), pd.Timestamp("2025-07-20"))
    assert r["alert"] is False


def test_pocas_reviews_no_concluye():
    df = _reviews("2025-07-01", [1, 1, 1])
    r = compare(df, pd.Timestamp("2025-07-03"))
    assert r["status"] == "datos_insuficientes"
