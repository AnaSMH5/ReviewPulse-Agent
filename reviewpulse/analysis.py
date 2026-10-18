from typing import cast

import pandas as pd
from scipy.stats import fisher_exact

from reviewpulse import config
from reviewpulse.metrics import window_summary


def compare(series: pd.DataFrame, week: pd.Timestamp) -> dict:
    w = config.WINDOW_WEEKS
    cur = window_summary(series, week, w)
    prev = window_summary(series, week - pd.Timedelta(weeks=w), w)

    result = {"week": week.date(), "current": cur, "previous": prev,
              "status": None, "alert": False, "reason": ""}

    if cur["n"] < config.MIN_REVIEWS or prev["n"] < config.MIN_REVIEWS:
        result["status"] = "datos_insuficientes"
        result["reason"] = f"Se necesitan >= {config.MIN_REVIEWS} reviews en cada ventana " \
                           f"(actual={cur['n']}, anterior={prev['n']})"
        return result

    rise = cur["pct_negative"] - prev["pct_negative"]
    result["delta_pct_negative"] = rise
    result["delta_mean_rating"] = cur["mean_rating"] - prev["mean_rating"]

    tabla = [[cur["n_negative"], cur["n"] - cur["n_negative"]],
             [prev["n_negative"], prev["n"] - prev["n_negative"]]]
    p = cast(float, fisher_exact(tabla, alternative="greater")[1])
    result["p_value"] = p

    if rise >= config.NEG_RISE and p < config.ALERT_P_VALUE:
        result["status"] = "alerta"
        result["alert"] = True
        result["reason"] = f"Negativas {prev['pct_negative']:.0%} -> {cur['pct_negative']:.0%} (p={p:.3f})"
    else:
        result["status"] = "normal"
    return result
