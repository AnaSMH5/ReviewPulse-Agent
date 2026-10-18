"""Fuente simulada de reseñas (el entorno del agente).

Entrega las reseñas de una semana SIN rating ni tema (el agente solo percibe lo que percibiría en
producción) y puede, de forma aleatoria y reproducible (semilla):
    - caerse (SourceError)                          -> p_fail, o fail_next=N para forzar N caídas seguidas
    - traer reseñas sin fecha (metadatos faltantes) -> p_missing
    - traer reseñas con el texto en blanco          -> p_blank
    - traer un pico de reseñas negativas inyectado  -> spikes={semana: cuántas}  (para la fase 3)
"""
import numpy as np
import pandas as pd

from reviewpulse import config

COLS = ["id", "text", "created_at", "week_start"]


class SourceError(Exception):
    """La fuente no está disponible (caída o error de conexión)."""


class SimulatedSource:
    def __init__(self, df: pd.DataFrame, seed: int = 0, p_fail: float = 0.0, p_missing: float = 0.0,
                 p_blank: float = 0.0, fail_next: int = 0, spikes: dict | None = None):
        self._df = df
        self._neg = df[df["rating"] <= config.NEGATIVE_MAX_RATING]
        self._rng = np.random.default_rng(seed)
        self.p_fail, self.p_missing, self.p_blank = p_fail, p_missing, p_blank
        self.fail_next = fail_next
        self.spikes = {pd.Timestamp(k): int(v)
                       for k, v in (spikes or {}).items()}

    def fetch(self, week: pd.Timestamp) -> pd.DataFrame:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise SourceError("fuente no disponible (falla forzada)")
        if self._rng.random() < self.p_fail:
            raise SourceError("fuente no disponible (falla aleatoria)")

        sub = self._df[self._df["week_start"] == week][COLS].copy()
        sub["id"] = sub["id"].astype(str)

        k = self.spikes.get(pd.Timestamp(week), 0)
        # pico de negativas con verdad conocida
        if k > 0 and len(self._neg):
            extra = self._neg.sample(n=k, replace=True, random_state=int(
                self._rng.integers(1 << 30)))[COLS].copy()
            # id único; el sufijo # lo reconoce el clasificador fuera de muestra
            extra["id"] = [f"{i}#{j}" for j, i in enumerate(extra["id"])]
            extra["week_start"] = pd.Timestamp(week)
            extra["created_at"] = pd.Timestamp(
                week) + pd.to_timedelta(self._rng.integers(0, 6 * 86400, size=k), unit="s")
            sub = pd.concat([sub, extra], ignore_index=True)

        sub.loc[self._rng.random(len(sub)) <
                self.p_missing, "created_at"] = pd.NaT
        sub.loc[self._rng.random(len(sub)) < self.p_blank, "text"] = ""
        return sub.reset_index(drop=True)
