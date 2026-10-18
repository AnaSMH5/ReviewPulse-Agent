"""Dos formas de clasificar para el agente (misma interfaz: predict(df) -> df con polarity y topic).

ModelClassifier      -> usa los modelos guardados en models/ (producción).
OutOfFoldClassifier  -> predicciones FUERA DE MUESTRA: cada reseña la predice un modelo que no la vio al
                        entrenar (validación cruzada). Sirve para evaluar el agente sin fuga de datos,
                        porque el dataset es el mismo del modelo y no hay otro conjunto aparte.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from reviewpulse import classify, config


class ModelClassifier:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return classify.predict(df, self.models_dir)


class OutOfFoldClassifier:
    def __init__(self, labeled: pd.DataFrame, seed: int = config.RANDOM_STATE):
        ids = labeled["id"].astype(str).to_numpy()
        X = labeled["text"].to_numpy()
        pol_p = cross_val_predict(classify.make_model(), X, labeled["rating"].astype(int).to_numpy(),
                                  cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")
        pol_cls = np.unique(labeled["rating"].astype(int))
        d = labeled.dropna(subset=["main_topic"])
        top_p = cross_val_predict(classify.make_model(), d["text"].to_numpy(), d["main_topic"].to_numpy(),
                                  cv=StratifiedKFold(4, shuffle=True, random_state=seed), method="predict_proba")
        top_cls = np.unique(d["main_topic"])
        self._pol = {i: (int(pol_cls[p.argmax()]), float(p.max())) for i, p in zip(ids, pol_p)}
        self._top = {i: (top_cls[p.argmax()], float(p.max())) for i, p in zip(d["id"].astype(str), top_p)}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        base = out["id"].astype(str).str.split("#").str[0]        # los ids con # son copias de una reseña real
        valido = out["text"].map(lambda t: isinstance(t, str) and t.strip() != "").astype(bool)
        out["valid_text"] = valido
        pol = [self._pol.get(i) if v else None for i, v in zip(base, valido)]
        top = [self._top.get(i) if v else None for i, v in zip(base, valido)]
        out["polarity"] = pd.array([p[0] if p else pd.NA for p in pol], dtype="Int64")
        out["polarity_conf"] = [p[1] if p else np.nan for p in pol]
        out["topic"] = pd.Series([t[0] if t else None for t in top], index=out.index, dtype=object)
        out["topic_conf"] = [t[1] if t else np.nan for t in top]
        out["needs_review"] = ~valido | (out["topic_conf"] < config.TOPIC_REVIEW_THRESHOLD)
        return out