import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline, make_union

from reviewpulse import config


# ---------------------------------------------------------------- el modelo
def make_model():
    """Convierte texto en números (TF-IDF) y aprende con regresión logística.
    """
    palabras = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    letras = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, strip_accents="unicode")
    return make_pipeline(
        make_union(palabras, letras),
        LogisticRegression(C=10, max_iter=3000, class_weight="balanced"),
    )


def _paths(models_dir):
    d = models_dir or config.MODELS_DIR
    return d / "polarity.joblib", d / "topic.joblib"


# ---------------------------------------------------------------- entrenar
def train(df: pd.DataFrame, models_dir=None) -> None:
    """Entrena con TODAS las reseñas etiquetadas y guarda los dos modelos en disco."""
    pol_path, topic_path = _paths(models_dir)
    pol_path.parent.mkdir(parents=True, exist_ok=True)

    pol = make_model().fit(df["text"], df["rating"])
    joblib.dump(pol, pol_path)

    con_tema = df.dropna(subset=["main_topic"])
    topic = make_model().fit(con_tema["text"], con_tema["main_topic"])
    joblib.dump(topic, topic_path)


# ---------------------------------------------------------------- predecir
def predict(df: pd.DataFrame, models_dir=None) -> pd.DataFrame:
    """Añade 4 columnas: polarity, polarity_conf, topic, topic_conf (+ needs_review)."""
    pol_path, topic_path = _paths(models_dir)
    if not (pol_path.exists() and topic_path.exists()):
        raise FileNotFoundError("No hay modelos guardados. Ejecuta primero:  python train.py")

    pol, topic = joblib.load(pol_path), joblib.load(topic_path)
    out = df.copy()

    p = pol.predict_proba(out["text"])                       # probabilidad de cada rating
    out["polarity"] = pol.classes_[p.argmax(axis=1)].astype(int)
    out["polarity_conf"] = p.max(axis=1)

    t = topic.predict_proba(out["text"])
    out["topic"] = topic.classes_[t.argmax(axis=1)]
    out["topic_conf"] = t.max(axis=1)

    # Si el modelo duda del tema, lo marcamos para que una persona lo revise
    out["needs_review"] = out["topic_conf"] < config.TOPIC_REVIEW_THRESHOLD
    return out


# ---------------------------------------------------------------- evaluar
def evaluate(df: pd.DataFrame) -> dict:
    """Mide el acierto con validación cruzada.
    """
    # ---- polaridad
    y = df["rating"].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    p = cross_val_predict(make_model(), df["text"].to_numpy(), y, cv=cv)
    neg_real, neg_pred = y <= config.NEGATIVE_MAX_RATING, p <= config.NEGATIVE_MAX_RATING
    y3 = np.where(y <= 2, 0, np.where(y == 3, 1, 2))
    p3 = np.where(p <= 2, 0, np.where(p == 3, 1, 2))
    polaridad = {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, p),
        "baseline_accuracy": float(pd.Series(y).value_counts(normalize=True).iloc[0]),
        "accuracy_within_1": float(np.mean(np.abs(y - p) <= 1)),
        "mae": float(np.mean(np.abs(y - p))),
        "f1_macro": f1_score(y, p, average="macro"),
        "accuracy_3_clases": accuracy_score(y3, p3),
        "negativas_precision": precision_score(neg_real, neg_pred),
        "negativas_recall": recall_score(neg_real, neg_pred),
        "negativas_f1": f1_score(neg_real, neg_pred),
    }

    # ---- tema
    d = df.dropna(subset=["main_topic"])
    t = d["main_topic"].to_numpy()
    cv_t = StratifiedKFold(4, shuffle=True, random_state=config.RANDOM_STATE)
    pt = cross_val_predict(make_model(), d["text"].to_numpy(), t, cv=cv_t)
    tema = {
        "n": int(len(t)),
        "accuracy_estricta": accuracy_score(t, pt),
        "baseline_accuracy": float(pd.Series(t).value_counts(normalize=True).iloc[0]),
        "f1_macro": f1_score(t, pt, average="macro"),
        # Permisiva: acierta si el tema predicho es UNO de los temas que la reseña menciona.
        "accuracy_permisiva": float(np.mean([pred in lista for pred, lista in zip(pt, d["topics"])])),
    }
    return {"polaridad": {k: round(float(v), 3) for k, v in polaridad.items()},
            "tema": {k: round(float(v), 3) for k, v in tema.items()}}


def save_metrics(metrics: dict, path=None) -> None:
    path = path or config.REPORTS / "classifier_metrics.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
