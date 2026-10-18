"""PASO 4 - Clasificar reseñas: polaridad (1-5) y tema principal.

Hay DOS modelos independientes, ambos del mismo tipo:
    texto  ->  [TF-IDF]  ->  [Regresión logística]  ->  polaridad  (1..5)
                                                    ->  tema       (9 temas)

Cómo se usa:
    train(df_etiquetado)    UNA VEZ: aprende de reseñas que ya tienen rating y tema y guarda los modelos.
    predict(df_nuevo)       CADA SEMANA: añade polarity y topic a reseñas sin etiquetar.
    evaluate(df_etiquetado) mide qué tan bien acierta el modelo (para tu informe).
"""
import json
import shutil
from datetime import datetime

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

    TF-IDF mide qué palabras distinguen a una reseña de las demás. Usamos dos vistas:
      - palabras sueltas y pares de palabras ("aire acondicionado")
      - trozos de 2 a 5 letras (ayuda con errores de ortografía y variantes: limpio/limpieza)
    class_weight="balanced": como hay muchas más reseñas de 5★ que de 2★, el modelo
    le da más peso a las clases poco frecuentes para no ignorarlas.
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
def fit(df: pd.DataFrame) -> tuple:
    """Entrena los dos modelos y los DEVUELVE sin guardarlos (sirve para comparar antes de decidir).

    Cada modelo usa solo las filas que tienen su etiqueta: una corrección de tema sin rating
    sirve para el modelo de tema, y una reseña con rating sin tema sirve para el de polaridad.
    """
    con_rating = df.dropna(subset=["rating"])
    pol = make_model().fit(con_rating["text"], con_rating["rating"].astype(int))
    con_tema = df.dropna(subset=["main_topic"])
    topic = make_model().fit(con_tema["text"], con_tema["main_topic"])
    return pol, topic


def save(pol, topic, models_dir=None) -> None:
    pol_path, topic_path = _paths(models_dir)
    pol_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pol, pol_path)
    joblib.dump(topic, topic_path)


def train(df: pd.DataFrame, models_dir=None) -> None:
    """Entrena con TODAS las reseñas etiquetadas y guarda los dos modelos en disco."""
    save(*fit(df), models_dir)


def archive_models(models_dir=None):
    """Antes de reemplazar los modelos, guarda una copia de los actuales en models/archive/<fecha>/."""
    pol_path, topic_path = _paths(models_dir)
    if not (pol_path.exists() and topic_path.exists()):
        return None
    destino = pol_path.parent / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pol_path, destino)
    shutil.copy2(topic_path, destino)
    return destino


def score(pol, topic, df: pd.DataFrame) -> dict:
    """Mide dos modelos YA entrenados sobre un conjunto de prueba (necesita rating, main_topic y topics)."""
    y, p = df["rating"].to_numpy(), pol.predict(df["text"])
    pt = topic.predict(df["text"])
    return {
        "polaridad_mae": float(np.mean(np.abs(y - p))),
        "negativas_f1": float(f1_score(y <= config.NEGATIVE_MAX_RATING, p <= config.NEGATIVE_MAX_RATING)),
        "tema_estricto": float(accuracy_score(df["main_topic"], pt)),
        "tema_permisivo": float(np.mean([pred in lista for pred, lista in zip(pt, df["topics"])])),
    }


# ---------------------------------------------------------------- predecir
def predict(df: pd.DataFrame, models_dir=None) -> pd.DataFrame:
    """Añade: polarity, polarity_conf, topic, topic_conf, needs_review y valid_text.

    Un texto vacío o nulo (None / NaN / solo espacios) NO se clasifica: queda sin polaridad ni tema
    y se marca para revisión. Así el agente no se cae ni inventa una calificación.
    """
    pol_path, topic_path = _paths(models_dir)
    if not (pol_path.exists() and topic_path.exists()):
        raise FileNotFoundError("No hay modelos guardados. Ejecuta primero:  python train.py")

    pol, topic = joblib.load(pol_path), joblib.load(topic_path)
    out = df.copy()
    valido = out["text"].map(lambda t: isinstance(t, str) and t.strip() != "").astype(bool)
    out["valid_text"] = valido
    out["polarity"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["polarity_conf"] = np.nan
    out["topic"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out["topic_conf"] = np.nan

    if valido.any():
        texto = out.loc[valido, "text"]
        p = pol.predict_proba(texto)                          # probabilidad de cada rating
        out.loc[valido, "polarity"] = pol.classes_[p.argmax(axis=1)].astype(int)
        out.loc[valido, "polarity_conf"] = p.max(axis=1)
        t = topic.predict_proba(texto)
        out.loc[valido, "topic"] = topic.classes_[t.argmax(axis=1)]
        out.loc[valido, "topic_conf"] = t.max(axis=1)

    # Si el texto no es válido o el modelo duda del tema, se marca para que una persona lo revise
    out["needs_review"] = ~valido | (out["topic_conf"] < config.TOPIC_REVIEW_THRESHOLD)
    return out


# ---------------------------------------------------------------- evaluar
def evaluate(df: pd.DataFrame) -> dict:
    """Mide el acierto con validación cruzada.

    Con solo 300 reseñas, separar un 30% para probar deja ~90 casos: poco y muy ruidoso.
    La validación cruzada divide las reseñas en 5 grupos; entrena con 4 y predice el
    quinto, y repite hasta que TODAS las reseñas fueron predichas por un modelo que
    nunca las vio al entrenar. Así el resultado es honesto y más estable.
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
