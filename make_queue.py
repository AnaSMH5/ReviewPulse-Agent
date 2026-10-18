"""Clasifica reseñas nuevas y manda a la cola de revisión las que el modelo duda.

    python make_queue.py data/inbox/nuevas.csv

El archivo debe tener el formato original del sitio (como data/raw/reviews.csv).
Antes necesitas tener los modelos:  python train.py
"""
import sys

from reviewpulse import classify, feedback
from reviewpulse.data import load_raw_reviews


def main(path):
    df = load_raw_reviews(path)
    pred = classify.predict(df)
    n = feedback.export_queue(pred)
    print(f"{len(df)} reseñas clasificadas | {int(pred['needs_review'].sum())} con duda | {n} nuevas en la cola")
    print("Abre data/feedback/review_queue.csv, escribe el tema correcto en `topic_correct` y luego corre:  python retrain.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python make_queue.py <archivo_de_reseñas.csv>")
    main(sys.argv[1])
