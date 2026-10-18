"""Entrena el clasificador. Ejecuta:  python train.py

1) Mide qué tan bien acierta (validación cruzada)  -> reports/classifier_metrics.json
2) Entrena con todas las reseñas y guarda          -> models/polarity.joblib, models/topic.joblib
3) Prueba el modelo con reseñas inventadas
"""
import pandas as pd

from reviewpulse import classify
from reviewpulse.data import load_reviews


def main():
    df = load_reviews()                      # reviews_con_temas.csv (trae rating y temas)

    print("Evaluando (puede tardar ~1 minuto)...")
    m = classify.evaluate(df)
    classify.save_metrics(m)
    for bloque, vals in m.items():
        print(f"\n{bloque.upper()}")
        for k, v in vals.items():
            print(f"  {k:22s} {v}")

    print("\nEntrenando con todas las reseñas y guardando modelos...")
    classify.train(df)

    ejemplos = pd.DataFrame({"text": [
        "Habitación sucia, con cucarachas en el baño y el aire acondicionado dañado. Pésimo.",
        "Excelente atención del personal, muy amables, y la comida deliciosa. Volveremos.",
        "Buena ubicación frente a la playa pero el desayuno es repetitivo.",
    ]})
    print("\nPrueba con reseñas nuevas:")
    res = classify.predict(ejemplos)
    print(res[["polarity", "topic", "topic_conf", "needs_review"]].round(2).to_string())


if __name__ == "__main__":
    main()
