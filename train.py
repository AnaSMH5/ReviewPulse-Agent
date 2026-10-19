"""Entrena y EVALÚA los modelos de polaridad y tema.

  python train.py        -> entrena solo con train.csv (210), evalúa en test.csv (90) y con
                            validación cruzada sobre las 300; guarda reports/classifier_metrics.json
  python train.py --all  -> igual, pero el modelo guardado se entrena con las 300 (despliegue)
"""
import argparse

from reviewpulse import classify, config, manifest
from reviewpulse.data import load_reviews, load_model_split

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="guardar el modelo entrenado con las 300 reseñas")
    a = ap.parse_args()

    todas, train, test = load_reviews(), load_model_split(
        "train"), load_model_split("test")
    classify.train(todas if a.all else train)           # primero se entrena y se guardan los modelos...
    manifest.write("train.csv + test.csv (300)" if a.all else "train.csv",   # ...y DESPUÉS se registra su huella
                   len(todas) if a.all else len(train), incluye_test=a.all)
    metricas = {"modelo_guardado_entrenado_con": len(todas) if a.all else len(train),
                "validacion_cruzada_300": classify.evaluate(todas),
                "holdout_test": classify.evaluate_holdout(train, test)}
    classify.save_metrics(metricas)
    print(f"Modelos guardados en {config.MODELS_DIR}")
    print(f"Métricas en {config.REPORTS / 'classifier_metrics.json'}")
    for k in ("validacion_cruzada_300", "holdout_test"):
        print(k, "polaridad:", metricas[k]["polaridad"]["accuracy"],
              "| tema estricta:", metricas[k]["tema"]["accuracy_estricta"])
