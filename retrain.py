"""Reentrena el agente con tus correcciones, y solo reemplaza el modelo si NO empeora.

    python retrain.py            # reentrena SOLO si encuentra correcciones nuevas en review_queue.csv
    python retrain.py --force    # reentrena con todas las correcciones guardadas, aunque no haya nuevas

El agente NO vigila el Excel: no se entera cuando lo editas. Se entera cuando corres este script, que
abre review_queue.csv y busca filas con `topic_correct` escrito. Cierra el Excel antes de correrlo.

Cómo decide (la parte importante):
    - data/processed/test.csv es un EXAMEN FIJO: nunca se usa para entrenar al comparar.
    - Se entrenan dos modelos: el "base" (solo train.csv) y el "candidato" (train.csv + tus correcciones).
    - Ambos hacen el examen. Si el candidato no empeora más de RETRAIN_MAX_DROP, se acepta.
    - Si se acepta: se guarda una copia del modelo anterior en models/archive/ y el modelo definitivo
      se entrena con TODO (train + test + correcciones).
    - Si no: se conserva el modelo actual y tus correcciones quedan guardadas para revisarlas.
"""
import json
import sys

import pandas as pd

from reviewpulse import classify, config, feedback


def _load_split(nombre: str) -> pd.DataFrame:
    raw = pd.read_csv(config.DATA_PROCESSED / nombre, encoding="utf-8-sig")
    return pd.DataFrame({"id": raw["id"].astype(str), "text": raw["texto_completo"], "rating": raw["rating"],
                         "main_topic": raw["tema_principal"], "topics": raw["temas"].str.split("|")})


def main(force: bool = False) -> dict:
    ap = feedback.apply_queue()
    print(f"Correcciones nuevas aceptadas: {ap['added']} | pendientes en la cola: {ap['pending']}")
    if ap["invalid"]:
        print(f"  ATENCIÓN: tema no válido en las reseñas {ap['invalid']}. Usa uno de los 9 temas oficiales.")
    if ap["added"] == 0 and not force:
        print("No encontré correcciones nuevas: no hay nada que aprender. "
              "(Si ya las escribiste, revisa que guardaste el archivo como CSV y con el mismo nombre.)")
        return {"decision": "sin_novedades"}

    train_df, test_df = _load_split("train.csv"), _load_split("test.csv")
    corr = feedback.load_corrections()
    corr = corr[~corr["id"].astype(str).isin(set(train_df["id"]) | set(test_df["id"]))]   # no duplicar ni contaminar el examen
    if corr.empty:
        print("No hay correcciones para aprender. Nada que hacer.")
        return {"decision": "sin_correcciones"}

    base = classify.fit(train_df)
    cand = classify.fit(pd.concat([train_df, corr], ignore_index=True))
    mb, mc = classify.score(*base, test_df), classify.score(*cand, test_df)

    tol = config.RETRAIN_MAX_DROP
    aceptado = (mc["negativas_f1"] >= mb["negativas_f1"] - tol
                and mc["tema_permisivo"] >= mb["tema_permisivo"] - tol
                and mc["polaridad_mae"] <= mb["polaridad_mae"] + tol)

    print(f"\n{'examen fijo (test.csv)':28s} {'base':>8s} {'candidato':>10s}")
    for k in mb:
        print(f"{k:28s} {mb[k]:8.3f} {mc[k]:10.3f}")

    if aceptado:
        final = classify.fit(pd.concat([train_df, test_df, corr], ignore_index=True))
        archivo = classify.archive_models()
        classify.save(*final)
        print(f"\nACEPTADO: modelo reemplazado ({len(corr)} correcciones). Copia anterior: {archivo}")
    else:
        print("\nRECHAZADO: el candidato empeora. Se conserva el modelo actual; tus correcciones siguen guardadas.")

    resultado = {"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "decision": "aceptado" if aceptado else "rechazado",
                 "n_correcciones": int(len(corr)), "base": {k: round(v, 3) for k, v in mb.items()},
                 "candidato": {k: round(v, 3) for k, v in mc.items()}}
    config.REPORTS.mkdir(exist_ok=True)
    with (config.REPORTS / "retrain_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(resultado, ensure_ascii=False) + "\n")
    return resultado


if __name__ == "__main__":
    main(force="--force" in sys.argv)
