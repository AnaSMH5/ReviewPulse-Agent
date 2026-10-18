"""Corre el agente semana a semana contra la fuente SIMULADA (entorno estocástico).

    python run_agent.py --start 2025-03-17 --weeks 4 --seed 1 --p-fail 0.3 --p-missing 0.2 --oof
    python run_agent.py --start 2022-03-21 --weeks 10 --history-dir logs/demo

--oof        clasifica con predicciones fuera de muestra (sin fuga: el modelo no vio esa reseña). Recomendado para evaluar.
sin --oof    usa los modelos guardados en models/  (antes:  python train.py)
--history-dir  carpeta donde se escribe TODO (log, historial, resúmenes, avisos simulados): una por corrida.
"""
import argparse
from pathlib import Path

import pandas as pd

from reviewpulse.agent import ReviewPulseAgent, build_history
from reviewpulse.classifiers import ModelClassifier, OutOfFoldClassifier
from reviewpulse.data import load_reviews
from reviewpulse.source import SimulatedSource


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-03-17", help="lunes de la primera semana a simular")
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-fail", type=float, default=0.0, help="probabilidad de que la fuente falle en cada consulta")
    ap.add_argument("--p-missing", type=float, default=0.0, help="probabilidad de que a una reseña le falte la fecha")
    ap.add_argument("--p-blank", type=float, default=0.0, help="probabilidad de que una reseña llegue con el texto en blanco")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--history-dir", default=None)
    a = ap.parse_args(argv)

    df = load_reviews()
    inicio = pd.Timestamp(a.start)
    clasif = OutOfFoldClassifier(df) if a.oof else ModelClassifier()
    fuente = SimulatedSource(df, seed=a.seed, p_fail=a.p_fail, p_missing=a.p_missing, p_blank=a.p_blank)
    agente = ReviewPulseAgent(fuente, clasif, build_history(df, inicio, clasif),
                              history_dir=Path(a.history_dir) if a.history_dir else None, max_retries=a.max_retries)
    filas = []
    for semana in pd.date_range(inicio, periods=a.weeks, freq="7D"):
        r = agente.run_week(semana)
        filas.append({k: r[k] for k in ("semana", "estado", "completo", "status", "alerta", "n", "reintentos")})
    print("\n=== Resumen de la corrida ===")
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\nLog: {agente.dir / 'agent_log.jsonl'}")
    return filas


if __name__ == "__main__":
    main()