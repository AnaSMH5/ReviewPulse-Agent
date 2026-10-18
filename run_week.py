"""Corre el agente para UNA semana. Ejecuta:

    python run_week.py                 # la última semana con reseñas del histórico
    python run_week.py 2025-03-17      # una semana concreta -> simula esa corrida

Flujo:  cargar -> serie semanal -> comparar -> reporte -> GUARDAR -> publicar -> avisar (solo si hay alerta)
Se guarda ANTES de publicar y avisar: si falla la red, no se pierde lo calculado.
"""
import sys

import pandas as pd

from reviewpulse import notify, report, store
from reviewpulse.analysis import compare
from reviewpulse.data import load_reviews
from reviewpulse.metrics import weekly_series


def run(week=None) -> dict:
    df = load_reviews()
    # Si las reseñas fueran nuevas (sin tema), aquí iría:  df = classify.predict(df)

    if week:
        dia = pd.Timestamp(week).normalize()
        inicio = dia - pd.Timedelta(days=dia.weekday())          # si te dan un miércoles, usa su lunes
    else:
        inicio = df["week_start"].max()
    df = df[df["week_start"] <= inicio]                          # simula que no conocemos el futuro
    if df.empty:
        raise SystemExit(f"No hay reseñas hasta la semana del {inicio.date()}")

    serie = weekly_series(df)
    result = compare(serie, inicio)
    rep = report.build_report(df, result)
    rutas = store.save_run(rep)                                  # 1) guardar primero
    semana = rep["week"]

    pub = notify.publish_summary(rep)                            # 2) publicar en el sistema de reputación
    store.log_action(semana, "summary_published", pub)

    if rep["alert"]:                                             # 3) avisar al gerente solo si hay alerta
        if store.already_notified(semana):
            store.log_action(semana, "notification_skipped_duplicate")
        else:
            res = notify.notify_manager(rep)
            store.log_action(semana, "manager_notified", {**res, "actions": rep["actions"]})

    print(rep["markdown"])
    print(f"\nGuardado en: {rutas['summary']}")
    return rep


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
