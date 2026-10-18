"""ReviewPulse-Agent: agente basado en modelo con ciclo percepción -> decisión -> acción.

Los estados y acciones son los de la formalización AIMA (tabla de transiciones).
El estado interno (self.series, el historial semanal) es lo que lo hace "basado en modelo".

Qué hace el código y qué hacen las herramientas externas:
    - Código: ciclo, estados, comparación de ventanas (analysis.py), reporte (report.py), historial (store.py).
    - scikit-learn (TF-IDF + regresión logística): polaridad y tema (classify.py).
    - No hay LLM: el resumen se arma con plantillas.

Uso:  python train.py
      python -m reviewpulse.agent --start 2025-03-17 --weeks 4 --p-fail 0.3 --seed 1
"""
import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reviewpulse import classify, config, notify, report, store
from reviewpulse.analysis import compare
from reviewpulse.metrics import weekly_series
from reviewpulse.source import SimulatedSource, SourceError

# ACTIONS(s): una acción por estado (tabla de la plantilla AIMA)
ACTIONS = {
    "Inactivo": "consultar_fuente",
    "Lote_recibido": "preprocesar",
    "Lote_preprocesado": "clasificar",
    "Reseñas_clasificadas": "agregar_por_tema",
    "Resultados_agregados": "comparar_semanas",
    "Anomalía_detectada": "notificar_gerente",
    "Alerta_enviada": "publicar_resumen",
    "Sin_anomalía": "publicar_resumen",
    "Sin_reseñas": "registrar_semana_vacía",
    "Falla_fuente": "reintentar",
}
# por qué se elige esa acción (queda en el log)
REASON = {
    "Inactivo": "empieza el ciclo semanal: hay que pedir las reseñas nuevas",
    "Lote_recibido": "hay reseñas sin procesar: limpiar texto y revisar metadatos",
    "Lote_preprocesado": "texto listo: asignar polaridad y tema",
    "Reseñas_clasificadas": "reseñas etiquetadas: actualizar memoria y series",
    "Resultados_agregados": "series al día: comparar contra la ventana anterior",
    "Anomalía_detectada": "el aumento de negativas supera el umbral: avisar al gerente",
    "Alerta_enviada": "gerente avisado: publicar el resumen con la alerta",
    "Sin_anomalía": "no hay anomalía: publicar el resumen",
    "Sin_reseñas": "la semana no tuvo reseñas: dejar constancia",
    "Falla_fuente": "la fuente no respondió: reintentar mientras queden intentos",
}
GOAL = "Resumen_completado"


def build_history(labeled: pd.DataFrame, before: pd.Timestamp, classifier) -> pd.DataFrame:
    """Memoria inicial del agente: las reseñas anteriores a `before`, ya clasificadas."""
    return _clean(classifier.predict(labeled[labeled["week_start"] < before]))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Quita lo que no se pudo clasificar y deja polarity como entero (lo que esperan metrics y report)."""
    d = df.dropna(subset=["polarity", "topic"]).copy()
    d["polarity"] = d["polarity"].astype(int)
    return d.reset_index(drop=True)


class ReviewPulseAgent:
    """Agente basado en modelos que procesa una semana de reseñas simuladas."""

    def __init__(self, source, classifier, history: pd.DataFrame, history_dir=None,
                 max_retries: int = 3, backoff_s: float = 0.0, run_id: str | None = None):
        self.source, self.classifier = source, classifier
        self.reviews = _clean(
            history) if "polarity" in history else history.copy()
        self.dir = Path(history_dir or config.HISTORY_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_retries, self.backoff_s = max_retries, backoff_s
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self._reset(None)

    def _reset(self, week):
        self.week, self.state, self.retries = week, "Inactivo", 0
        self.batch = self.n = self.fuente = self.result = self.rep = None
        self.outcome = {}

    # ------------------------------------------------------------ log
    def _log(self, kind: str, **data):
        fila = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "run_id": self.run_id,
                "semana": self.week.date().isoformat() if self.week is not None else None,
                "tipo": kind, "estado": self.state, **report.to_plain(data)}
        with (self.dir / "agent_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ ciclo

    def run_week(self, week) -> dict:
        """Procesa una semana hasta la meta o hasta agotar los reintentos de la fuente."""
        dia = pd.Timestamp(week).normalize()
        # siempre el lunes de esa semana
        self._reset(dia - pd.Timedelta(days=dia.weekday()))
        self._log("inicio_ciclo")
        while self.state != GOAL:
            if self.state == "Falla_fuente" and self.retries >= self.max_retries:
                self._log("abandono", motivo="reintentos agotados",
                          intentos=self.retries + 1)
                store.log_action(self.week.date().isoformat(), "source_failure_abandoned",
                                 {"attempts": self.retries + 1}, self.dir)
                break
            percept = self.perceive()
            action = self.decide(percept)
            self.act(action, percept)
        self._log("fin_ciclo", completo=self.state == GOAL)
        return self.summary()

    # PERCEPCIÓN: lo único que el agente "ve" del entorno
    def perceive(self) -> dict:
        if self.state == "Inactivo":
            try:
                self.batch = self.source.fetch(self.week)
                self.fuente, self.n = True, len(self.batch)
            except SourceError as e:
                self.batch, self.fuente, self.n = None, False, None
                self._log("error_fuente", detalle=str(e))
        percept = {"fuente": self.fuente, "n": self.n,
                   "comparacion": self.result["status"] if self.result else None}
        self._log("percepcion", **percept)
        return percept

    # DECISIÓN: elige la acción disponible en el estado actual (ACTIONS(s))

    def decide(self, percept: dict) -> str:
        action = ACTIONS[self.state]
        self._log("decision", accion=action,
                  percepcion=percept, razon=REASON[self.state])
        return action

    # ACCIÓN: ejecuta y devuelve el nuevo estado (RESULT(s, a))

    def act(self, action: str, percept: dict) -> None:
        anterior = self.state
        try:
            self.state = getattr(self, f"_a_{action}")(percept)
        except Exception as e:                                       # un fallo inesperado queda en el log y se propaga
            self._log("error", accion=action,
                      detalle=f"{type(e).__name__}: {e}")
            raise
        self._log("accion", accion=action, desde=anterior, hacia=self.state)

    # ------------------------------------------------------------ acciones
    def _a_consultar_fuente(self, percept):
        if not percept["fuente"]:
            return "Falla_fuente"
        return "Lote_recibido" if percept["n"] > 0 else "Sin_reseñas"

    def _a_preprocesar(self, percept):
        b = self.batch.copy()
        b["text"] = b["text"].map(lambda t: re.sub(
            r"\s+", " ", t).strip() if isinstance(t, str) else "")
        sin_texto, sin_fecha = b["text"] == "", b["created_at"].isna()
        # se marca y la reseña sigue en el lote
        b["meta_incompleta"] = sin_fecha
        # sin fecha exacta: se asume el inicio de su semana
        b.loc[sin_fecha, "created_at"] = self.week
        # sin texto no hay nada que clasificar
        b = b[~sin_texto].reset_index(drop=True)
        self._log("preproceso", recibidas=int(len(sin_texto)), sin_texto=int(sin_texto.sum()),
                  meta_incompleta=int(sin_fecha.sum()), quedan=int(len(b)))
        self.batch = b
        # extensión documentada
        return "Lote_preprocesado" if len(b) else "Sin_reseñas"

    def _a_clasificar(self, percept):
        self.batch = _clean(self.classifier.predict(self.batch))
        self._log("clasificacion", n=int(len(self.batch)),
                  a_revision=int(self.batch["needs_review"].sum()))
        return "Reseñas_clasificadas"

    def _a_agregar_por_tema(self, percept):
        b = self.batch
        self.reviews = pd.concat(
            [self.reviews[self.reviews["week_start"] != self.week], b], ignore_index=True)
        self.series = weekly_series(self.reviews)
        por_tema = b.groupby("topic")["polarity"].agg(
            n="size", negativas=lambda x: int((x <= config.NEGATIVE_MAX_RATING).sum()))
        self._log("agregacion", por_tema=por_tema.to_dict("index"))
        return "Resultados_agregados"

    def _a_comparar_semanas(self, percept):
        self.result = compare(self.series, self.week)
        self.rep = report.build_report(self.reviews, self.result)
        self._log("comparacion", status=self.result["status"],
                  alerta=self.result["alert"], motivo=self.result["reason"])
        return "Anomalía_detectada" if self.result["alert"] else "Sin_anomalía"

    def _a_notificar_gerente(self, percept):
        w = self.rep["week"]
        # no avisar dos veces lo mismo
        if store.already_notified(w, self.dir):
            store.log_action(
                w, "notification_skipped_duplicate", None, self.dir)
            self.outcome["notificacion"] = "duplicado_omitido"
        else:
            res = notify.notify_manager(self.rep, self.dir)
            store.log_action(w, "manager_notified", {
                             **res, "actions": self.rep["actions"]}, self.dir)
            self.outcome["notificacion"] = res["mode"]
        return "Alerta_enviada"

    def _a_publicar_resumen(self, percept):
        w = self.rep["week"]
        # 1) guardar primero: si falla la red no se pierde
        store.save_run(self.rep, self.dir)
        # 2) publicar en el sistema de reputación
        pub = notify.publish_summary(self.rep)
        store.log_action(w, "summary_published", pub, self.dir)
        self.outcome["publicacion"] = pub["mode"]
        return GOAL

    def _a_registrar_semana_vacía(self, percept):
        w = self.week.date().isoformat()
        if self.reviews.empty:
            store.log_action(w, "empty_week_recorded", {
                             "sin_historial": True}, self.dir)
            return GOAL
        # las ventanas vigentes siguen existiendo
        result = compare(weekly_series(self.reviews), self.week)
        result.update(status="sin_reseñas", alert=False,
                      reason="Semana sin reseñas nuevas; se muestra el estado de las ventanas vigentes.")
        self.result, self.rep = result, report.build_report(
            self.reviews, result)
        store.save_run(self.rep, self.dir)
        pub = notify.publish_summary(self.rep)
        store.log_action(w, "empty_week_recorded", pub, self.dir)
        self.outcome["publicacion"] = pub["mode"]
        return GOAL

    def _a_reintentar(self, percept):
        self.retries += 1
        if self.retries > self.max_retries:
            self._log("reintentos_agotados", max_retries=self.max_retries)
            return GOAL
        self._log("reintento", numero=self.retries,
                  max_retries=self.max_retries)
        return "Inactivo"

    def summary(self) -> dict:
        return {"semana": self.week.date().isoformat(), "estado": self.state, "completo": self.state == GOAL,
                "alerta": bool(self.result and self.result["alert"]), "status": self.result and self.result["status"],
                "n": self.n, "reintentos": self.retries, "reporte": self.rep, **self.outcome}
