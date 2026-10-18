"""Pruebas del agente de punta a punta contra la fuente simulada (sin tocar tu carpeta history/)."""
import json
from datetime import datetime

import pandas as pd
import pytest

from reviewpulse.agent import GOAL, ReviewPulseAgent, build_history
from reviewpulse.analysis import compare
from reviewpulse.classifiers import OutOfFoldClassifier
from reviewpulse.data import load_reviews
from reviewpulse.metrics import weekly_series
from reviewpulse.source import SimulatedSource

DF = load_reviews()
# predicciones fuera de muestra: no necesita models/
OOF = OutOfFoldClassifier(DF)
SERIE = weekly_series(DF)


def _semana(con_reseñas: bool, evaluable: bool = True):
    """Primera semana que cumple lo pedido (según histórico)."""
    for s in SERIE.index[60:]:
        if (SERIE.loc[s, "n"] > 0) == con_reseñas:
            if not con_reseñas or not evaluable or compare(SERIE, s)["status"] != "datos_insuficientes":
                return s
    raise AssertionError("no hay una semana así en el histórico")


def _agente(tmp_path, semana, **fuente):
    return ReviewPulseAgent(SimulatedSource(DF, **fuente), OOF, build_history(DF, semana, OOF), history_dir=tmp_path)


def _log(tmp_path):
    return [json.loads(l) for l in (tmp_path / "agent_log.jsonl").read_text(encoding="utf-8").splitlines()]


def test_el_agente_no_ve_las_etiquetas():
    lote = SimulatedSource(DF).fetch(_semana(True))
    assert not {"rating", "main_topic", "topics"} & set(lote.columns)


def test_semana_normal_llega_a_la_meta_y_guarda_todo(tmp_path):
    s = _semana(True)
    r = _agente(tmp_path, s).run_week(s)
    assert r["estado"] == GOAL and r["completo"] and r["n"] > 0
    assert (tmp_path / "summaries" / f"{s.date()}.md").exists()
    assert len(store_history(tmp_path)) == 1


def store_history(tmp_path):
    return pd.read_csv(tmp_path / "weekly_history.csv")


def test_pico_de_negativas_dispara_alerta_y_aviso(tmp_path):
    s = _semana(True)
    r = _agente(tmp_path, s, spikes={s: 60}).run_week(s)
    assert r["alerta"] is True and r["notificacion"] == "dry_run"
    estados = [f["hacia"] for f in _log(tmp_path) if f["tipo"] == "accion"]
    assert "Anomalía_detectada" in estados and "Alerta_enviada" in estados and estados[-1] == GOAL
    assert (tmp_path / "outbox" / f"{s.date()}_aviso.txt").exists()


def test_semana_sin_reseñas_deja_constancia(tmp_path):
    s = _semana(False)
    r = _agente(tmp_path, s).run_week(s)
    assert r["completo"] and r["status"] == "sin_reseñas" and r["alerta"] is False
    assert "Sin_reseñas" in [f["hacia"]
                             for f in _log(tmp_path) if f["tipo"] == "accion"]


def test_caida_de_la_fuente_se_recupera_con_reintentos(tmp_path):
    s = _semana(True)
    r = _agente(tmp_path, s, fail_next=2).run_week(s)
    assert r["completo"] and r["reintentos"] == 2
    assert sum(f["tipo"] == "error_fuente" for f in _log(tmp_path)) == 2


def test_si_la_fuente_nunca_responde_abandona_sin_decir_que_termino(tmp_path):
    s = _semana(True)
    ag = _agente(tmp_path, s, fail_next=99)
    ag.max_retries = 2
    r = ag.run_week(s)
    assert r["completo"] is False and r["estado"] == "Falla_fuente" and r["reintentos"] == 2
    assert sum(f["tipo"] == "error_fuente" for f in _log(
        tmp_path)) == 3          # 1 intento + 2 reintentos
    assert any(f["tipo"] == "abandono" for f in _log(tmp_path))
    # no se publicó ningún resumen
    assert not list(tmp_path.glob("summaries/*.md"))


def test_metadatos_incompletos_se_marcan_y_no_se_pierden(tmp_path):
    s = _semana(True)
    ag = _agente(tmp_path, s, p_missing=1.0)
    r = ag.run_week(s)
    pre = next(f for f in _log(tmp_path) if f["tipo"] == "preproceso")
    assert r["completo"] and pre["meta_incompleta"] == pre["recibidas"] == pre["quedan"] > 0


def test_textos_en_blanco_se_descartan_y_no_rompen(tmp_path):
    s = _semana(True)
    r = _agente(tmp_path, s, p_blank=1.0).run_week(s)
    # nada clasificable: se trata como semana vacía
    assert r["completo"] and r["status"] == "sin_reseñas"


def test_el_log_trae_percepcion_decision_y_accion_con_marca_de_tiempo(tmp_path):
    s = _semana(True)
    _agente(tmp_path, s).run_week(s)
    filas = _log(tmp_path)
    assert {"percepcion", "decision", "accion"} <= {f["tipo"] for f in filas}
    tiempos = [datetime.fromisoformat(f["ts"]) for f in filas]
    assert tiempos == sorted(tiempos) and all(t.tzinfo for t in tiempos)
    assert all(f["razon"] for f in filas if f["tipo"] == "decision")


def test_correr_dos_veces_la_misma_semana_no_duplica(tmp_path):
    s = _semana(True)
    ag = _agente(tmp_path, s)
    ag.run_week(s)
    n1 = len(ag.reviews)
    ag.run_week(s)
    assert len(ag.reviews) == n1 and len(store_history(tmp_path)) == 1


def test_un_miercoles_se_procesa_como_su_lunes(tmp_path):
    s = _semana(True)
    r = _agente(tmp_path, s).run_week(s + pd.Timedelta(days=2))
    assert r["semana"] == str(s.date()) and r["n"] > 0
