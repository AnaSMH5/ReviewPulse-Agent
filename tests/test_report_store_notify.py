"""Pruebas de report.py, store.py y notify.py con datos inventados (no dependen de tus CSV)."""
import json

import pandas as pd
import pytest

import run_week
from reviewpulse import config, notify, report, store
from reviewpulse.analysis import compare
from reviewpulse.metrics import weekly_series

SEMANA = pd.Timestamp("2025-12-22")      # lunes; las ventanas son de 26 semanas hacia atrás


@pytest.fixture(autouse=True)
def sin_canales(monkeypatch):
    """Que ninguna variable de entorno real active un envío durante las pruebas."""
    for v in ["REVIEWPULSE_WEBHOOK", "REVIEWPULSE_ENDPOINT", "REVIEWPULSE_TOKEN",
              "REVIEWPULSE_SMTP_HOST", "REVIEWPULSE_ALERT_TO"]:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(notify.time, "sleep", lambda s: None)


def _fila(fecha, rating, tema, texto="texto"):
    f = pd.Timestamp(fecha)
    return {"created_at": f, "week_start": f.normalize() - pd.Timedelta(days=f.weekday()),
            "rating": rating, "main_topic": tema, "text": texto}


@pytest.fixture
def df_empeora():
    # Ventana anterior (ene-jun): 20 reseñas, 2 negativas (10%)
    ant = [_fila(f"2025-01-{3 + i:02d}", 1 if i < 2 else 5, "Ubicación") for i in range(20)]
    # Ventana actual (jul-dic): 20 reseñas, 12 negativas (60%): 8 de Limpieza, 4 de Atención
    act = [_fila(f"2025-07-{3 + i:02d}", 1, "Limpieza e higiene", "Sucio. Escríbeme a juan@correo.com o al 300 123 4567") for i in range(8)]
    act += [_fila(f"2025-08-{3 + i:02d}", 2, "Atención y personal", "Mala atención") for i in range(4)]
    act += [_fila(f"2025-09-{3 + i:02d}", 5, "Ubicación", "Excelente ubicación") for i in range(8)]
    return pd.DataFrame(ant + act)


def _reporte(df):
    return report.build_report(df, compare(weekly_series(df), SEMANA))


# ------------------------------------------------------------------ report
def test_reporte_con_alerta_nombra_el_tema_que_explica(df_empeora):
    rep = _reporte(df_empeora)
    assert rep["alert"] is True and "ALERTA" in rep["markdown"]
    assert rep["actions"][0]["topic"] == "Limpieza e higiene"          # el que más negativas acumula
    assert "Limpieza e higiene" in rep["short"]


def test_el_ejemplo_enmascara_correos_y_telefonos(df_empeora):
    md = _reporte(df_empeora)["markdown"]
    assert "[email]" in md and "[tel]" in md
    assert "juan@correo.com" not in md and "300 123 4567" not in md


def test_lo_que_sale_del_agente_no_lleva_textos_de_huespedes(df_empeora):
    rep = _reporte(df_empeora)
    for destino in (rep["short"], json.dumps(rep["payload"])):
        assert "Sucio" not in destino and "juan" not in destino
    json.dumps(rep)                                                     # todo es serializable


def test_datos_insuficientes_no_alerta_y_lo_dice():
    df = pd.DataFrame([_fila("2025-12-01", 1, "Habitaciones")] * 3)
    rep = _reporte(df)
    assert rep["status"] == "datos_insuficientes" and rep["alert"] is False
    assert "Datos insuficientes" in rep["markdown"] and "pocas reseñas" in rep["markdown"]


# ------------------------------------------------------------------ store
def test_guardar_dos_veces_la_misma_semana_no_duplica(tmp_path, df_empeora):
    rep = _reporte(df_empeora)
    store.save_run(rep, tmp_path)
    rutas = store.save_run(rep, tmp_path)
    assert len(store.load_history(tmp_path)) == 1
    t = pd.read_csv(tmp_path / "topic_history.csv")
    assert not t.duplicated(["week", "topic"]).any()
    assert rutas["summary"].read_text(encoding="utf-8") == rep["markdown"]


def test_ya_notificado_solo_cuenta_envios_reales(tmp_path):
    assert store.already_notified("2025-12-22", tmp_path) is False
    store.log_action("2025-12-22", "manager_notified", {"mode": "dry_run"}, tmp_path)
    assert store.already_notified("2025-12-22", tmp_path) is False     # simulación no cuenta
    store.log_action("2025-12-22", "manager_notified", {"mode": "sent"}, tmp_path)
    assert store.already_notified("2025-12-22", tmp_path) is True
    assert store.already_notified("2025-12-29", tmp_path) is False


# ------------------------------------------------------------------ notify
def test_sin_configurar_es_simulacion(tmp_path, df_empeora):
    res = notify.notify_manager(_reporte(df_empeora), tmp_path)
    assert res["mode"] == "dry_run"
    assert (tmp_path / "outbox" / "2025-12-22_aviso.txt").exists()
    assert notify.publish_summary(_reporte(df_empeora))["mode"] == "dry_run"


def test_webhook_envia_el_mensaje_corto(monkeypatch, tmp_path, df_empeora):
    rep, enviado = _reporte(df_empeora), {}
    monkeypatch.setenv("REVIEWPULSE_WEBHOOK", "https://ejemplo.test/hook/SECRETO")
    monkeypatch.setattr(notify, "_post_json", lambda url, payload, headers=None: enviado.update(payload) or 200)
    res = notify.notify_manager(rep, tmp_path)
    assert res["mode"] == "sent" and res["channels"] == ["webhook"]
    assert enviado["text"] == rep["short"]


def test_un_envio_fallido_no_rompe_ni_filtra_la_url(monkeypatch, tmp_path, df_empeora):
    def falla(*a, **k):
        raise ConnectionError("no se pudo conectar a https://ejemplo.test/hook/SECRETO")
    monkeypatch.setenv("REVIEWPULSE_WEBHOOK", "https://ejemplo.test/hook/SECRETO")
    monkeypatch.setattr(notify, "_post_json", falla)
    res = notify.notify_manager(_reporte(df_empeora), tmp_path)
    assert res["mode"] == "error"
    assert "SECRETO" not in json.dumps(res)


def test_publicar_reintenta_y_manda_clave_de_idempotencia(monkeypatch, df_empeora):
    intentos = []

    def a_veces_falla(url, payload, headers=None):
        intentos.append(headers)
        if len(intentos) < 3:
            raise TimeoutError()
        return 200
    monkeypatch.setenv("REVIEWPULSE_ENDPOINT", "https://reputacion.test/api")
    monkeypatch.setattr(notify, "_post_json", a_veces_falla)
    res = notify.publish_summary(_reporte(df_empeora))
    assert res == {"mode": "sent", "attempts": 3, "error": None}
    assert intentos[-1]["Idempotency-Key"] == "reviewpulse-2025-12-22"


def test_url_que_no_es_http_se_rechaza():
    with pytest.raises(ValueError):
        notify._post_json("file:///etc/passwd", {})


# ------------------------------------------------------------------ de punta a punta con tus datos reales
def test_run_week_completo(monkeypatch, tmp_path):
    """Corre una semana que SÍ tuvo alerta en tu histórico y revisa que todo quede guardado."""
    monkeypatch.setattr(config, "HISTORY_DIR", tmp_path)
    rep = run_week.run("2025-03-19")                       # un miércoles: debe usar el lunes 2025-03-17
    assert rep["week"] == "2025-03-17" and rep["alert"] is True
    assert (tmp_path / "summaries" / "2025-03-17.md").exists()
    assert len(store.load_history(tmp_path)) == 1
    assert (tmp_path / "outbox" / "2025-03-17_aviso.txt").exists()       # simulado: no se envió nada
    eventos = [json.loads(l)["event"] for l in (tmp_path / "actions_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert eventos == ["summary_published", "manager_notified"]
