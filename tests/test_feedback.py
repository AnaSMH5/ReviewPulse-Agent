"""Pruebas del ciclo de aprendizaje: cola de revisión -> correcciones -> reentrenamiento."""
import json

import pandas as pd
import pytest

import make_queue
import retrain
from reviewpulse import classify, config, feedback
from reviewpulse.data import load_reviews

NUEVAS = [  # reseñas inventadas, en el formato original del sitio
    ("Moho en el baño", "La habitación tenía moho en el baño y el aire acondicionado no enfriaba nada.", 2),
    ("Piscina cerrada", "La piscina estuvo cerrada tres días y en la playa no había sillas disponibles.", 2),
    ("Muy amables", "El personal de recepción fue muy amable y nos ayudó con todo desde el primer momento.", 5),
    ("Buffet repetitivo", "El desayuno era siempre lo mismo y los jugos sabían aguados.", 3),
    ("Ruido", "El ascensor estaba dañado y las instalaciones necesitan mantenimiento urgente.", 2),
    ("Cobro extra", "Nos cobraron un impuesto adicional que no estaba en la reserva, caro para lo que ofrecen.", 1),
]


@pytest.fixture
def carpetas(tmp_path, monkeypatch):
    """Todo lo que escribe el ciclo va a una carpeta temporal, nunca a tu proyecto."""
    monkeypatch.setattr(config, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    return tmp_path


def _archivo_nuevas(tmp_path):
    df = pd.DataFrame([{"Id": 9000000 + i, "Rating": r, "Review Title": t, "Review Text": x,
                        "Created Date": "2026-03-10T15:00:00Z"} for i, (t, x, r) in enumerate(NUEVAS)])
    ruta = tmp_path / "nuevas.csv"
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    return ruta


def _cola_corregida(temas):
    """Escribe una cola con los temas ya llenados, como si una persona los hubiera corregido."""
    filas = [{"id": str(9000000 + i), "text": f"{t}. {x}", "rating": r, "topic_predicted": "?", "topic_conf": 0.1,
              "topic_correct": tema} for i, ((t, x, r), tema) in enumerate(zip(NUEVAS, temas))]
    d = feedback._dir(None)
    feedback._write(pd.DataFrame(filas)[feedback.QUEUE_COLS], d / feedback.QUEUE_FILE)


TEMAS_OK = ["Habitaciones", "Piscina, playa y entretenimiento", "Atención y personal",
            "Comida y bebidas", "Mantenimiento e instalaciones", "Precio / relación calidad-precio"]


def test_los_temas_oficiales_cubren_todos_los_de_tus_datos():
    assert set(load_reviews()["main_topic"]) <= set(feedback.TOPICS)


def test_la_cola_solo_recibe_las_dudosas_y_no_repite(carpetas, monkeypatch):
    classify.train(load_reviews())
    monkeypatch.setattr(config, "TOPIC_REVIEW_THRESHOLD", 1.1)          # fuerza que todas queden "dudosas"
    ruta = _archivo_nuevas(carpetas)
    make_queue.main(str(ruta))
    cola = pd.read_csv(config.FEEDBACK_DIR / feedback.QUEUE_FILE, sep=";", encoding="utf-8-sig")
    assert len(cola) == len(NUEVAS) and cola["topic_correct"].isna().all()
    make_queue.main(str(ruta))                                           # segunda vez: no duplica
    assert len(pd.read_csv(config.FEEDBACK_DIR / feedback.QUEUE_FILE, sep=";", encoding="utf-8-sig")) == len(NUEVAS)


def test_solo_se_aceptan_los_temas_oficiales(carpetas):
    temas = ["atencion y personal", "TEMA INVENTADO", "", "Habitaciones", "", ""]   # sin tildes/mayúsculas, inválido, vacío
    _cola_corregida(temas)
    r = feedback.apply_queue()
    assert r == {"added": 2, "pending": 3, "invalid": ["9000001"]}
    corr = feedback.load_corrections()
    assert sorted(corr["main_topic"]) == ["Atención y personal", "Habitaciones"]      # quedó con el nombre oficial
    cola = pd.read_csv(config.FEEDBACK_DIR / feedback.QUEUE_FILE, sep=";", encoding="utf-8-sig")
    assert len(cola) == 4                                                              # 3 pendientes + 1 inválida siguen ahí


def test_retrain_acepta_archiva_el_modelo_anterior_y_registra(carpetas, monkeypatch):
    classify.train(load_reviews())                                       # modelo "actual"
    viejo = (config.MODELS_DIR / "topic.joblib").stat().st_mtime_ns
    monkeypatch.setattr(config, "RETRAIN_MAX_DROP", 1.0)                 # tolerancia enorme: se acepta seguro
    _cola_corregida(TEMAS_OK)
    r = retrain.main()
    assert r["decision"] == "aceptado" and r["n_correcciones"] == len(NUEVAS)
    assert any((config.MODELS_DIR / "archive").iterdir())                # copia del modelo anterior
    assert (config.MODELS_DIR / "topic.joblib").stat().st_mtime_ns > viejo
    log = (config.REPORTS / "retrain_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(log[-1])["decision"] == "aceptado"


def test_retrain_rechaza_si_el_candidato_empeora_y_no_toca_el_modelo(carpetas, monkeypatch):
    classify.train(load_reviews())
    antes = (config.MODELS_DIR / "topic.joblib").stat().st_mtime_ns
    monkeypatch.setattr(config, "RETRAIN_MAX_DROP", -1.0)                # imposible de cumplir: exige mejorar +1.0
    _cola_corregida(TEMAS_OK)
    r = retrain.main()
    assert r["decision"] == "rechazado"
    assert (config.MODELS_DIR / "topic.joblib").stat().st_mtime_ns == antes
    assert not (config.MODELS_DIR / "archive").exists()
    assert len(feedback.load_corrections()) == len(NUEVAS)               # las correcciones NO se pierden


def test_retrain_sin_nada_nuevo_no_hace_nada(carpetas):
    assert retrain.main()["decision"] == "sin_novedades"
    assert retrain.main(force=True)["decision"] == "sin_correcciones"      # ni a la fuerza hay con qué entrenar


def test_retrain_no_repite_si_no_hay_correcciones_nuevas(carpetas, monkeypatch):
    classify.train(load_reviews())
    monkeypatch.setattr(config, "RETRAIN_MAX_DROP", 1.0)
    _cola_corregida(TEMAS_OK)
    assert retrain.main()["decision"] == "aceptado"
    copias = len(list((config.MODELS_DIR / "archive").iterdir()))
    assert retrain.main()["decision"] == "sin_novedades"                   # segunda corrida sin tocar el Excel
    assert len(list((config.MODELS_DIR / "archive").iterdir())) == copias  # no reemplazó ni archivó nada
    assert retrain.main(force=True)["decision"] == "aceptado"              # --force sí reentrena


def test_excel_guarda_con_comas_y_codificacion_antigua(carpetas):
    """El 'CSV' clásico de Excel puede usar comas y cp1252 (tildes rotas si se lee como UTF-8)."""
    d = feedback._dir(None)
    (d / feedback.QUEUE_FILE).write_text(
        "id,text,rating,topic_predicted,topic_conf,topic_correct\n"
        '9100000,"Mala atención, nunca más",1,x,0.2,Atención y personal\n', encoding="cp1252")
    assert feedback.apply_queue()["added"] == 1
    assert feedback.load_corrections().loc[0, "text"] == "Mala atención, nunca más"


def test_archivo_sin_encabezados_da_un_error_que_se_entiende(carpetas):
    (feedback._dir(None) / feedback.QUEUE_FILE).write_text("hola;mundo\n1;2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="encabezados"):
        feedback.apply_queue()
