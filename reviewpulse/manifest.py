"""Registro de con qué datos se entrenaron los modelos guardados en models/.

train.py escribe models/manifest.json (con qué reseñas se entrenó y la huella SHA-256 de cada archivo).
evaluate.py lo consulta para saber si models/ sirve como examen limpio: solo sirve si se entrenó SIN las
reseñas de test.csv y nadie lo ha reemplazado desde entonces (por ejemplo, retrain.py al aceptar correcciones).
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from reviewpulse import classify, config

NOMBRE = "manifest.json"


def _dir(models_dir):
    return Path(models_dir) if models_dir else config.MODELS_DIR


def _huellas(models_dir) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in classify._paths(models_dir)}


def write(entrenado_con: str, n: int, incluye_test: bool, models_dir=None) -> Path:
    ruta = _dir(models_dir) / NOMBRE
    ruta.write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrenado_con": entrenado_con, "n_entrenamiento": int(n), "incluye_test": bool(incluye_test),
        "sha256": _huellas(models_dir)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def check(models_dir=None) -> tuple:
    """(valido, motivo, manifest). valido=True solo si los modelos actuales no vieron test.csv."""
    ruta = _dir(models_dir) / NOMBRE
    if not ruta.exists():
        return False, "no hay models/manifest.json: los modelos no se entrenaron con train.py (¿retrain.py o a mano?). Ejecuta python train.py", None
    m = json.loads(ruta.read_text(encoding="utf-8"))
    try:
        if _huellas(models_dir) != m["sha256"]:
            return False, "los modelos de models/ cambiaron después de train.py (¿retrain.py aceptó correcciones?). Ejecuta python train.py", m
    except FileNotFoundError:
        return False, "faltan archivos en models/. Ejecuta python train.py", m
    if m["incluye_test"]:
        return False, "los modelos se entrenaron con las 300 reseñas (train.py --all), incluido test.csv: el examen quedaría contaminado. Ejecuta python train.py sin --all", m
    return True, "modelos entrenados solo con train.csv y sin cambios posteriores", m
