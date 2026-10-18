from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TIMEZONE = "America/Bogota"    # las fechas del sitio vienen en UTC; el hotel está en Colombia (UTC-5)

# --- Qué es una review "negativa" ---
NEGATIVE_MAX_RATING = 2        # rating 1 o 2 = negativa

# --- Comparación de ventanas ---
WINDOW_WEEKS = 26              # tamaño de cada ventana en semanas (26 = ~6 meses): actual y anterior
MIN_REVIEWS = 8                # mínimo de reviews en CADA ventana para poder concluir algo

# --- Cuándo se dispara la alerta ---
NEG_RISE = 0.15                # el % de negativas debe subir al menos 15 puntos (0.15)
ALERT_P_VALUE = 0.10           # y la subida no debe ser "casualidad" (prueba de Fisher)

# --- Clasificador (classify.py) ---
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
TOPIC_REVIEW_THRESHOLD = 0.30   # si la confianza del tema es menor, se marca para revisión humana

# --- Reporte, historial y notificación (report.py, store.py, notify.py) ---
HOTEL_NAME = "Hotel Dorado Plaza"
HISTORY_DIR = ROOT / "history"   # aquí se guarda el historial, los resúmenes y las acciones
TOP_TOPICS_IN_REPORT = 3         # cuántos temas prioritarios aparecen con acciones sugeridas
MIN_TOPIC_REVIEWS = 5            # un tema con menos reseñas que esto se marca "poca muestra"