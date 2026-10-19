# ReviewPulse-Agent

Agente de análisis de sentimiento en reseñas de hotel. Cada semana lee las reseñas nuevas de un hotel piloto
(en español), asigna a cada una una **polaridad (1 a 5)** y un **tema principal**, compara la semana con las
anteriores, **publica un resumen por tema** y **avisa al gerente** si detecta un aumento inusual de reseñas negativas.

Proyecto del curso de Inteligencia Artificial 
Ingeniería de Sistemas y Computación
Escuela de Transformación Digital
Universidad Tecnológica de Bolívar
Docente: Edwin Alexander Puertas del Castillo.

Autores: 
- Alejandro Villarreal Imitola (T00077164)
- Ana Sofia Meza Herrera (T00077991)

**Tipo de agente (AIMA):** basado en modelo (mantiene estado interno: las reseñas ya clasificadas y el historial semanal).
**Entorno:** simulado y estocástico (la fuente de reseñas puede caerse, venir vacía o con metadatos incompletos).
**Sin LLM:** la clasificación usa scikit-learn y el resumen se arma con plantillas.

---

## 1. Instalación

Requiere Python 3.10 o superior (probado con 3.12). Comprobar version con `python --version`

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Uso rápido (todos los comandos, desde la raíz del repositorio)

| Qué quieres hacer | Comando |
|---|---|
| Entrenar el modelo y ver sus métricas | `python train.py` |
| Correr el agente semana a semana contra la fuente simulada | `python run_agent.py --start 2025-03-17 --weeks 3 --seed 3 --p-fail 0.4 --history-dir logs/demo` |
| Correr el agente semana a semana contra la fuente simulada sin que el modelo vea las reseñas que clasifica | agregar `--oof` al comando anterior |
| Correr el agente sobre UNA semana con los datos etiquetados | `python run_week.py 2025-03-17` |
| Backtest de alertas sobre todo el histórico | `python main.py` |
| Pruebas automáticas (35) | `pytest` |
| Pruebas de la fase 3 contra las medidas de desempeño | `python evaluate.py` |
| Clasificar reseñas nuevas y generar la cola de revisión humana | `python make_queue.py data/inbox/nuevas.csv` |
| Reentrenar con las correcciones del revisor | `python retrain.py` |

**Primera ejecución completa de ejemplo:**

```powershell
python train.py
python run_agent.py --start 2025-03-17 --weeks 3 --seed 3 --p-fail 0.4 --history-dir logs/demo
dir logs\demo
pytest
```

Opciones de `run_agent.py`:

| Opción | Significado |
|---|---|
| `--start` | lunes de la primera semana a simular |
| `--weeks` | cuántas semanas procesar |
| `--seed` | semilla del azar del simulador (reproducible) |
| `--p-fail` | probabilidad de que la fuente falle en cada consulta |
| `--p-missing` | probabilidad de que una reseña llegue sin fecha |
| `--p-blank` | probabilidad de que una reseña llegue con el texto en blanco |
| `--max-retries` | reintentos de la fuente antes de abandonar (por defecto 3) |
| `--oof` | clasifica con predicciones fuera de muestra (ver sección 6) |
| `--history-dir` | carpeta donde se escribe TODO lo de esa corrida |

## 3. Estructura del repositorio

```
ReviewPulse-Agent/
├── run_agent.py        agente completo contra la fuente simulada (ciclo percepción-decisión-acción)
├── run_week.py         una semana con datos etiquetados (reporte -> guardar -> publicar -> avisar)
├── main.py             backtest de alertas sobre el histórico
├── train.py            entrena los modelos
├── evaluate.py         fase 3: pruebas contra las medidas de desempeño (lee thresholds.json)
├── thresholds.json     fórmulas y umbrales, fijados ANTES de ejecutar evaluate.py
├── make_queue.py       clasifica reseñas nuevas y manda a revisión las dudosas
├── retrain.py          reentrena con las correcciones y solo acepta el modelo si no empeora
├── registro_fase3.py   registro reproducible de las pruebas oficiales
├── reviewpulse/
│   ├── agent.py        el agente: estados, acciones, percepción / decisión / acción y log
│   ├── source.py       fuente simulada de reseñas (el entorno)
│   ├── classifiers.py  ModelClassifier (models/) y OutOfFoldClassifier (sin fuga de datos)
│   ├── classify.py     modelos de polaridad y tema (TF-IDF + regresión logística)
│   ├── analysis.py     comparación de ventanas y regla de alerta
│   ├── metrics.py      series semanales
│   ├── report.py       resumen semanal por tema
│   ├── store.py        historial y bitácora de acciones
│   ├── notify.py       publicar el resumen y avisar al gerente (simulado por defecto)
│   ├── feedback.py     cola de revisión humana y correcciones
│   ├── manifest.py     registro de con qué datos se entrenó models/
│   ├── data.py         carga de datos
│   └── config.py       parámetros (umbrales, ventanas, rutas)
├── tests/              35 pruebas automáticas
├── notebooks/          preparación de datos (02_prepare_data.ipynb genera train.csv y test.csv)
├── data/               raw/ (formato original del sitio) y processed/ (reviews_eda, train, test, serie semanal)
├── models/             modelos entrenados (no se sube a git)
├── logs/               evidencia de ejecuciones y de la fase 3
```

## 4. Cómo funciona el agente

Cada semana el agente recorre estos estados (formalización AIMA del documento de diseño). 
Una sola acción está disponible en cada estado; la que ramifica es el resultado de ejecutarla.

| Estado | Acción | Siguiente estado |
|---|---|---|
| Inactivo | consultar_fuente | Lote_recibido, Sin_reseñas o Falla_fuente |
| Lote_recibido | preprocesar | Lote_preprocesado (o Sin_reseñas si no queda ningún texto) |
| Lote_preprocesado | clasificar | Reseñas_clasificadas |
| Reseñas_clasificadas | agregar_por_tema | Resultados_agregados |
| Resultados_agregados | comparar_semanas | Sin_anomalía o Anomalía_detectada |
| Anomalía_detectada | notificar_gerente | Alerta_enviada |
| Alerta_enviada / Sin_anomalía | publicar_resumen | **Resumen_completado** (meta) |
| Sin_reseñas | registrar_semana_vacía | Resumen_completado |
| Falla_fuente | reintentar | Inactivo (máximo `--max-retries` veces; luego abandona y lo registra) |

En el código (`reviewpulse/agent.py`) el ciclo está separado en tres métodos:

- `perceive()`: lo único que el agente ve del entorno (las reseñas de la fuente, **sin rating ni tema**).
- `decide()`: elige la acción permitida en el estado actual y deja escrita la razón en el log.
- `act()`: ejecuta la acción, calcula el nuevo estado y lo registra.

**Estado interno (por qué es basado en modelo):** `self.reviews` guarda todas las reseñas ya clasificadas (memoria de donde salen las ventanas de comparación) y `self.batch` el lote de la semana en curso.

**Regla de alerta:** se compara la ventana actual (26 semanas) con la anterior. Hay alerta si el porcentaje de negativas sube al menos 15 puntos y la subida no es casualidad (prueba de Fisher, p ≤ 0,10), con un mínimo de 8 reseñas en cada ventana. Los parámetros están en `reviewpulse/config.py`.

## 5. Qué hace el código y qué hacen las herramientas externas

| Parte | Quién la hace |
|---|---|
| Ciclo del agente, estados, decisiones, reintentos, regla de alerta, resumen, historial, log | Código propio (Python) |
| Polaridad (1 a 5) y tema de cada reseña | scikit-learn: TF-IDF + regresión logística (`classify.py`) |
| Comparación estadística de ventanas | scipy (prueba exacta de Fisher) |
| Datos y series | pandas / numpy |
| Redacción del resumen | Plantillas en `report.py` (no se usa ningún LLM) |
| Publicar el resumen y avisar al gerente | `notify.py`: simulado por defecto; envío real con las variables de entorno de la sección 8 |

## 6. Datos y por qué existen dos clasificadores

El conjunto de datos tiene 300 reseñas en español de un hotel (`data/processed/reviews_eda.csv`), con rating (1 a 5) y temas etiquetados. 
`train.csv` (210) y `test.csv` (90) salen de una división estratificada con semilla fija (`notebooks/02_prepare_data.ipynb`).

Como solo hay un conjunto de datos, el agente simulado procesa reseñas con las que el modelo pudo haber entrenado.
Para no inflar los resultados hay dos clasificadores con la misma interfaz (`reviewpulse/classifiers.py`):

- **`ModelClassifier`** usa los modelos guardados en `models/` (el modelo real). Sin `--oof`, `run_agent.py` usa este.
- **`OutOfFoldClassifier`** usa validación cruzada: cada reseña la predice un modelo que no la vio al entrenar.
  Es el que usan `evaluate.py`, las pruebas automáticas y `run_agent.py --oof`.

## 7. Logs y evidencia

`run_agent.py` escribe en la carpeta de `--history-dir` (por defecto `history/`):

| Archivo | Contenido |
|---|---|
| `agent_log.jsonl` | cada percepción, decisión (con su razón), acción, error y abandono, con marca de tiempo UTC |
| `actions_log.jsonl` | publicaciones y avisos al gerente |
| `weekly_history.csv`, `topic_history.csv` | historial por semana y por tema |
| `summaries/<semana>.md` | el resumen semanal publicado |
| `outbox/<semana>_aviso.txt` | el aviso al gerente (simulado) |

Usa una carpeta por corrida (`--history-dir logs/<nombre>`) para que la evidencia quede separada. Los logs
de las pruebas automáticas se guardan con `pytest --basetemp=logs/pytest`.

## 8. Conexión real (opcional)

Por defecto, publicar y avisar son **simulaciones** (escriben archivos, no envían nada). 
Para activar un canal real se definen variables de entorno:

| Variable | Para qué |
|---|---|
| `REVIEWPULSE_WEBHOOK` | webhook (Slack, Teams...) que recibe el aviso |
| `REVIEWPULSE_SMTP_HOST`, `_PORT`, `_USER`, `_PASSWORD`, `REVIEWPULSE_ALERT_FROM`, `REVIEWPULSE_ALERT_TO` | aviso por correo |
| `REVIEWPULSE_ENDPOINT`, `REVIEWPULSE_TOKEN` | sistema de gestión de reputación (recibe el resumen en JSON) |


## 9. Pruebas

- `pytest` ejecuta las 35 pruebas (análisis, reporte/historial/avisos, aprendizaje con revisión humana y agente de
  punta a punta, incluidos los casos de falla: fuente caída, semana sin reseñas, metadatos incompletos, textos en blanco).
- `python evaluate.py` ejecuta las pruebas de la fase 3 contra las medidas de desempeño del diseño y deja la evidencia en `logs/fase3/` (incluida `tabla_resultados.md`).

**Orden correcto para la evaluación** (para que el examen sobre `models/` sea válido):

```powershell
python train.py          # sin --all
python evaluate.py       # no ejecutes retrain.py ni train.py --all entre los dos
```

Los umbrales están en `thresholds.json` y se fijaron antes de ejecutar. No los modifiques después de ver los resultados.
Si el modelo guardado ya vio `test.csv` (`train.py --all` o un reentrenamiento aceptado), `evaluate.py` marca esas
filas como "No evaluable" en lugar de reportar un resultado contaminado.

## 10. Ciclo de aprendizaje con revisión humana

1. `python make_queue.py <archivo.csv>` clasifica reseñas nuevas y deja en `data/feedback/review_queue.csv` las que el modelo duda.
2. Una persona escribe el tema correcto en la columna `topic_correct` (cierra el Excel antes de seguir).
3. `python retrain.py` aprende de las correcciones y solo reemplaza el modelo si no empeora en `test.csv`; guarda una copia del modelo anterior en `models/archive/`.

El agente no vigila el archivo: se entera cuando se ejecuta `retrain.py`. 
Los archivos de `data/feedback/` contienen textos de huéspedes: no los subas a repositorios públicos.

## 11. Limitaciones

- Un solo hotel y 300 reseñas en español: pocas para reentrenar y para medir con precisión; el modelo y el agente comparten dataset.
- No se detecta el idioma ni se filtran reseñas que no estén en español.
- El simulador reproduce reseñas históricas semana a semana; no hay conexión a una plataforma real.
- Sarcasmo, ironía y expresiones regionales no se tratan de forma específica.
- Las metas de polaridad (1 a 5) y de tema estricto del diseño no se cumplen con el modelo actual; el detalle y su análisis están en `logs/fase3/tabla_resultados.md` y en el documento de entrega.

## 12. Reproducibilidad

- Semillas: división de datos (`random_state=42`), validación cruzada y simulador (`--seed`), todas reproducibles.
- Para fijar las versiones exactas con las que obtuviste tus resultados: `pip freeze > requirements-lock.txt`.