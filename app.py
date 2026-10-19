"""
ReviewPulse-Agent: demostración en vivo (Streamlit).
"""
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import evaluate as ev
from reviewpulse import classify, config
from reviewpulse.agent import ReviewPulseAgent, build_history
from reviewpulse.classifiers import ModelClassifier, OutOfFoldClassifier
from reviewpulse.data import load_model_split, load_reviews
from reviewpulse.metrics import weekly_series
from reviewpulse.source import SimulatedSource

ROOT = config.ROOT
MODO_VIVO = "Modelo entrenado ahora con train.csv (rápido)"
MODO_OOF = "Predicciones fuera de muestra (sin fuga, como en la evaluación; ~30 s la primera vez)"

st.set_page_config(page_title="ReviewPulse-Agent", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- recursos (se calculan una vez por sesión del servidor)
@st.cache_data(show_spinner=False)
def datos() -> pd.DataFrame:
    return load_reviews()


@st.cache_resource(show_spinner="Entrenando el modelo desde cero (TF-IDF + regresión logística)...")
def clasificador(modo: str):
    t0 = time.time()
    if modo == MODO_OOF:
        c = OutOfFoldClassifier(datos(), seed=config.RANDOM_STATE)
    else:
        carpeta = Path(tempfile.mkdtemp(prefix="rp_modelo_"))
        classify.save(*classify.fit(load_model_split("train")), carpeta)      # solo train.csv: 210 reseñas
        c = ModelClassifier(carpeta)
    c.segundos = time.time() - t0
    return c


@st.cache_resource(show_spinner=False)
def casos():
    """Casos típicos y de falla esperada, con el resultado que debe dar el agente (mismos criterios de evaluate.py)."""
    df = datos()
    s = weekly_series(df)
    semana = pd.Timestamp("2023-09-04")                    # semana sin alerta real, con ventanas suficientes
    k = ev._k_min(df, semana)                              # menor pico de negativas que dispara alerta con ratings reales
    vacia = next(x for x in s.index[60:] if s.loc[x, "n"] == 0)
    base = dict(semana=semana, spike=0, p_fail=0.0, p_missing=0.0, p_blank=0.0, fail_next=0)
    return {
        "TÍPICO 1: semana normal": dict(base, tipo="Típico", esperado="Ciclo completo, sin alerta, resumen publicado.",
                                       ok=lambda r: r["completo"] and not r["alerta"]),
        f"TÍPICO 2: pico de negativas (+{3 * k} reseñas negativas)": dict(
            base, tipo="Típico", spike=3 * k, esperado="Ciclo completo y ALERTA: aviso (simulado) al gerente.",
            ok=lambda r: r["completo"] and r["alerta"]),
        "FALLA 1: la fuente de reseñas se cae y no vuelve": dict(
            base, tipo="Falla esperada", fail_next=99,
            esperado="Reintenta 3 veces y se detiene en Falla_fuente: sin caerse y sin inventar datos.",
            ok=lambda r: (not r["completo"]) and r["estado"] == "Falla_fuente"),
        "FALLA 2: semana sin reseñas": dict(
            base, tipo="Falla esperada", semana=vacia,
            esperado="Registra la semana vacía (status sin_reseñas) y NO alerta.",
            ok=lambda r: r["completo"] and r["status"] == "sin_reseñas" and not r["alerta"]),
        "FALLA 3: todas las reseñas llegan con el texto en blanco": dict(
            base, tipo="Falla esperada", p_blank=1.0, esperado="Descarta lo que no se puede clasificar, termina y NO alerta.",
            ok=lambda r: r["completo"] and not r["alerta"]),
        "FALLA 4: todas las reseñas llegan sin fecha": dict(
            base, tipo="Falla esperada", p_missing=1.0, esperado="Les asigna el inicio de la semana y sigue procesando.",
            ok=lambda r: r["completo"] and r["n"] > 0),
        "Personalizado (elige tú los parámetros)": dict(base, tipo="Libre", esperado="Depende de lo que elijas.", ok=None),
    }


# ---------------------------------------------------------------- ejecutar el agente una semana
def ejecutar_agente(p: dict, modo: str, seed: int, max_retries: int) -> dict:
    df, clasif = datos(), clasificador(modo)
    carpeta = Path(tempfile.mkdtemp(prefix="rp_corrida_"))              # cada corrida escribe en su propia carpeta
    fuente = SimulatedSource(df, seed=seed, p_fail=p["p_fail"], p_missing=p["p_missing"], p_blank=p["p_blank"],
                             fail_next=p["fail_next"], spikes={p["semana"]: p["spike"]} if p["spike"] else None)
    agente = ReviewPulseAgent(fuente, clasif, build_history(df, p["semana"], clasif), history_dir=carpeta,
                              max_retries=max_retries, run_id="app")
    t0 = time.time()
    r = agente.run_week(p["semana"])
    aviso = next(iter((carpeta / "outbox").glob("*.txt")), None)
    return {"r": r, "log": pd.read_json(carpeta / "agent_log.jsonl", lines=True), "seg": time.time() - t0,
            "aviso": aviso.read_text(encoding="utf-8") if aviso else None}


def correr_streaming(cmd: list, caja) -> tuple:
    """Ejecuta un comando y va mostrando las últimas líneas de su salida."""
    lineas = []
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    for linea in proc.stdout:
        if "[SIMULADO" not in linea:                       # un aviso simulado por alerta: ruido en pantalla
            lineas.append(linea.rstrip())
            caja.code("\n".join(lineas[-12:]), language="text")
    return proc.wait(), lineas


# ================================================================= interfaz
st.title("📈 ReviewPulse-Agent")
st.caption("Agente que revisa las reseñas de un hotel, compara el tono reciente con el periodo anterior y avisa al gerente "
           "si empeora. Reseñas: Hotel Dorado Plaza (300, 2020-2026). La fuente de reseñas está simulada.")
t_inicio, t_agente, t_pruebas = st.tabs(["1) Cómo usar", "2) Agente en vivo", "3) Pruebas y tabla de resultados"])

# ---------------------------------------------------------------- 1) instrucciones
with t_inicio:
    st.markdown("""
**Qué es:** un agente basado en modelo con ciclo *percepción → decisión → acción*. Cada semana consulta la fuente de reseñas,
las limpia, clasifica polaridad (1 a 5) y tema, compara las últimas 26 semanas con las 26 anteriores y, si el porcentaje de
negativas sube de forma significativa, avisa al gerente. **No usa LLM**: el resumen se arma con plantillas.

**Cómo probarlo (5 minutos):**
1. Ve a **2) Agente en vivo**. La primera vez la app entrena el modelo desde cero (unos segundos).
2. Elige un caso de la lista y pulsa **Ejecutar el agente**. Compara *lo esperado* con *lo obtenido* y revisa el ciclo del agente.
3. Ve a **3) Pruebas y tabla de resultados** y pulsa **Ejecutar pruebas unitarias** y luego **Entrenar y evaluar**.

**Casos de prueba sugeridos:**
""")
    cs = casos()
    st.table(pd.DataFrame([{"Caso": n, "Tipo": c["tipo"], "Resultado esperado": c["esperado"]} for n, c in cs.items()]))
    st.markdown("""
**Medidas de desempeño que evalúa la pestaña 3** (umbrales en `thresholds.json`, fijados antes de ejecutar):
polaridad (M1, M1R, M1D), tema principal (M2, M2R, M2D), detección de incrementos de negativas y falsos positivos
(M3D, M3F, M3RD, M3RF) y robustez ante fallas (R1).

**Nota:** esta app de Streamlit Community Cloud se duerme tras 12 horas sin visitas; si ves el botón
*"Yes, get this app back up!"*, púlsalo y espera un minuto.
""")

# ---------------------------------------------------------------- 2) agente en vivo
with t_agente:
    cs = casos()
    c1, c2 = st.columns([3, 2])
    nombre = c1.selectbox("Caso", list(cs))
    modo = c2.radio("Clasificador", [MODO_VIVO, MODO_OOF], index=0)
    caso = dict(cs[nombre])
    if caso["tipo"] == "Libre":
        a, b, c, d, e = st.columns(5)
        caso["semana"] = pd.Timestamp(a.date_input("Semana", pd.Timestamp("2023-09-04")))
        caso["spike"] = b.number_input("Pico de negativas", 0, 300, 0)
        caso["p_fail"] = c.slider("Prob. de caída de la fuente", 0.0, 1.0, 0.2, 0.05)
        caso["p_missing"] = d.slider("Reseñas sin fecha", 0.0, 1.0, 0.1, 0.05)
        caso["p_blank"] = e.slider("Reseñas sin texto", 0.0, 1.0, 0.0, 0.05)
    d1, d2, d3 = st.columns(3)
    seed = d1.number_input("Semilla (reproducible)", 0, 9999, 0)
    max_retries = d2.number_input("Reintentos máximos", 0, 10, 3)
    if d3.button("Reentrenar desde cero (limpiar caché)"):
        st.cache_resource.clear()
        st.rerun()
    caso["semana"] = pd.Timestamp(caso["semana"])
    caso["semana"] -= pd.Timedelta(days=caso["semana"].weekday())          # siempre el lunes
    st.info(f"**Esperado:** {caso['esperado']}  \n**Semana:** {caso['semana'].date()} | **Pico inyectado:** {caso['spike']} "
            f"| **Caídas forzadas:** {caso['fail_next']} | **Prob. de caída:** {caso['p_fail']} | **Sin fecha:** {caso['p_missing']} "
            f"| **Sin texto:** {caso['p_blank']}")

    if st.button("▶ Ejecutar el agente", type="primary"):
        res = ejecutar_agente(caso, modo, int(seed), int(max_retries))
        r, log = res["r"], res["log"]
        st.success(f"Modelo: {getattr(clasificador(modo), 'segundos', 0):.1f} s de entrenamiento | "
                   f"el agente tardó {res['seg']:.1f} s")
        m = st.columns(6)
        m[0].metric("Estado final", r["estado"])
        m[1].metric("Ciclo completo", "sí" if r["completo"] else "no")
        m[2].metric("Alerta", "SÍ" if r["alerta"] else "no")
        m[3].metric("Comparación", str(r["status"]))
        m[4].metric("Reseñas recibidas", "—" if r["n"] is None else r["n"])
        m[5].metric("Reintentos", r["reintentos"])
        if caso["ok"]:
            (st.success if caso["ok"](r) else st.error)(
                ("✅ El agente se comportó como se esperaba. " if caso["ok"](r) else "❌ Resultado distinto al esperado. ") + caso["esperado"])
        if res["aviso"]:
            st.warning("📨 Aviso al gerente (SIMULADO: no se envió nada, solo se guarda en `outbox/`):\n\n" + res["aviso"])
        if r.get("reporte"):
            with st.expander("Resumen semanal generado (por tema, con acciones sugeridas)", expanded=True):
                st.markdown(r["reporte"]["markdown"])
        with st.expander(f"Ciclo del agente: {len(log)} eventos (percepción, decisión, acción)"):
            cols = [c for c in ("tipo", "estado", "accion", "desde", "hacia", "razon", "detalle") if c in log]
            st.dataframe(log[cols].astype(str), width="stretch", hide_index=True)

# ---------------------------------------------------------------- 3) pruebas y tabla
with t_pruebas:
    st.subheader("Pruebas unitarias (pytest)")
    if st.button("▶ Ejecutar pruebas unitarias"):
        with st.spinner("Ejecutando pytest..."):
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        (st.success if p.returncode == 0 else st.error)(
            "Todas las pruebas pasaron." if p.returncode == 0 else f"Hubo fallas (código {p.returncode}).")
        st.code("\n".join(p.stdout.strip().splitlines()[-15:]), language="text")

    st.subheader("Evaluación contra las medidas de desempeño")
    s1, s2 = st.columns([2, 3])
    n_sem = s1.slider("Semillas (repeticiones)", 1, 5, 1)
    s2.caption("1 semilla ≈ 1 minuto (prueba rápida). **5 semillas** = la corrida válida para la entrega (≈ 6 a 10 minutos). "
               "El resultado queda registrado en `logs/fase3_registro.txt`.")
    if st.button("▶ Entrenar y evaluar"):
        with st.status("Ejecutando la evaluación completa...", expanded=True) as estado:
            caja = st.empty()
            st.write("1/2 Entrenando el modelo guardado (`train.py`, solo con train.csv)")
            cod, _ = correr_streaming([sys.executable, "-u", "train.py"], caja)
            if cod == 0:
                st.write(f"2/2 Evaluando (`registro_fase3.py --seeds {n_sem}`)")
                cod, _ = correr_streaming([sys.executable, "-u", "registro_fase3.py", "--seeds", str(n_sem),
                                           "--nota", "ejecutada desde la app"], caja)
            estado.update(label="Evaluación terminada" if cod == 0 else f"Falló (código {cod})",
                          state="complete" if cod == 0 else "error")
        if n_sem < 5:
            st.warning("Con menos de 5 semillas la corrida NO es válida para la entrega (mínimo de repeticiones).")

    tabla_csv = ROOT / "logs" / "fase3" / "tabla_resultados.csv"
    if tabla_csv.exists():
        st.markdown(f"**Tabla de resultados** (última evaluación: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tabla_csv.stat().st_mtime))})")
        st.dataframe(pd.read_csv(tabla_csv, encoding="utf-8-sig"), width="stretch", hide_index=True)
    reg = ROOT / "logs" / "fase3_registro.txt"
    if reg.exists():
        bloques = re.split(r"^(?==+\nCORRIDA #)", reg.read_text(encoding="utf-8"), flags=re.M)
        bloques = [b for b in bloques if "CORRIDA #" in b]
        with st.expander(f"Registro de corridas: {len(bloques)} registradas (se muestra la última)"):
            st.code(bloques[-1], language="text")
    elif not tabla_csv.exists():
        st.info("Aún no hay resultados en este servidor: pulsa **Entrenar y evaluar**.")
