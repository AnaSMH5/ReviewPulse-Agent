"""Fase 3: pruebas del agente contra las medidas de desempeño de la sección 7 del diseño.

    python evaluate.py                # todo (tarda unos minutos)
    python evaluate.py --seeds 2      # prueba rápida (NO usar para la entrega)

Lee los umbrales de thresholds.json (fijados antes de ejecutar) y deja la evidencia en logs/fase3/:
    m12_casos.csv        cada caso de las medidas 1 y 2, con su semilla
    m3_casos.csv         cada escenario de la medida 3 y de robustez, con su semilla
    m3_retro_semanas.csv semana a semana de la prueba retrospectiva
    m3/<semilla>_<caso>/ el log completo del agente de cada escenario (agent_log.jsonl, resúmenes, avisos)
    tabla_resultados.md / .csv   la tabla de resultados que pide la entrega
Las predicciones son FUERA DE MUESTRA (validación cruzada): el modelo nunca clasifica una reseña con la que entrenó.
"""
import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from reviewpulse import classify, config, manifest
from reviewpulse.agent import ReviewPulseAgent, build_history
from reviewpulse.analysis import compare
from reviewpulse.classifiers import OutOfFoldClassifier
from reviewpulse.data import load_model_split, load_reviews
from reviewpulse.metrics import weekly_series
from reviewpulse.source import SimulatedSource

OUT = config.ROOT / "logs" / "fase3"
# entorno estocástico de los escenarios del agente
P_FAIL, P_MISSING = 0.2, 0.1
OPS = {">=": lambda v, u: v >= u, "<": lambda v, u: v < u}


# ======================================================================= medidas 1 y 2
def medidas_1_2(df: pd.DataFrame, seeds: list) -> pd.DataFrame:
    filas = []

    def add(caso, tipo, seed, n, metrica, valor):
        filas.append({"caso": caso, "tipo": tipo, "semilla": seed,
                     "n": int(n), "metrica": metrica, "valor": float(valor)})

    palabras = df["text"].str.split().str.len()
    corto, largo = palabras <= palabras.quantile(
        0.25), palabras >= palabras.quantile(0.75)
    raros = df["main_topic"].map(df["main_topic"].value_counts()) < 15

    for s in seeds:
        p = OutOfFoldClassifier(df, seed=s).predict(df)
        y, yp = df["rating"].astype(int).to_numpy(
        ), p["polarity"].astype(int).to_numpy()
        def tr3(v): return np.where(v <= 2, 0, np.where(v == 3, 1, 2))
        pol_ok, pol3_ok = y == yp, tr3(y) == tr3(yp)
        top_ok = (df["main_topic"] == p["topic"]).to_numpy()
        perm_ok = np.array([t in l for t, l in zip(p["topic"], df["topics"])])
        add("C01_global_300", "normal", s, len(
            df), "polaridad_1a5", pol_ok.mean())
        add("C01_global_300", "normal", s, len(df),
            "polaridad_3clases", pol3_ok.mean())
        add("C01_global_300", "normal", s, len(
            df), "tema_estricta", top_ok.mean())
        add("C01_global_300", "normal", s, len(
            df), "tema_permisiva", perm_ok.mean())
        # clases, incluidas las minoritarias
        for r in range(1, 6):
            m = y == r
            add(f"C0{2 + r}_rating_{r}", "límite", s,
                m.sum(), "polaridad_1a5", pol_ok[m].mean())
        add("C08_texto_corto", "límite", s, corto.sum(),
            "polaridad_1a5", pol_ok[corto.to_numpy()].mean())
        add("C09_texto_largo", "límite", s, largo.sum(),
            "polaridad_1a5", pol_ok[largo.to_numpy()].mean())
        add("C10_temas_minoritarios", "límite", s, raros.sum(),
            "tema_estricta", top_ok[raros.to_numpy()].mean())

    tr, te = load_model_split("train"), load_model_split(
        "test")                     # examen fijo (un solo modelo, sin azar)
    h = classify.evaluate_holdout(tr, te)
    add("C11_holdout_90", "normal", "fijo",
        h["polaridad"]["n_test"], "polaridad_1a5", h["polaridad"]["accuracy"])
    add("C11_holdout_90", "normal", "fijo",
        h["tema"]["n_test"], "tema_estricta", h["tema"]["accuracy_estricta"])

    # casos de falla esperada: el modelo NO debe inventar una etiqueta
    with tempfile.TemporaryDirectory() as tmp:
        classify.save(*classify.fit(df), Path(tmp))
        vacios = classify.predict(pd.DataFrame(
            {"text": ["", "   ", None, np.nan]}), Path(tmp))
        ilegibles = classify.predict(pd.DataFrame(
            {"text": ["asdf qwer zxcv", "zzz 123 ???", "lorem ipsum dolor sit amet", "xx"]}), Path(tmp))
    add("C12_texto_vacio", "falla_esperada", "fijo", len(vacios), "manejado_correcto",
        (vacios["polarity"].isna() & vacios["needs_review"]).mean())
    add("C13_texto_ilegible", "falla_esperada", "fijo", len(ilegibles),
        "manejado_correcto", ilegibles["needs_review"].mean())
    return pd.DataFrame(filas)


def examen_modelo_guardado() -> dict:
    """Evalúa el modelo REAL de models/ sobre test.csv, solo si ese modelo nunca vio test.csv."""
    valido, motivo, m = manifest.check()
    res = {"valido": valido, "motivo": motivo, "manifest": m}
    if valido:
        te = load_model_split("test")
        # carga models/ (el modelo desplegado)
        p = classify.predict(te)
        y, yp = te["rating"].astype(int).to_numpy(
        ), p["polarity"].astype(int).to_numpy()
        res["n"] = int(len(te))
        res["polaridad_1a5"] = float((y == yp).mean())
        res["tema_estricta"] = float((te["main_topic"] == p["topic"]).mean())
    return res


# ======================================================================= medida 3
def _alerta_real(df_true: pd.DataFrame, semana, extra: int = 0) -> dict:
    """Verdad: la misma regla de alerta, pero calculada con los ratings reales (+ negativas inyectadas)."""
    d = df_true[["week_start", "rating"]]
    if extra:
        d = pd.concat([d, pd.DataFrame(
            {"week_start": [semana] * extra, "rating": [2] * extra})], ignore_index=True)
    return compare(weekly_series(d), semana)


def _k_min(df_true, semana, tope=150):
    for k in range(1, tope + 1):
        if _alerta_real(df_true, semana, k)["alert"]:
            return k
    return None


def _semanas_control(df, cuantas=10):
    serie = weekly_series(df)
    cand = [s for s in serie.index[26:] if serie.loc[s, "n"] > 0
            and (r := _alerta_real(df, s))["status"] != "datos_insuficientes" and not r["alert"]]
    elegidas = []
    # repartidas en el tiempo
    for i in np.linspace(0, len(cand) - 1, cuantas + 6).round().astype(int):
        k = _k_min(df, cand[i])
        if k is not None and len(elegidas) < cuantas:
            elegidas.append((cand[i], k))
    return elegidas


def _correr(df, oof, semana, seed, carpeta: Path, spike=0, max_retries=3, **fuente):
    if carpeta.exists():
        shutil.rmtree(carpeta)
    src = SimulatedSource(df, seed=seed, spikes={
                          semana: spike} if spike else None, **fuente)
    ag = ReviewPulseAgent(src, oof, build_history(df, semana, oof), history_dir=carpeta, max_retries=max_retries,
                          run_id=carpeta.name)
    return ag.run_week(semana)


def medida_3(df: pd.DataFrame, seeds: list) -> pd.DataFrame:
    semanas = _semanas_control(df)
    vacia = next(s for s in weekly_series(
        df).index[60:] if weekly_series(df).loc[s, "n"] == 0)
    filas = []
    for s in seeds:
        oof = OutOfFoldClassifier(df, seed=s)
        base = OUT / "m3"
        for w, k in semanas:
            for caso, spike in (("control", 0), ("moderado", math.ceil(1.5 * k)), ("grande", 3 * k)):
                r = _correr(df, oof, w, s, base / f"s{s}_{caso}_{w.date()}",
                            spike=spike, p_fail=P_FAIL, p_missing=P_MISSING)
                filas.append({"grupo": "pico" if spike else "control", "caso": f"{caso}_{w.date()}", "semilla": s,
                              "spike": spike, "esperado_alerta": bool(spike), "alerta": r["alerta"],
                              "completo": r["completo"], "ok": r["alerta"] == bool(spike)})
        w0 = semanas[0][0]
        fallas = {                                                       # casos de falla esperada del agente
            "fuente_caida_permanente": (dict(week=w0, spike=0, fail_next=99), lambda r: not r["completo"] and r["estado"] == "Falla_fuente"),
            "semana_sin_resenas": (dict(week=vacia, spike=0), lambda r: r["completo"] and r["status"] == "sin_reseñas" and not r["alerta"]),
            "textos_en_blanco": (dict(week=w0, spike=0, p_blank=1.0), lambda r: r["completo"] and not r["alerta"]),
            "fechas_faltantes_100pct": (dict(week=w0, spike=0, p_missing=1.0), lambda r: r["completo"] and r["n"] > 0),
        }
        for nombre, (kw, criterio) in fallas.items():
            w = kw.pop("week")
            r = _correr(df, oof, w, s, base / f"s{s}_falla_{nombre}", **kw)
            filas.append({"grupo": "falla_esperada", "caso": nombre, "semilla": s, "spike": 0, "esperado_alerta": False,
                          "alerta": r["alerta"], "completo": r["completo"], "ok": bool(criterio(r))})
    return pd.DataFrame(filas)


def retrospectiva(df: pd.DataFrame) -> pd.DataFrame:
    """Recorre el histórico semana a semana con el agente y compara sus alertas con las calculadas con los ratings reales."""
    serie = weekly_series(df)
    inicio = serie.index[52]
    oof = OutOfFoldClassifier(df, seed=config.RANDOM_STATE)
    carpeta = OUT / "m3" / "retro"
    if carpeta.exists():
        shutil.rmtree(carpeta)
    ag = ReviewPulseAgent(SimulatedSource(df, seed=0), oof, build_history(
        df, inicio, oof), history_dir=carpeta, run_id="retro")
    filas = []
    for w in serie.index[52:]:
        r = ag.run_week(w)
        real = _alerta_real(df, w)
        filas.append({"semana": w.date(), "n": int(serie.loc[w, "n"]), "status_real": real["status"], "alerta_real": real["alert"],
                      "alerta_agente": r["alerta"], "status_agente": r["status"]})
    return pd.DataFrame(filas)


# ======================================================================= tabla
def _ms(serie):
    v = np.array(serie, dtype=float)
    return float(v.mean()), float(v.std(ddof=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None,
                    help="usar solo las primeras N semillas (prueba rápida)")
    a = ap.parse_args()
    cfg = json.loads(
        (config.ROOT / "thresholds.json").read_text(encoding="utf-8"))
    seeds = cfg["semillas"][: a.seeds] if a.seeds else cfg["semillas"]
    M = cfg["medidas"]
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_reviews()

    print("Medidas 1 y 2 ...")
    m12 = medidas_1_2(df, seeds)
    guardado = examen_modelo_guardado()
    if guardado["valido"]:
        for met in ("polaridad_1a5", "tema_estricta"):
            m12.loc[len(m12)] = {"caso": "C14_modelo_guardado_test90", "tipo": "normal", "semilla": "fijo",
                                 "n": guardado["n"], "metrica": met, "valor": guardado[met]}
    (OUT / "modelo_guardado.json").write_text(json.dumps(guardado,
                                                         ensure_ascii=False, indent=2), encoding="utf-8")
    m12.to_csv(OUT / "m12_casos.csv", index=False, encoding="utf-8-sig")
    print("Medida 3 (escenarios del agente) ...")
    m3 = medida_3(df, seeds)
    m3.to_csv(OUT / "m3_casos.csv", index=False, encoding="utf-8-sig")
    print("Medida 3 (prueba retrospectiva) ...")
    retro = retrospectiva(df)
    retro.to_csv(OUT / "m3_retro_semanas.csv",
                 index=False, encoding="utf-8-sig")

    glob = m12[m12["caso"] == "C01_global_300"]

    def g(metrica):
        return _ms(glob[glob["metrica"] == metrica]["valor"])

    def por_semilla(sub, col): return [
        sub[sub["semilla"] == s][col].mean() for s in seeds]
    pico, ctrl, falla = m3[m3["grupo"] == "pico"], m3[m3["grupo"]
                                                      == "control"], m3[m3["grupo"] == "falla_esperada"]
    # el agente no alerta en semanas vacías: no se evalúan
    ev = retro[(retro["n"] > 0) & (
        retro["status_real"] != "datos_insuficientes")]
    pos, neg = ev[ev["alerta_real"]], ev[~ev["alerta_real"]]
    f_m12 = m12[m12["tipo"] == "falla_esperada"]
    rob = list(falla["ok"].astype(float)) + \
        list((f_m12["valor"] == 1.0).astype(float))

    n_cases12 = m12["caso"].nunique()
    filas = [
        ("M1", *g("polaridad_1a5"),
         f"{n_cases12} casos x {len(seeds)} semillas", "m12_casos.csv"),
        ("M1R", *g("polaridad_3clases"),
         f"{n_cases12} casos x {len(seeds)} semillas", "m12_casos.csv"),
        ("M2", *g("tema_estricta"),
         f"{n_cases12} casos x {len(seeds)} semillas", "m12_casos.csv"),
        ("M2R", *g("tema_permisiva"),
         f"{n_cases12} casos x {len(seeds)} semillas", "m12_casos.csv"),
        ("M1D", guardado.get("polaridad_1a5"), 0.0, f"{guardado['n']} reseñas de test.csv" if guardado["valido"]
         else f"0 (no evaluable: {guardado['motivo']})", "m12_casos.csv; modelo_guardado.json"),
        ("M2D", guardado.get("tema_estricta"), 0.0, f"{guardado['n']} reseñas de test.csv" if guardado["valido"]
         else f"0 (no evaluable: {guardado['motivo']})", "m12_casos.csv; modelo_guardado.json"),
        ("M3D", *_ms(por_semilla(pico, "alerta")),
         f"{pico['caso'].nunique()} escenarios x {len(seeds)} semillas", "m3_casos.csv; m3/s*_moderado_*, s*_grande_*"),
        ("M3F", *_ms(por_semilla(ctrl, "alerta")),
         f"{ctrl['caso'].nunique()} escenarios x {len(seeds)} semillas", "m3_casos.csv; m3/s*_control_*"),
        ("M3RD", float(pos["alerta_agente"].mean()) if len(pos) else float(
            "nan"), 0.0, f"{len(pos)} semanas con alerta real", "m3_retro_semanas.csv; m3/retro/"),
        ("M3RF", float(neg["alerta_agente"].mean()) if len(neg) else float(
            "nan"), 0.0, f"{len(neg)} semanas sin alerta real", "m3_retro_semanas.csv; m3/retro/"),
        ("R1", float(np.mean(rob)), float(np.std(rob)),
         f"{len(rob)} ejecuciones de falla esperada", "m3_casos.csv; m12_casos.csv"),
    ]
    tabla = []
    for clave, media, desv, casos, evidencia in filas:
        m = M[clave]
        if media is None:
            tabla.append({"Medida": f"{clave}: {m['nombre']}", "Fórmula": m["formula"], "Umbral": f"{m['op']} {m['umbral']}",
                          "Casos ejecutados": casos, "Resultado obtenido": "No evaluable", "¿Cumple?": "N/A",
                          "Evidencia": f"logs/fase3/{evidencia}"})
            continue
        tabla.append({"Medida": f"{clave}: {m['nombre']}", "Fórmula": m["formula"], "Umbral": f"{m['op']} {m['umbral']}",
                      "Casos ejecutados": casos, "Resultado obtenido": f"{media:.3f} ± {desv:.3f}",
                      "¿Cumple?": "SÍ" if OPS[m["op"]](media, m["umbral"]) else "NO", "Evidencia": f"logs/fase3/{evidencia}"})
    t = pd.DataFrame(tabla)
    t.to_csv(OUT / "tabla_resultados.csv", index=False, encoding="utf-8-sig")
    try:
        # necesita el paquete tabulate
        md = t.to_markdown(index=False)
    except ImportError:
        md = t.to_string(index=False)
    (OUT / "tabla_resultados.md").write_text(md, encoding="utf-8")
    print("\n" + t[["Medida", "Umbral", "Casos ejecutados",
          "Resultado obtenido", "¿Cumple?"]].to_string(index=False))
    print(f"\nEvidencia en {OUT}")


if __name__ == "__main__":
    main()
