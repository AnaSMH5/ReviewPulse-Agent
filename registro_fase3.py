"""Fase 3 - Registro de corridas. Ejecuta evaluate.py y AGREGA el resultado a logs/fase3_registro.txt

    python registro_fase3.py                    # corrida completa (la que va en la entrega)
    python registro_fase3.py --seeds 2          # prueba rápida: queda marcada como NO VÁLIDA para la entrega
    python registro_fase3.py --solo-registrar    # no ejecuta nada: registra lo que haya hoy en logs/fase3/
    python registro_fase3.py --nota "qué cambié"  # agrega una nota a la corrida

Cada corrida deja:
  * un bloque nuevo al final de logs/fase3_registro.txt (nunca borra los anteriores) con las 7 columnas de la
    tabla de la entrega: medida, fórmula, umbral, casos ejecutados, resultado obtenido, ¿cumple?, evidencia;
  * una copia de los CSV de evidencia en logs/fase3_corridas/<id_de_corrida>/ (evaluate.py sobrescribe
    logs/fase3/ en cada corrida, así que sin esta copia la evidencia de las corridas anteriores se pierde).

También guarda lo que sirve para saber si la corrida es confiable: commit y cambios sin commitear, huella de
thresholds.json (avisa si cambió respecto a la corrida anterior, porque los umbrales deben fijarse ANTES de
ejecutar), parámetros del agente, huella de los datos y si el modelo guardado sirve como examen limpio.
Si evaluate.py falla, el bloque se escribe igual con el error y lo que haya quedado.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from reviewpulse import config

ROOT = config.ROOT
FASE3 = ROOT / "logs" / "fase3"
REGISTRO = ROOT / "logs" / "fase3_registro.txt"
CORRIDAS = ROOT / "logs" / "fase3_corridas"
THRESHOLDS = ROOT / "thresholds.json"
ARCHIVOS_EVIDENCIA = ["tabla_resultados.csv", "tabla_resultados.md", "m12_casos.csv", "m3_casos.csv",
                      "m3_retro_semanas.csv", "modelo_guardado.json"]

# Requisitos de la entrega
MIN_SEMILLAS, MIN_CASOS, MIN_FALLAS = 5, 10, 2
ANCHO = 78


# ---------------------------------------------------------------- utilidades
def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=20)
        return r.stdout.rstrip("\n") if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _sha(path: Path, n: int = 12) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n] if path.exists() else "(no existe)"


def _num(texto):
    """'0.617 ± 0.019' -> 0.617 ; '>= 0.85' -> 0.85 ; devuelve None si no hay número."""
    m = re.search(r"-?\d+(?:\.\d+)?", str(texto))
    return float(m.group()) if m else None


def _fecha(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _titulo(texto: str) -> str:
    return f"\n{'-' * 4} {texto} {'-' * max(3, ANCHO - len(texto) - 6)}"


def _leer_csv(nombre: str, desde: float):
    """Devuelve (DataFrame o None, aviso). Si el archivo es de una corrida anterior, lo dice."""
    ruta = FASE3 / nombre
    if not ruta.exists():
        return None, "no existe"
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    return df, ("" if ruta.stat().st_mtime >= desde - 0.5 else f"de una corrida anterior ({_fecha(ruta.stat().st_mtime)})")


# ---------------------------------------------------------------- ejecutar evaluate.py
def correr_evaluate(extra: list) -> tuple:
    """Ejecuta evaluate.py mostrando solo el avance. Devuelve (código, últimas líneas, n_avisos_simulados, segundos)."""
    t0, cola, simulados = time.time(), [], 0
    proc = subprocess.Popen([sys.executable, "-u", "evaluate.py", *extra], cwd=ROOT, text=True, encoding="utf-8",
                            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    for linea in proc.stdout:
        linea = linea.rstrip()
        if "[SIMULADO" in linea:               # un aviso simulado por cada alerta: se cuentan, no se muestran
            simulados += 1
            continue
        cola = (cola + [linea])[-25:]
        if linea.startswith("Medida"):
            print(f"  {linea}", flush=True)
    return proc.wait(), cola, simulados, time.time() - t0


# ---------------------------------------------------------------- contexto de la corrida
def contexto(extra_nota: str, solo_registrar: bool) -> dict:
    cfg = json.loads(THRESHOLDS.read_text(encoding="utf-8")) if THRESHOLDS.exists() else {}
    sucios = [l[3:] for l in (_git("status", "--porcelain") or "").splitlines()
              if not l[3:].startswith(("logs/", "reports/", "models/"))]
    previo = None
    if REGISTRO.exists():
        hallados = re.findall(r"thresholds\.json sha256\s*:\s*([0-9a-f]{12})", REGISTRO.read_text(encoding="utf-8"))
        previo = hallados[-1] if hallados else None
    try:
        import evaluate as ev
        entorno = {"p_fail": ev.P_FAIL, "p_missing": ev.P_MISSING}
    except Exception:
        entorno = {}
    return {"cfg": cfg, "sucios": sucios, "previo": previo, "entorno": entorno, "nota": extra_nota,
            "commit": (_git("log", "-1", "--format=%h \"%s\" (%cI)") or "sin git").strip(),
            "rama": (_git("rev-parse", "--abbrev-ref", "HEAD") or "?").strip(),
            "thr_commit": (_git("log", "-1", "--format=%h %cI", "--", "thresholds.json") or "sin commit").strip(),
            "thr_pendiente": bool(_git("status", "--porcelain", "--", "thresholds.json")),
            "thr_sha": _sha(THRESHOLDS)}


def parametros() -> str:
    c = config
    return (f"WINDOW_WEEKS={c.WINDOW_WEEKS} | MIN_REVIEWS={c.MIN_REVIEWS} | NEG_RISE={c.NEG_RISE} | "
            f"ALERT_P_VALUE={c.ALERT_P_VALUE} | NEGATIVE_MAX_RATING={c.NEGATIVE_MAX_RATING} | "
            f"TOPIC_REVIEW_THRESHOLD={c.TOPIC_REVIEW_THRESHOLD} | RANDOM_STATE={c.RANDOM_STATE}")


# ---------------------------------------------------------------- análisis de lo que dejó evaluate.py
def requisitos(m12, m3, retro, semillas: list) -> list:
    """Comprueba lo que exige la entrega: >=10 casos (normales, límite, >=2 de falla esperada) y >=5 repeticiones."""
    L = []
    ok = lambda b: "SÍ" if b else "NO"
    L.append(f"Repeticiones (semillas) : {len(semillas)} -> mínimo {MIN_SEMILLAS}: {ok(len(semillas) >= MIN_SEMILLAS)}")
    if m12 is not None:
        c = m12.drop_duplicates("caso").groupby("tipo").size().to_dict()
        total = sum(c.values())
        L.append(f"Medidas 1 y 2           : {total} casos distintos (normales {c.get('normal', 0)}, límite {c.get('límite', 0)}, "
                 f"falla esperada {c.get('falla_esperada', 0)}) -> >={MIN_CASOS}: {ok(total >= MIN_CASOS)} | "
                 f">={MIN_FALLAS} de falla: {ok(c.get('falla_esperada', 0) >= MIN_FALLAS)}")
    if m3 is not None:
        c = m3.drop_duplicates("caso").groupby("grupo").size().to_dict()
        total = sum(c.values())
        L.append(f"Medida 3 (escenarios)  : {total} escenarios distintos (control {c.get('control', 0)}, pico {c.get('pico', 0)}, "
                 f"falla esperada {c.get('falla_esperada', 0)}) -> >={MIN_CASOS}: {ok(total >= MIN_CASOS)} | "
                 f">={MIN_FALLAS} de falla: {ok(c.get('falla_esperada', 0) >= MIN_FALLAS)}")
    if retro is not None:
        ev = retro[(retro["n"] > 0) & (retro["status_real"] != "datos_insuficientes")]
        L.append(f"Medida 3 (retrospectiva): {len(retro)} semanas recorridas, {len(ev)} evaluables "
                 f"(con reseñas y con ventanas suficientes)")
    return L


def detalle(m12, m3, retro) -> list:
    """Desgloses útiles para interpretar la tabla. NO son parte de thresholds.json (son exploratorios)."""
    L = []
    if m12 is not None:
        L.append("Medidas 1 y 2 por caso (media ± desv. entre semillas):")
        g = m12.groupby(["caso", "tipo", "metrica"]).agg(n=("n", "first"), m=("valor", "mean"),
                                                        d=("valor", lambda s: s.std(ddof=0))).reset_index()
        for _, r in g.iterrows():
            L.append(f"  {r['caso']:<30s} {r['tipo']:<15s} n={int(r['n']):<4d} {r['metrica']:<18s} {r['m']:.3f} ± {r['d']:.3f}")
    if m3 is not None:
        pk = m3[m3["grupo"] == "pico"].copy()
        if len(pk):
            pk["intensidad"] = pk["caso"].str.split("_").str[0]
            L.append("Medida 3, detección según la intensidad del pico inyectado:")
            for nombre, g in pk.groupby("intensidad"):
                L.append(f"  {nombre:<10s} {int(g['alerta'].sum())}/{len(g)} alertas = {g['alerta'].mean():.3f} "
                         f"(k inyectado: mediana {g['spike'].median():.0f}, máx {g['spike'].max():.0f})")
        malos = m3[(m3["grupo"] == "falla_esperada") & (~m3["ok"].astype(bool))]
        L.append(f"Medida 3, fallas esperadas mal manejadas: {len(malos)}" +
                 ("" if malos.empty else " -> " + ", ".join(sorted(set(malos['caso'])))))
    if m12 is not None:
        malos = m12[(m12["tipo"] == "falla_esperada") & (m12["valor"] < 1.0)]
        L.append(f"Medidas 1 y 2, fallas esperadas mal manejadas: {len(malos)}" +
                 ("" if malos.empty else " -> " + ", ".join(f"{r.caso} ({r.valor:.2f})" for r in malos.itertuples())))
    if retro is not None:
        ev = retro[(retro["n"] > 0) & (retro["status_real"] != "datos_insuficientes")].copy()
        tp = int((ev["alerta_real"] & ev["alerta_agente"]).sum())
        fn = int((ev["alerta_real"] & ~ev["alerta_agente"]).sum())
        fp = int((~ev["alerta_real"] & ev["alerta_agente"]).sum())
        tn = int((~ev["alerta_real"] & ~ev["alerta_agente"]).sum())
        L.append(f"Retrospectiva (semanas): alerta real y agente alerta = {tp} | alerta real y agente NO = {fn} | "
                 f"falsa alarma = {fp} | ambos sin alerta = {tn}")
        # Episodios: semanas con alerta real separadas por <= 21 días cuentan como un mismo episodio.
        al = ev[ev["alerta_real"]].copy()
        if len(al):
            al["semana"] = pd.to_datetime(al["semana"])
            al = al.sort_values("semana")
            al["episodio"] = (al["semana"].diff().dt.days.fillna(0) > 21).cumsum()
            det = al.groupby("episodio")["alerta_agente"].any()
            L.append(f"Retrospectiva (episodios, exploratorio): {len(det)} episodios con alerta real; el agente alertó al menos "
                     f"una vez en {int(det.sum())} ({det.mean():.2f})")
        perdidas = ev[ev["alerta_real"] & ~ev["alerta_agente"]]["semana"].astype(str).tolist()
        if perdidas:
            L.append("Semanas con alerta real que el agente no detectó: " + ", ".join(perdidas))
    return L


def brecha(resultado: str, umbral: str, cumple: str) -> str:
    r, u = _num(resultado), _num(umbral)
    if r is None or u is None or cumple not in ("SÍ", "NO"):
        return ""
    d = abs(r - u) * 100
    return f"(margen +{d:.1f} puntos sobre el umbral)" if cumple == "SÍ" else f"(faltan {d:.1f} puntos para el umbral)"


# ---------------------------------------------------------------- escribir el bloque
def bloque(n: int, run_id: str, ctx: dict, ejec: dict, tabla, evid_dir: Path, semillas: list, extras: dict) -> str:
    sep = "=" * ANCHO
    L = [sep, f"CORRIDA #{n} | {datetime.now():%Y-%m-%d %H:%M:%S} | id {run_id}", sep]

    valida = len(semillas) >= MIN_SEMILLAS
    if ejec["modo"] == "solo_registrar":
        tipo = f"SOLO REGISTRO (no se ejecutó evaluate.py; se leyó lo que había en logs/fase3/)"
    elif valida:
        tipo = f"COMPLETA ({len(semillas)} semillas)"
    else:
        tipo = f"PRUEBA RÁPIDA - NO VÁLIDA PARA LA ENTREGA ({len(semillas)} semillas < {MIN_SEMILLAS} requeridas)"
    L.append(f"Tipo de corrida       : {tipo}")
    if ejec["modo"] == "correr":
        estado = f"OK, duró {int(ejec['seg'] // 60)} min {int(ejec['seg'] % 60)} s" if ejec["codigo"] == 0 \
            else f"FALLÓ (código de salida {ejec['codigo']})"
        L.append(f"Ejecución de evaluate : {estado} | avisos simulados generados por el agente: {ejec['simulados']}")
    if ctx["nota"]:
        L.append(f"Nota                  : {ctx['nota']}")
    L.append(f"Código                : commit {ctx['commit']} | rama {ctx['rama']}")
    L.append("Cambios sin commitear : " + (", ".join(ctx["sucios"]) if ctx["sucios"] else "ninguno") +
             ("   <- el resultado puede no corresponder al commit" if ctx["sucios"] else ""))
    cambio = "primera corrida registrada" if ctx["previo"] is None else \
        ("no" if ctx["previo"] == ctx["thr_sha"] else "SÍ  <-- OJO: los umbrales cambiaron desde la corrida anterior")
    L.append(f"thresholds.json sha256: {ctx['thr_sha']} | fijado_el {ctx['cfg'].get('fijado_el', '?')} | "
             f"último commit {ctx['thr_commit']} | modificaciones sin commitear: {'SÍ' if ctx['thr_pendiente'] else 'no'}")
    L.append(f"¿Cambiaron los umbrales desde la corrida anterior?: {cambio}")
    L.append(f"Parámetros del agente : {parametros()}")
    if ctx["entorno"]:
        L.append(f"Entorno simulado      : la fuente falla con p={ctx['entorno']['p_fail']} y {ctx['entorno']['p_missing']:.0%} de reseñas sin fecha")
    L.append(f"Datos                 : reviews_eda {_sha(config.DATA_PROCESSED / 'reviews_eda.csv', 8)} | "
             f"train {_sha(config.DATA_PROCESSED / 'train.csv', 8)} | test {_sha(config.DATA_PROCESSED / 'test.csv', 8)}")
    L.append(f"Modelo guardado       : {'válido como examen limpio' if extras['modelo'].get('valido') else 'NO válido como examen'} "
             f"- {extras['modelo'].get('motivo', 'sin información')}")
    L.append(f"Semillas              : {semillas}")

    if ejec["modo"] == "correr" and ejec["codigo"] != 0:
        L.append(_titulo("ERROR DE evaluate.py (últimas líneas)"))
        L += [f"  {l}" for l in ejec["cola"][-15:]]

    if tabla is None:
        L += [_titulo("TABLA DE RESULTADOS"), "  No hay tabla de resultados que registrar en esta corrida."]
    else:
        L.append(_titulo("TABLA DE RESULTADOS (resumen)"))
        L.append(f"  {'Medida':<8s}{'Umbral':<10s}{'Resultado obtenido':<22s}{'¿Cumple?':<10s}Casos ejecutados")
        for r in tabla.itertuples(index=False):
            clave = str(r[0]).split(":")[0]
            L.append(f"  {clave:<8s}{str(r[2]):<10s}{str(r[4]):<22s}{str(r[5]):<10s}{r[3]}")
        cumplen = int((tabla["¿Cumple?"] == "SÍ").sum())
        evaluables = int(tabla["¿Cumple?"].isin(["SÍ", "NO"]).sum())
        L.append(f"  -> {cumplen} de {evaluables} medidas evaluables cumplen su umbral "
                 f"({int((tabla['¿Cumple?'] == 'N/A').sum())} no evaluables).")

        L.append(_titulo("FICHAS (las 7 columnas de la entrega)"))
        for r in tabla.to_dict("records"):
            clave, _, nombre = str(r["Medida"]).partition(": ")
            evid = ", ".join(f"{evid_dir.relative_to(ROOT).as_posix()}/{p.strip()}" if p.strip().endswith((".csv", ".json", ".md"))
                             else f"logs/fase3/{p.strip()} (logs del agente; se sobrescriben en la siguiente corrida)"
                             for p in str(r["Evidencia"]).replace("logs/fase3/", "").split(";"))
            L += [f"[{clave}] {nombre}",
                  f"  Fórmula           : {r['Fórmula']}",
                  f"  Umbral            : {r['Umbral']} (fijado en thresholds.json antes de ejecutar)",
                  f"  Casos ejecutados  : {r['Casos ejecutados']}",
                  f"  Resultado obtenido: {r['Resultado obtenido']}",
                  f"  ¿Cumple?          : {r['¿Cumple?']} {brecha(r['Resultado obtenido'], r['Umbral'], r['¿Cumple?'])}".rstrip(),
                  f"  Evidencia         : {evid}", ""]
        L.pop()

    if extras["requisitos"]:
        L.append(_titulo("DISEÑO DE LA PRUEBA (requisitos de la entrega)"))
        L += [f"  {x}" for x in extras["requisitos"]]
    if extras["detalle"]:
        L.append(_titulo("DETALLE (exploratorio: no forma parte de thresholds.json)"))
        L += [f"  {x}" for x in extras["detalle"]]
    if extras["avisos"]:
        L.append(_titulo("AVISOS"))
        L += [f"  - {x}" for x in extras["avisos"]]

    L += [_titulo("NOTAS DE LECTURA"),
          "  - Desv. = desviación estándar poblacional entre semillas (ddof=0), como la calcula evaluate.py.",
          "  - M1/M2/M1R/M2R usan predicciones fuera de muestra (validación cruzada) sobre las 300 reseñas; la etiqueta",
          "    de polaridad es el rating de la reseña y la de tema es la de reviews_eda.csv, no un etiquetado manual aparte.",
          "  - La 'verdad' de M3RD/M3RF es la misma regla de alerta aplicada a los ratings reales: mide cuánto la degrada",
          "    el clasificador, no eventos reales identificados por personas. M3D/M3F usan picos inyectados (simulación).",
          f"  - Evidencia de esta corrida: {evid_dir.relative_to(ROOT).as_posix()}/ (logs/fase3/ se sobrescribe en cada corrida).",
          "  - Capturas / minuto del video: (agregar a mano en el documento de entrega)",
          ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- principal
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=None, help="usar solo las primeras N semillas (prueba rápida)")
    ap.add_argument("--solo-registrar", action="store_true", help="no ejecuta evaluate.py; registra lo que haya en logs/fase3/")
    ap.add_argument("--nota", default="", help="comentario para esta corrida")
    a = ap.parse_args()

    inicio = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx = contexto(a.nota, a.solo_registrar)
    todas = ctx["cfg"].get("semillas", [])
    semillas = todas[: a.seeds] if a.seeds else todas

    ejec = {"modo": "solo_registrar", "codigo": None, "cola": [], "simulados": 0, "seg": 0.0}
    if not a.solo_registrar:
        print(f"Corrida {run_id}: ejecutando evaluate.py ({len(semillas)} semillas). Puede tardar varios minutos...")
        codigo, cola, sim, seg = correr_evaluate(["--seeds", str(a.seeds)] if a.seeds else [])
        ejec = {"modo": "correr", "codigo": codigo, "cola": cola, "simulados": sim, "seg": seg}
    desde = 0.0 if a.solo_registrar else inicio

    tabla, aviso_t = _leer_csv("tabla_resultados.csv", desde)
    m12, aviso_12 = _leer_csv("m12_casos.csv", desde)
    m3, aviso_3 = _leer_csv("m3_casos.csv", desde)
    retro, aviso_r = _leer_csv("m3_retro_semanas.csv", desde)
    avisos = [f"{n}: {w}" + (" -> se ignora" if w.startswith("de una") else "")
              for n, w in (("tabla_resultados.csv", aviso_t), ("m12_casos.csv", aviso_12),
                           ("m3_casos.csv", aviso_3), ("m3_retro_semanas.csv", aviso_r)) if w]
    # un CSV de una corrida anterior NO se mezcla con esta corrida
    tabla, m12, m3, retro = (None if av.startswith("de una") else df
                             for df, av in ((tabla, aviso_t), (m12, aviso_12), (m3, aviso_3), (retro, aviso_r)))
    if a.solo_registrar:                       # el modo no ejecuta nada: las semillas salen de lo que haya en los CSV
        if m12 is not None:
            semillas = sorted(int(s) for s in m12["semilla"].astype(str).unique() if s.isdigit())
        ruta = FASE3 / "tabla_resultados.csv"
        avisos.append("modo --solo-registrar: se registró lo que había en logs/fase3/" +
                      (f" (generado {_fecha(ruta.stat().st_mtime)})" if ruta.exists() else ""))
    if ejec["modo"] == "correr" and ejec["codigo"] != 0:
        # evaluate.py escribe la tabla SOLO al final: tras un fallo, la que haya en disco es de una corrida anterior
        tabla = None
        avisos.append("evaluate.py falló: no hay tabla de resultados de esta corrida; se registra solo lo que alcanzó a guardar")
    if ctx["previo"] is not None and ctx["previo"] != ctx["thr_sha"]:
        avisos.append("thresholds.json cambió desde la corrida anterior: si se cambió DESPUÉS de ver resultados, "
                      "hay que declararlo en el documento (los umbrales deben fijarse antes de ejecutar)")
    if len(semillas) < MIN_SEMILLAS:
        avisos.append(f"menos de {MIN_SEMILLAS} semillas: no cumple el mínimo de repeticiones de la entrega")
    guardado = {}
    gp = FASE3 / "modelo_guardado.json"
    if gp.exists() and gp.stat().st_mtime >= desde - 0.5:
        guardado = json.loads(gp.read_text(encoding="utf-8"))
        if not guardado.get("valido"):
            avisos.append("M1D y M2D no se pudieron evaluar: ejecuta `python train.py` (sin --all) y vuelve a correr")

    # Copia de la evidencia de ESTA corrida (evaluate.py sobrescribe logs/fase3/)
    evid = CORRIDAS / run_id
    evid.mkdir(parents=True, exist_ok=True)
    for nombre in ARCHIVOS_EVIDENCIA:
        if (FASE3 / nombre).exists():
            shutil.copy2(FASE3 / nombre, evid / nombre)

    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    previo = REGISTRO.read_text(encoding="utf-8") if REGISTRO.exists() else ""
    n = len(re.findall(r"^CORRIDA #\d+", previo, flags=re.M)) + 1
    texto = bloque(n, run_id, ctx, ejec, tabla, evid, semillas,
                   {"modelo": guardado, "requisitos": requisitos(m12, m3, retro, semillas),
                    "detalle": detalle(m12, m3, retro), "avisos": avisos})
    with REGISTRO.open("a", encoding="utf-8") as f:
        f.write(("\n" if previo else "") + texto)

    print(f"\nCorrida #{n} registrada en {REGISTRO.relative_to(ROOT).as_posix()} "
          f"(evidencia: {evid.relative_to(ROOT).as_posix()}/)")
    if tabla is not None:
        print(tabla[["Medida", "Umbral", "Resultado obtenido", "¿Cumple?"]].assign(
            Medida=tabla["Medida"].str.split(":").str[0]).to_string(index=False))
    if ejec["modo"] == "correr" and ejec["codigo"] != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
