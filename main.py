import pandas as pd

from reviewpulse import config
from reviewpulse.analysis import compare
from reviewpulse.data import load_reviews
from reviewpulse.metrics import weekly_series


def main():
    df = load_reviews()
    serie = weekly_series(df)
    print(f"{len(df)} reviews | {serie.index.min().date()} -> {serie.index.max().date()} | {len(serie)} semanas\n")

    print("Últimas 5 semanas con reviews:")
    print(serie[serie["n"] > 0].tail(5).round(2), "\n")

    filas = [compare(serie, semana) for semana in serie.index]

    print(f"Backtest sobre {len(filas)} semanas (ventanas de {config.WINDOW_WEEKS} semanas):")
    print(pd.Series([f["status"] for f in filas]).value_counts().to_string(), "\n")

    print("Semanas con alerta:")
    for f in filas:
        if f["alert"]:
            print(f"  {f['week']}  {f['reason']}")

    config.REPORTS.mkdir(exist_ok=True)
    pd.json_normalize(filas).to_csv(config.REPORTS / "backtest.csv", index=False)


if __name__ == "__main__":
    main()
