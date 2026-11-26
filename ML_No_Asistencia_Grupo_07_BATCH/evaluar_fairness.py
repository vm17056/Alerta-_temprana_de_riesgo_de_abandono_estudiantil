from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.fairness_analysis import fairness_report, resumen_disparidades
from src.riesgo_abandono import cargar_modelo, normalizar_columnas_entrada, _normalizar_objetivo, FEATURES_ACADEMICAS


def leer(ruta: Path) -> pd.DataFrame:
    if ruta.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(ruta)
    return pd.read_csv(ruta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un reporte de fairness sin usar variables sensibles como predictores.")
    parser.add_argument("dataset", type=Path, help="Histórico con asistencia, notas, abandono y una variable sensible.")
    parser.add_argument("variable_sensible", help="Nombre de la columna sensible, por ejemplo genero o zona.")
    parser.add_argument("--modelo", type=Path, default=Path("models/modelo_abandono_estudiantil.joblib"))
    parser.add_argument("--salida", type=Path, default=Path("reports/fairness/fairness_report.csv"))
    args = parser.parse_args()

    df = normalizar_columnas_entrada(leer(args.dataset))
    if args.variable_sensible not in df.columns:
        raise ValueError(f"No existe la variable sensible '{args.variable_sensible}'.")
    if "abandono" not in df.columns:
        raise ValueError("El dataset debe contener la columna 'abandono'.")

    modelo = cargar_modelo(args.modelo)
    y_true = _normalizar_objetivo(df["abandono"])
    X = df[FEATURES_ACADEMICAS].apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1) & df[args.variable_sensible].notna()
    if mask.sum() < 10:
        raise ValueError("Se requieren al menos 10 registros completos para un reporte exploratorio de fairness.")

    pred = modelo.predict(X.loc[mask])
    reporte = fairness_report(y_true.loc[mask], pred, df.loc[mask, args.variable_sensible])
    resumen = resumen_disparidades(reporte)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    reporte.to_csv(args.salida, index=False, encoding="utf-8-sig")
    (args.salida.parent / "fairness_disparidades.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Reporte generado: {args.salida}")
    print(reporte.to_string(index=False))
    print("\nBrechas observadas:")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
