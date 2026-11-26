from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.riesgo_abandono import cargar_modelo, diagnosticar_estudiantes


def leer_dataset(ruta: Path) -> pd.DataFrame:
    if ruta.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(ruta)
    return pd.read_csv(ruta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera diagnóstico de riesgo de abandono por estudiante."
    )
    parser.add_argument("dataset", type=Path, help="CSV/XLSX con asistencia, notas y checklist.")
    parser.add_argument("--modelo", type=Path, default=None, help="Modelo .joblib opcional.")
    parser.add_argument("--salida", type=Path, default=Path("outputs/diagnostico_estudiantes.csv"))
    args = parser.parse_args()

    df = leer_dataset(args.dataset)
    modelo = cargar_modelo(args.modelo) if args.modelo else None
    resultado = diagnosticar_estudiantes(df, modelo=modelo)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.salida, index=False, encoding="utf-8-sig")

    print(f"Diagnóstico generado: {args.salida}")
    print(resultado[[
        c for c in ["id_estudiante", "grado", "riesgo_academico", "riesgo_checklist", "score_riesgo_final", "nivel_riesgo", "requiere_revision"]
        if c in resultado.columns
    ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
