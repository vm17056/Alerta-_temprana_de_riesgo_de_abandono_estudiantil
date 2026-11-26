from __future__ import annotations

import argparse
from pathlib import Path

from src.batch_analysis import exportar_diagnostico_excel, leer_archivo_tabular
from src.riesgo_abandono import cargar_modelo, diagnosticar_estudiantes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza un Excel/CSV completo de estudiantes y genera un diagnóstico de riesgo."
    )
    parser.add_argument(
        "archivo",
        type=Path,
        help="Archivo XLSX/XLS/CSV con estudiantes del salón.",
    )
    parser.add_argument(
        "--hoja",
        default=None,
        help="Nombre de la hoja de Excel. Por defecto utiliza la primera.",
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path("models/modelo_abandono_estudiantil.joblib"),
        help="Modelo supervisado opcional.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("outputs/diagnostico_salon.xlsx"),
        help="Archivo Excel de salida.",
    )
    args = parser.parse_args()

    df = leer_archivo_tabular(args.archivo, args.hoja)

    modelo = None
    if args.modelo.exists():
        modelo = cargar_modelo(args.modelo)

    resultado = diagnosticar_estudiantes(df, modelo=modelo)
    exportar_diagnostico_excel(resultado, args.salida, root=Path(__file__).resolve().parent)

    print("Análisis completado.")
    print(f"Archivo de entrada: {args.archivo}")
    print(f"Archivo de salida: {args.salida}")
    print()
    print(
        resultado[
            [
                c
                for c in [
                    "id_estudiante",
                    "grado",
                    "porcentaje_asistencia",
                    "promedio_notas",
                    "score_riesgo_final",
                    "nivel_riesgo",
                    "prioridad_seguimiento",
                ]
                if c in resultado.columns
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
