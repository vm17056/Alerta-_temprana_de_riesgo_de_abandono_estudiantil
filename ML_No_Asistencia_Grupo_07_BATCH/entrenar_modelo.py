from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.advanced_training import train_models


def leer_dataset(ruta: Path) -> pd.DataFrame:
    if ruta.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(ruta)
    return pd.read_csv(ruta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo avanzado de abandono con histórico etiquetado."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="CSV/XLSX con asistencia, notas y abandono.",
    )
    args = parser.parse_args()

    df = leer_dataset(args.dataset)
    metricas_cv, metricas_finales = train_models(df)

    print("Entrenamiento avanzado completado.")
    print()
    print("Validación cruzada:")
    print(metricas_cv.to_string(index=False))
    print()
    print("Prueba final:")
    for clave, valor in metricas_finales.items():
        if clave != "classification_report":
            print(f"{clave}: {valor}")
    print()
    print(
        "Modelo guardado en: models/modelo_abandono_estudiantil.joblib"
    )


if __name__ == "__main__":
    main()
