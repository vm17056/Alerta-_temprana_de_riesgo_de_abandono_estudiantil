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
        description="Entrena el modelo supervisado avanzado de abandono con histórico etiquetado."
    )
    parser.add_argument("dataset", type=Path, help="CSV/XLSX con asistencia, notas y abandono.")
    parser.add_argument("--salida-reportes", type=Path, default=Path("reports/training"))
    parser.add_argument("--salida-modelo", type=Path, default=Path("models"))
    args = parser.parse_args()

    df = leer_dataset(args.dataset)
    metricas_cv, metricas_finales = train_models(
        df,
        output_dir=args.salida_reportes,
        model_dir=args.salida_modelo,
    )

    print("Entrenamiento supervisado avanzado completado.")
    print()
    print("Validación cruzada:")
    print(metricas_cv.to_string(index=False))
    print()
    print("Prueba final:")
    for clave in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "split_method"]:
        if clave in metricas_finales:
            print(f"{clave}: {metricas_finales[clave]}")
    print()
    print("Mejores hiperparámetros Random Forest:")
    print(metricas_finales.get("best_rf_params", {}))
    print()
    print(f"Modelo guardado en: {args.salida_modelo / 'modelo_abandono_estudiantil.joblib'}")
    print(f"Reportes guardados en: {args.salida_reportes}")


if __name__ == "__main__":
    main()
