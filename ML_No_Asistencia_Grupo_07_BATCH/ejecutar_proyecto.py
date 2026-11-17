from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.riesgo_abandono import diagnosticar_estudiantes

RAIZ = Path(__file__).resolve().parent
ENTRADA_DEMO = RAIZ / "data" / "templates" / "plantilla_estudiantes.csv"
SALIDA_DEMO = RAIZ / "outputs" / "diagnostico_estudiantes.csv"


def main() -> None:
    """Ejecuta una demostración local sin fingir que existe un modelo entrenado."""
    df = pd.read_csv(ENTRADA_DEMO)
    resultado = diagnosticar_estudiantes(df, modelo=None)
    SALIDA_DEMO.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(SALIDA_DEMO, index=False, encoding="utf-8-sig")

    print("Proyecto ejecutado correctamente en modo diagnóstico heurístico.")
    print(f"Entrada demostrativa: {ENTRADA_DEMO.relative_to(RAIZ)}")
    print(f"Salida: {SALIDA_DEMO.relative_to(RAIZ)}")
    print()
    print(resultado[[
        "id_estudiante", "grado", "riesgo_academico", "riesgo_checklist",
        "score_riesgo_final", "nivel_riesgo", "requiere_revision"
    ]].to_string(index=False))
    print()
    print("Para entrenamiento supervisado use: python entrenar_modelo.py <historico.csv>")


if __name__ == "__main__":
    main()
