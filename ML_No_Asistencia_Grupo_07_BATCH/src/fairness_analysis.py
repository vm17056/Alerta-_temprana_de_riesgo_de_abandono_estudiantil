"""Evaluación ética y de equidad del modelo.

Las variables sensibles se utilizan únicamente para auditar resultados por grupo.
No forman parte de FEATURES_ACADEMICAS ni del entrenamiento del modelo.
"""
from __future__ import annotations

import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate
from sklearn.metrics import recall_score


def fairness_report(y_true, y_pred, sensitive) -> pd.DataFrame:
    """Devuelve métricas de desempeño por grupo sensible."""
    frame = MetricFrame(
        metrics={
            "recall": recall_score,
            "false_positive_rate": false_positive_rate,
            "selection_rate": selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    resultado = frame.by_group.reset_index()
    resultado = resultado.rename(columns={resultado.columns[0]: "grupo"})
    return resultado


def resumen_disparidades(reporte: pd.DataFrame) -> dict:
    """Calcula rangos entre grupos para apoyar la revisión humana."""
    columnas = [c for c in ["recall", "false_positive_rate", "selection_rate"] if c in reporte.columns]
    resumen = {}
    for columna in columnas:
        valores = pd.to_numeric(reporte[columna], errors="coerce").dropna()
        resumen[columna] = {
            "minimo": float(valores.min()) if not valores.empty else None,
            "maximo": float(valores.max()) if not valores.empty else None,
            "brecha_maxima": float(valores.max() - valores.min()) if not valores.empty else None,
        }
    return resumen
