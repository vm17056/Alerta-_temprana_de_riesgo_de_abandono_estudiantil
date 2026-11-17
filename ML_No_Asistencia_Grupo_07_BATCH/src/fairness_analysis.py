"""Herramientas de evaluación ética/fairness.

El análisis solo debe ejecutarse cuando el dataset tenga variables sensibles
observadas, por ejemplo género, edad o zona. Estas variables NO se usan como
predictores del modelo académico.
"""
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import recall_score


def fairness_report(y_true, y_pred, sensitive):
    return MetricFrame(
        metrics={"recall": recall_score, "selection_rate": selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
