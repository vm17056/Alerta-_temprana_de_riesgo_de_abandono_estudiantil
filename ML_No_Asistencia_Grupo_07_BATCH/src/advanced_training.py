"""Flujo oficial de entrenamiento supervisado del proyecto.

El modelo utiliza únicamente asistencia y promedio de notas como predictores.
El checklist social se aplica posteriormente como complemento del diagnóstico y
no se utiliza como predictor entrenado, de acuerdo con el planteamiento de la tesina.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.riesgo_abandono import (
    FEATURES_ACADEMICAS,
    COLUMNA_FECHA_CORTE,
    COLUMNA_OBJETIVO,
    _normalizar_objetivo,
    normalizar_columnas_entrada,
    validar_dataset_estudiantes,
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

ACADEMIC_FEATURES = FEATURES_ACADEMICAS
TARGET = COLUMNA_OBJETIVO


def _metricas(y_true, pred, prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "brier_score": float(brier_score_loss(y_true, prob)),
    }


def _split_train_test(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    """Usa división temporal si existe fecha_corte válida; de lo contrario, estratificada."""
    nota = "train_test_split estratificado"
    if COLUMNA_FECHA_CORTE in df.columns:
        fechas = pd.to_datetime(df[COLUMNA_FECHA_CORTE], errors="coerce")
        validas = fechas.notna().sum()
        if validas == len(df) and fechas.nunique() >= 2:
            orden = np.argsort(fechas.to_numpy())
            n_test = max(1, int(round(len(df) * test_size)))
            train_idx = orden[:-n_test]
            test_idx = orden[-n_test:]
            if y.iloc[train_idx].nunique() == 2 and y.iloc[test_idx].nunique() == 2:
                nota = "separación temporal por fecha_corte"
                return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], nota

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, nota


def _modelo_calibrado(modelo):
    """Compatibilidad entre versiones de scikit-learn."""
    try:
        return CalibratedClassifierCV(estimator=modelo, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=modelo, method="sigmoid", cv=3)


def train_models(
    df: pd.DataFrame,
    target: str = TARGET,
    random_state: int = 42,
    test_size: float = 0.20,
    output_dir: str | Path = "reports/training",
    model_dir: str | Path = "models",
):
    """Entrena, optimiza, valida y guarda el ensemble oficial del proyecto."""
    df = normalizar_columnas_entrada(df)
    missing = [c for c in [*ACADEMIC_FEATURES, target] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")

    limpio = validar_dataset_estudiantes(df, requiere_objetivo=True, permitir_academicos_faltantes=False)
    X = limpio[ACADEMIC_FEATURES].copy()
    y = limpio[target].copy()

    if len(limpio) < 30:
        raise ValueError("Se recomiendan al menos 30 registros para un entrenamiento reproducible.")
    if y.value_counts().min() < 5:
        raise ValueError("Se requieren al menos 5 casos por clase para validación cruzada.")

    X_train, X_test, y_train, y_test, split_method = _split_train_test(
        limpio, X, y, test_size=test_size, random_state=random_state
    )

    min_class_train = int(y_train.value_counts().min())
    n_splits = min(5, min_class_train)
    if n_splits < 3:
        raise ValueError("No hay suficientes casos por clase para realizar validación cruzada.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state)),
    ])
    rf_base = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=random_state)

    models = {
        "regresion_logistica": logistic,
        "random_forest": rf_base,
        "gradient_boosting": gb,
    }
    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
        )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    cv_rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_rows.append({
            "modelo": name,
            **{f"cv_{metrica}_media": float(np.mean(scores[f"test_{metrica}"])) for metrica in scoring},
            **{f"cv_{metrica}_std": float(np.std(scores[f"test_{metrica}"], ddof=1)) for metrica in scoring},
        })

    grid = GridSearchCV(
        estimator=rf_base,
        param_grid={
            "n_estimators": [150, 300],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt"],
        },
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    tuned_rf = grid.best_estimator_

    tuned_scores = cross_validate(tuned_rf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_rows.append({
        "modelo": "random_forest_optimizado",
        **{f"cv_{metrica}_media": float(np.mean(tuned_scores[f"test_{metrica}"])) for metrica in scoring},
        **{f"cv_{metrica}_std": float(np.std(tuned_scores[f"test_{metrica}"], ddof=1)) for metrica in scoring},
    })

    # Ensemble oficial: RF optimizado + Gradient Boosting + Logistic Regression.
    ensemble_estimators = [
        ("rf", tuned_rf),
        ("gb", gb),
        ("logistic", logistic),
    ]
    if XGB_AVAILABLE:
        ensemble_estimators.append(("xgb", models["xgboost"]))

    ensemble = VotingClassifier(estimators=ensemble_estimators, voting="soft", n_jobs=-1)
    ensemble.fit(X_train, y_train)

    # Se calibra la probabilidad del ensemble usando únicamente el conjunto de entrenamiento.
    calibrated = _modelo_calibrado(ensemble)
    calibrated.fit(X_train, y_train)
    calibrated._modelo_calibrado = True

    pred = calibrated.predict(X_test)
    prob = calibrated.predict_proba(X_test)[:, 1]
    final_metrics = _metricas(y_test, pred, prob)
    final_metrics.update({
        "split_method": split_method,
        "n_registros": int(len(limpio)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "prevalencia_abandono": float(y.mean()),
        "cv_folds": int(n_splits),
        "best_rf_params": grid.best_params_,
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    })

    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=min(5, max(2, len(y_test) // 10)), strategy="quantile")
    calibration_df = pd.DataFrame({"probabilidad_predicha": prob_pred, "frecuencia_observada": prob_true})

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_f1_media", ascending=False).reset_index(drop=True)
    test_df = pd.DataFrame([{"modelo_final": "VotingClassifier_calibrado", **{k: v for k, v in final_metrics.items() if isinstance(v, (int, float))}}])
    pd.DataFrame([grid.best_params_]).to_json(output / "mejores_hiperparametros.json", orient="records", indent=2)
    cv_df.to_csv(output / "metricas_validacion_cruzada.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output / "metricas_prueba_final.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_matrix(y_test, pred), index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).to_csv(output / "matriz_confusion.csv")
    calibration_df.to_csv(output / "calibracion.csv", index=False)

    metadata = {
        "modelo_final": "VotingClassifier_calibrado",
        "features": ACADEMIC_FEATURES,
        "objetivo": TARGET,
        "split_method": split_method,
        "cv_folds": n_splits,
        "xgboost_disponible": XGB_AVAILABLE,
        "nota_metodologica": "El checklist no participa en el entrenamiento. Se aplica después como complemento contextual.",
        "mejores_hiperparametros_random_forest": grid.best_params_,
        "metricas_prueba_final": {k: v for k, v in final_metrics.items() if k not in {"classification_report"}},
    }
    (output / "resumen_entrenamiento.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    joblib.dump(calibrated, model_path / "modelo_abandono_estudiantil.joblib")
    (model_path / "metadata_modelo_abandono.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    return cv_df, final_metrics
