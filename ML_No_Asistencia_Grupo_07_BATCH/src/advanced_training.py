from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.riesgo_abandono import _normalizar_objetivo, normalizar_columnas_entrada

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

ACADEMIC_FEATURES = ["porcentaje_asistencia", "promedio_notas"]
TARGET = "abandono"


def train_models(df: pd.DataFrame, target: str = TARGET):
    df = normalizar_columnas_entrada(df)
    missing = [c for c in [*ACADEMIC_FEATURES, target] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")

    X = df[ACADEMIC_FEATURES].copy()
    y = _normalizar_objetivo(df[target])
    if len(df) < 20:
        raise ValueError("Se requieren al menos 20 registros para el entrenamiento.")
    if y.value_counts().min() < 5:
        raise ValueError("Se requieren al menos 5 casos por clase para validación cruzada.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            eval_metric="logloss", random_state=42, n_estimators=200, max_depth=3
        )

    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "modelo": name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_f1": scores["test_f1"].mean(),
            "cv_roc_auc": scores["test_roc_auc"].mean(),
        })

    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    gb = models["gradient_boosting"]
    ensemble = VotingClassifier(
        estimators=[("rf", grid.best_estimator_), ("gb", gb)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)

    pred = ensemble.predict(X_test)
    prob = ensemble.predict_proba(X_test)[:, 1]
    final_metrics = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, prob)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "best_rf_params": grid.best_params_,
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(ensemble, "models/modelo_abandono_estudiantil.joblib")
    return pd.DataFrame(rows), final_metrics
