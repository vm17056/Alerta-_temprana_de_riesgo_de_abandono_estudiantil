from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_ACADEMICAS = ["porcentaje_asistencia", "promedio_notas"]
COLUMNA_OBJETIVO = "abandono"

# Alias aceptados al importar archivos de Excel/CSV.
ALIASES_COLUMNAS = {
    "id_estudiante": {
        "id_estudiante", "id estudiante", "id", "codigo", "código",
        "codigo_estudiante", "código_estudiante", "student_id", "estudiante"
    },
    "grado": {"grado", "nivel", "curso", "grado_nivel", "nivel_educativo"},
    "porcentaje_asistencia": {
        "porcentaje_asistencia", "porcentaje asistencia", "asistencia",
        "asistencia_%", "asistencia (%)", "attendance", "attendance_percent"
    },
    "promedio_notas": {
        "promedio_notas", "promedio notas", "promedio", "nota_promedio",
        "nota promedio", "notas", "calificacion", "calificación", "grade_average"
    },
    "trabajo_infantil": {"trabajo_infantil", "trabajo infantil"},
    "migracion": {"migracion", "migración"},
    "embarazo_adolescente": {"embarazo_adolescente", "embarazo adolescente"},
    "pobreza": {"pobreza", "situacion pobreza", "situación pobreza"},
    "vive_lejos": {"vive_lejos", "vive lejos", "distancia", "vive lejos del centro educativo"},
    "bullying": {"bullying", "acoso escolar", "acoso"},
    "abandono": {"abandono", "desercion", "deserción", "abandono escolar", "dropout"},
}

CRITERIOS_MAYOR_PESO = [
    "trabajo_infantil",
    "migracion",
    "embarazo_adolescente",
    "pobreza",
]
CRITERIOS_PESO_NORMAL = [
    "vive_lejos",
    "bullying",
]
CRITERIOS_CHECKLIST = CRITERIOS_MAYOR_PESO + CRITERIOS_PESO_NORMAL

PESOS_CHECKLIST = {
    **{criterio: 2.0 for criterio in CRITERIOS_MAYOR_PESO},
    **{criterio: 1.0 for criterio in CRITERIOS_PESO_NORMAL},
}


@dataclass(frozen=True)
class ConfiguracionRiesgo:
    """Parámetros de diagnóstico configurables por la institución.

    Los valores por defecto son heurísticos del proyecto, no umbrales oficiales.
    Deben calibrarse con datos históricos y criterios del centro educativo.
    """

    umbral_asistencia: float = 85.0
    asistencia_critica: float = 70.0
    nota_aprobacion: float = 6.0
    nota_critica: float = 4.0
    peso_academico: float = 0.70
    peso_checklist: float = 0.30
    corte_riesgo_medio: float = 0.40
    corte_riesgo_alto: float = 0.65

    def validar(self) -> None:
        if not (0 <= self.asistencia_critica < self.umbral_asistencia <= 100):
            raise ValueError("Los umbrales de asistencia son inválidos.")
        if not (0 <= self.nota_critica < self.nota_aprobacion):
            raise ValueError("Los umbrales de notas son inválidos.")
        if not np.isclose(self.peso_academico + self.peso_checklist, 1.0):
            raise ValueError("peso_academico + peso_checklist debe ser 1.0.")
        if not (0 <= self.corte_riesgo_medio < self.corte_riesgo_alto <= 1):
            raise ValueError("Los cortes de riesgo son inválidos.")


CONFIG_DEFAULT = ConfiguracionRiesgo()


def normalizar_columnas_entrada(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres comunes de columnas provenientes de Excel/CSV."""
    salida = df.copy()
    renombres = {}
    mapa = {}
    for destino, aliases in ALIASES_COLUMNAS.items():
        for alias in aliases:
            clave = " ".join(str(alias).strip().lower().replace("-", " ").replace("_", " ").split())
            mapa[clave] = destino
    for columna in salida.columns:
        clave = " ".join(str(columna).strip().lower().replace("-", " ").replace("_", " ").split())
        if clave in mapa and mapa[clave] not in salida.columns:
            renombres[columna] = mapa[clave]
    return salida.rename(columns=renombres)


def _normalizar_booleano(valor) -> int:
    if pd.isna(valor):
        return 0
    if isinstance(valor, (bool, np.bool_)):
        return int(valor)
    if isinstance(valor, (int, np.integer, float, np.floating)):
        return int(float(valor) > 0)
    texto = str(valor).strip().lower()
    verdaderos = {"1", "si", "sí", "s", "yes", "y", "true", "verdadero", "x"}
    falsos = {"0", "no", "n", "false", "falso", ""}
    if texto in verdaderos:
        return 1
    if texto in falsos:
        return 0
    raise ValueError(f"Valor de checklist no reconocido: {valor!r}")


def _normalizar_objetivo(serie: pd.Series) -> pd.Series:
    valores = serie.map(_normalizar_booleano).astype("int8")
    if valores.nunique() < 2:
        raise ValueError("La columna 'abandono' debe contener casos de abandono y no abandono.")
    return valores


def validar_dataset_estudiantes(df: pd.DataFrame, requiere_objetivo: bool = False) -> pd.DataFrame:
    df = normalizar_columnas_entrada(df)
    faltantes = [c for c in FEATURES_ACADEMICAS if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas académicas obligatorias: " + ", ".join(faltantes)
        )
    if requiere_objetivo and COLUMNA_OBJETIVO not in df.columns:
        raise ValueError("Para entrenar se requiere la columna histórica 'abandono'.")

    limpio = df.copy()
    for columna in FEATURES_ACADEMICAS:
        limpio[columna] = pd.to_numeric(limpio[columna], errors="coerce")
    if limpio[FEATURES_ACADEMICAS].isna().any().any():
        filas = limpio.index[limpio[FEATURES_ACADEMICAS].isna().any(axis=1)].tolist()[:10]
        raise ValueError(f"Hay valores académicos vacíos/no numéricos. Filas: {filas}")

    if not limpio["porcentaje_asistencia"].between(0, 100).all():
        raise ValueError("porcentaje_asistencia debe estar entre 0 y 100.")

    # Se admite escala 0-10 o 0-100 para notas. Una escala distinta debe normalizarse antes.
    max_nota = float(limpio["promedio_notas"].max())
    min_nota = float(limpio["promedio_notas"].min())
    if min_nota < 0 or max_nota > 100:
        raise ValueError("promedio_notas debe estar en una escala 0-10 o 0-100.")
    if max_nota > 10:
        limpio["promedio_notas"] = limpio["promedio_notas"] / 10.0
        limpio.attrs["nota_convertida_de_100_a_10"] = True

    if requiere_objetivo and COLUMNA_OBJETIVO in limpio.columns:
        limpio[COLUMNA_OBJETIVO] = _normalizar_objetivo(limpio[COLUMNA_OBJETIVO])

    return limpio


def puntaje_academico_heuristico(
    df: pd.DataFrame, config: ConfiguracionRiesgo = CONFIG_DEFAULT
) -> pd.Series:
    """Devuelve riesgo académico 0..1 con reglas lineales transparentes.

    50% proviene de asistencia y 50% de promedio de notas. Este score NO es una
    probabilidad estadística de abandono; se usa únicamente cuando no existe un
    modelo entrenado con una etiqueta histórica de abandono.
    """
    config.validar()
    limpio = validar_dataset_estudiantes(df)

    asistencia = limpio["porcentaje_asistencia"].astype(float)
    riesgo_asistencia = (
        (config.umbral_asistencia - asistencia)
        / (config.umbral_asistencia - config.asistencia_critica)
    ).clip(0, 1)

    notas = limpio["promedio_notas"].astype(float)
    riesgo_notas = (
        (config.nota_aprobacion - notas)
        / (config.nota_aprobacion - config.nota_critica)
    ).clip(0, 1)

    return (0.5 * riesgo_asistencia + 0.5 * riesgo_notas).rename("riesgo_academico")


def puntaje_checklist(df: pd.DataFrame) -> pd.Series:
    """Calcula un score 0..1 con la escala fija del checklist.

    La escala completa es 10 puntos: cuatro criterios de peso 2 y dos de peso 1.
    Los criterios no informados no se consideran positivos, pero tampoco aumentan
    el riesgo. El diagnóstico final ajusta proporcionalmente el peso social según
    la cantidad de información disponible.
    """
    salida = df.copy()
    acumulado = pd.Series(0.0, index=salida.index)

    for criterio, peso in PESOS_CHECKLIST.items():
        if criterio not in salida.columns:
            continue
        valores = salida[criterio]
        informado = ~valores.isna()
        normalizados = valores.map(_normalizar_booleano)
        acumulado = acumulado + normalizados * peso * informado.astype(float)

    total_maximo = sum(PESOS_CHECKLIST.values())
    return (acumulado / total_maximo).rename("riesgo_checklist")


def peso_checklist_disponible(df: pd.DataFrame) -> pd.Series:
    """Devuelve el peso 0..10 de los criterios realmente informados por fila."""
    peso = pd.Series(0.0, index=df.index)
    for criterio, valor in PESOS_CHECKLIST.items():
        if criterio in df.columns:
            peso = peso + df[criterio].notna().astype(float) * valor
    return peso.rename("peso_checklist_disponible")


def estado_checklist(df: pd.DataFrame) -> pd.Series:
    """Indica si el checklist no está informado, es parcial o está completo."""
    columnas = [c for c in CRITERIOS_CHECKLIST if c in df.columns]
    if not columnas:
        return pd.Series("NO INFORMADO", index=df.index, name="estado_checklist")
    informados = df[columnas].notna().sum(axis=1)
    return informados.map(
        lambda n: "COMPLETO" if n == len(CRITERIOS_CHECKLIST)
        else ("PARCIAL" if n > 0 else "NO INFORMADO")
    ).rename("estado_checklist")


def clasificar_riesgo(score: float, config: ConfiguracionRiesgo = CONFIG_DEFAULT) -> str:
    if score >= config.corte_riesgo_alto:
        return "ALTO"
    if score >= config.corte_riesgo_medio:
        return "MEDIO"
    return "BAJO"


def factores_detectados(fila: pd.Series) -> str:
    encontrados: list[str] = []
    for criterio in CRITERIOS_CHECKLIST:
        if criterio in fila.index and _normalizar_booleano(fila[criterio]):
            encontrados.append(criterio)
    return ", ".join(encontrados) if encontrados else "Ninguno reportado"


def entrenar_modelo(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.25,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Entrena modelos supervisados usando EXCLUSIVAMENTE asistencia y notas."""
    limpio = validar_dataset_estudiantes(df, requiere_objetivo=True)
    X = limpio[FEATURES_ACADEMICAS]
    y = limpio[COLUMNA_OBJETIVO]

    if len(limpio) < 20:
        raise ValueError("Se requieren al menos 20 registros para una división train/test básica.")
    if y.value_counts().min() < 4:
        raise ValueError("Se requieren al menos 4 casos por clase para una evaluación mínima.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    candidatos: dict[str, Pipeline] = {
        "regresion_logistica": Pipeline([
            ("escalador", StandardScaler()),
            ("modelo", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("modelo", RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }

    filas_metricas = []
    entrenados: dict[str, Pipeline] = {}
    detalles: dict[str, dict] = {}

    for nombre, pipeline in candidatos.items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metricas = {
            "modelo": nombre,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        filas_metricas.append(metricas)
        entrenados[nombre] = pipeline
        detalles[nombre] = {
            "matriz_confusion": confusion_matrix(y_test, pred).tolist(),
            "reporte_clasificacion": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }

    comparacion = pd.DataFrame(filas_metricas).sort_values(
        ["balanced_accuracy", "f1", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    mejor_nombre = str(comparacion.iloc[0]["modelo"])

    metadata = {
        "mejor_modelo": mejor_nombre,
        "features": FEATURES_ACADEMICAS,
        "objetivo": COLUMNA_OBJETIVO,
        "n_registros": int(len(limpio)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "prevalencia_abandono": float(y.mean()),
        "nota": (
            "El modelo usa solo asistencia y notas. El checklist se aplica después, "
            "como complemento del diagnóstico y no como predictor entrenado."
        ),
        "detalle_evaluacion": detalles,
    }
    return entrenados[mejor_nombre], comparacion, metadata


def guardar_modelo(
    modelo: Pipeline,
    metadata: dict,
    directorio: str | Path,
    config: ConfiguracionRiesgo = CONFIG_DEFAULT,
) -> None:
    ruta = Path(directorio)
    ruta.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta / "modelo_abandono_estudiantil.joblib")
    contenido = dict(metadata)
    contenido["configuracion_riesgo"] = asdict(config)
    (ruta / "metadata_modelo_abandono.json").write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cargar_modelo(ruta: str | Path) -> Pipeline:
    return joblib.load(ruta)


def factores_academicos_detectados(fila: pd.Series, config: ConfiguracionRiesgo = CONFIG_DEFAULT) -> str:
    """Describe las señales académicas que activan seguimiento."""
    factores = []
    asistencia = float(fila["porcentaje_asistencia"])
    nota = float(fila["promedio_notas"])
    if asistencia < config.umbral_asistencia:
        factores.append("Baja asistencia")
    if nota < config.nota_aprobacion:
        factores.append("Bajo rendimiento académico")
    return ", ".join(factores) if factores else "Sin alerta académica"


def diagnosticar_estudiantes(
    df: pd.DataFrame,
    modelo: Pipeline | None = None,
    config: ConfiguracionRiesgo = CONFIG_DEFAULT,
) -> pd.DataFrame:
    """Genera diagnóstico para uno o múltiples estudiantes.

    Si el checklist no está disponible, el resultado final utiliza el riesgo
    académico completo y marca el checklist como NO INFORMADO.
    """
    config.validar()
    df = normalizar_columnas_entrada(df)
    limpio = validar_dataset_estudiantes(df)
    salida = df.copy()

    if modelo is not None:
        riesgo_academico = pd.Series(
            modelo.predict_proba(limpio[FEATURES_ACADEMICAS])[:, 1],
            index=limpio.index,
            name="riesgo_academico",
        )
        metodo = "modelo_supervisado"
    else:
        riesgo_academico = puntaje_academico_heuristico(limpio, config)
        metodo = "reglas_heuristicas_sin_etiqueta_historica"

    riesgo_social = puntaje_checklist(salida)
    estado = estado_checklist(salida)
    peso_disponible = peso_checklist_disponible(salida)

    # El 30% se aplica proporcionalmente a la información social disponible.
    # Si no existe ningún dato del checklist, el riesgo final es 100% académico.
    peso_checklist_efectivo = (
        config.peso_checklist * peso_disponible / sum(PESOS_CHECKLIST.values())
    ).clip(0, config.peso_checklist)
    riesgo_social_informado = peso_disponible > 0
    peso_academico_efectivo = 1.0 - peso_checklist_efectivo

    score_final = (
        peso_academico_efectivo * riesgo_academico
        + peso_checklist_efectivo * riesgo_social.fillna(0)
    ).clip(0, 1)

    salida["riesgo_academico"] = riesgo_academico.round(4)
    salida["riesgo_checklist"] = riesgo_social.round(4)
    salida["estado_checklist"] = estado
    salida["peso_checklist_aplicado"] = peso_checklist_efectivo.round(3)
    salida["factores_academicos_detectados"] = salida.apply(
        lambda fila: factores_academicos_detectados(fila, config), axis=1
    )
    salida["score_riesgo_final"] = score_final.round(4)
    salida["nivel_riesgo"] = score_final.map(
        lambda x: clasificar_riesgo(float(x), config)
    )
    salida["factores_checklist_detectados"] = salida.apply(
        factores_detectados, axis=1
    )
    salida["metodo_diagnostico"] = metodo
    salida["requiere_revision"] = salida["nivel_riesgo"].isin(["MEDIO", "ALTO"])
    salida["prioridad_seguimiento"] = salida["nivel_riesgo"].map(
        {"ALTO": "ALTA", "MEDIO": "MEDIA", "BAJO": "BAJA"}
    )
    return salida


def diagnosticar_estudiante(
    df: pd.DataFrame | dict,
    modelo: Pipeline | None = None,
    config: ConfiguracionRiesgo = CONFIG_DEFAULT,
) -> pd.DataFrame:
    """Evalúa un estudiante individual o un DataFrame de estudiantes."""
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("diagnosticar_estudiante requiere un DataFrame o un diccionario.")
    return diagnosticar_estudiantes(df, modelo=modelo, config=config)

def columnas_esperadas() -> list[str]:
    return [
        "id_estudiante",
        "grado",
        "porcentaje_asistencia",
        "promedio_notas",
        *CRITERIOS_CHECKLIST,
        "abandono (solo histórico para entrenamiento; opcional)",
    ]
