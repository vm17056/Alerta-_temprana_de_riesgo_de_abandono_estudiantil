from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

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
COLUMNA_FECHA_CORTE = "fecha_corte"

ALIASES_COLUMNAS = {
    "id_estudiante": {
        "id_estudiante", "id estudiante", "id", "codigo", "código",
        "codigo_estudiante", "código_estudiante", "student_id", "estudiante"
    },
    "grado": {"grado", "nivel", "curso", "grado_nivel", "nivel_educativo"},
    "fecha_corte": {"fecha_corte", "fecha corte", "fecha", "periodo", "periodo_corte"},
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
    "pobreza": {"pobreza", "situacion pobreza", "situación pobreza", "estado de pobreza"},
    "bullying": {"bullying", "bullying o acoso escolar", "acoso escolar", "acoso"},
    "repitencia_escolar": {"repitencia_escolar", "repitencia escolar", "repitencia"},
    "dificultades_aprendizaje": {"dificultades_aprendizaje", "dificultades de aprendizaje", "dificultades aprendizaje"},
    "antecedentes_abandono_escolar": {"antecedentes_abandono_escolar", "antecedentes abandono escolar", "antecedentes de abandono escolar"},
    "violencia_intrafamiliar": {"violencia_intrafamiliar", "violencia intrafamiliar"},
    "falta_transporte_accesible": {"falta_transporte_accesible", "falta transporte accesible", "falta de transporte accesible"},
    "distancia_vivienda_centro_educativo": {
        "distancia_vivienda_centro_educativo", "distancia entre la vivienda y el centro educativo",
        "distancia vivienda centro educativo", "vive_lejos", "vive lejos", "distancia",
        "vive lejos del centro educativo"
    },
    "limitaciones_conectividad_recursos_tecnologicos": {
        "limitaciones_conectividad_recursos_tecnologicos", "limitaciones de conectividad y acceso a recursos tecnológicos",
        "limitaciones conectividad", "conectividad", "acceso tecnologia", "acceso tecnología"
    },
    "abandono": {"abandono", "desercion", "deserción", "abandono escolar", "dropout"},
}

CRITERIOS_PESO_1 = FEATURES_ACADEMICAS.copy()
CRITERIOS_PESO_2 = [
    "trabajo_infantil", "migracion", "embarazo_adolescente", "pobreza",
    "dificultades_aprendizaje", "antecedentes_abandono_escolar", "violencia_intrafamiliar",
]
CRITERIOS_PESO_3 = [
    "bullying", "repitencia_escolar", "falta_transporte_accesible",
    "distancia_vivienda_centro_educativo", "limitaciones_conectividad_recursos_tecnologicos",
]
CRITERIOS_CHECKLIST = CRITERIOS_PESO_2 + CRITERIOS_PESO_3
CRITERIOS_DIAGNOSTICO = CRITERIOS_PESO_1 + CRITERIOS_CHECKLIST

PESOS_DIAGNOSTICO = {
    "porcentaje_asistencia": 0.10,
    "promedio_notas": 0.10,
    **{criterio: 0.50 / len(CRITERIOS_PESO_2) for criterio in CRITERIOS_PESO_2},
    **{criterio: 0.30 / len(CRITERIOS_PESO_3) for criterio in CRITERIOS_PESO_3},
}

DESCRIPCIONES_CRITERIOS = {
    "porcentaje_asistencia": "Porcentaje de asistencia",
    "promedio_notas": "Promedio de notas",
    "trabajo_infantil": "Trabajo infantil",
    "migracion": "Migración",
    "embarazo_adolescente": "Embarazo adolescente",
    "pobreza": "Estado de pobreza",
    "bullying": "Bullying o acoso escolar",
    "repitencia_escolar": "Repitencia escolar",
    "dificultades_aprendizaje": "Dificultades de aprendizaje",
    "antecedentes_abandono_escolar": "Antecedentes de abandono escolar",
    "violencia_intrafamiliar": "Violencia intrafamiliar",
    "falta_transporte_accesible": "Falta de transporte accesible",
    "distancia_vivienda_centro_educativo": "Distancia entre la vivienda y el centro educativo",
    "limitaciones_conectividad_recursos_tecnologicos": "Limitaciones de conectividad y acceso a recursos tecnológicos",
}

@dataclass(frozen=True)
class ConfiguracionRiesgo:
    umbral_asistencia: float = 85.0
    asistencia_critica: float = 70.0
    nota_aprobacion: float = 6.0
    nota_critica: float = 4.0
    peso_academico: float = 0.20
    peso_checklist: float = 0.80
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


def _clave_columna(valor) -> str:
    return " ".join(str(valor).strip().lower().replace("-", " ").replace("_", " ").split())


def normalizar_columnas_entrada(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres y convierte celdas vacías del checklist en NA."""
    salida = df.copy()
    mapa = {}
    for destino, aliases in ALIASES_COLUMNAS.items():
        for alias in aliases:
            mapa[_clave_columna(alias)] = destino

    renombres = {}
    for columna in salida.columns:
        clave = _clave_columna(columna)
        destino = mapa.get(clave)
        if destino and destino not in salida.columns:
            renombres[columna] = destino
    salida = salida.rename(columns=renombres)

    for criterio in CRITERIOS_CHECKLIST:
        if criterio in salida.columns:
            def limpiar(valor):
                if pd.isna(valor):
                    return pd.NA
                if isinstance(valor, str):
                    texto = valor.strip()
                    if texto == "" or texto.lower() in {"no informado", "no informado/a", "n/a", "na", "null", "none"}:
                        return pd.NA
                    return texto
                return valor
            salida[criterio] = salida[criterio].map(limpiar)

    if COLUMNA_FECHA_CORTE in salida.columns:
        salida[COLUMNA_FECHA_CORTE] = pd.to_datetime(salida[COLUMNA_FECHA_CORTE], errors="coerce")

    return salida


def _normalizar_booleano(valor) -> int:
    if pd.isna(valor):
        return 0
    if isinstance(valor, (bool, np.bool_)):
        return int(valor)
    if isinstance(valor, (int, np.integer, float, np.floating)):
        return int(float(valor) > 0)
    texto = str(valor).strip().lower()
    verdaderos = {"1", "si", "sí", "s", "yes", "y", "true", "verdadero", "x"}
    falsos = {"0", "no", "n", "false", "falso"}
    if texto in verdaderos:
        return 1
    if texto in falsos:
        return 0
    raise ValueError(f"Valor de checklist no reconocido: {valor!r}")


def _normalizar_objetivo(serie: pd.Series) -> pd.Series:
    if serie.isna().any():
        raise ValueError("La columna 'abandono' contiene valores vacíos.")
    valores = serie.map(_normalizar_booleano).astype("int8")
    if valores.nunique() < 2:
        raise ValueError("La columna 'abandono' debe contener casos de abandono y no abandono.")
    return valores


def validar_dataset_estudiantes(
    df: pd.DataFrame,
    requiere_objetivo: bool = False,
    permitir_academicos_faltantes: bool = False,
) -> pd.DataFrame:
    df = normalizar_columnas_entrada(df)
    faltantes = [c for c in FEATURES_ACADEMICAS if c not in df.columns]
    if faltantes:
        raise ValueError("Faltan columnas académicas obligatorias: " + ", ".join(faltantes))
    if requiere_objetivo and COLUMNA_OBJETIVO not in df.columns:
        raise ValueError("Para entrenar se requiere la columna histórica 'abandono'.")

    limpio = df.copy()
    for columna in FEATURES_ACADEMICAS:
        limpio[columna] = pd.to_numeric(limpio[columna], errors="coerce")

    faltantes_academicos = limpio[FEATURES_ACADEMICAS].isna().any(axis=1)
    if faltantes_academicos.any() and not permitir_academicos_faltantes:
        filas = limpio.index[faltantes_academicos].tolist()[:10]
        raise ValueError(f"Hay valores académicos vacíos/no numéricos. Filas: {filas}")

    asistencia_valida = limpio["porcentaje_asistencia"].dropna().between(0, 100)
    if not asistencia_valida.all():
        raise ValueError("porcentaje_asistencia debe estar entre 0 y 100.")

    notas_validas = limpio["promedio_notas"].dropna()
    if not notas_validas.empty:
        if notas_validas.min() < 0 or notas_validas.max() > 100:
            raise ValueError("promedio_notas debe estar en una escala 0-10 o 0-100.")
        if notas_validas.max() > 10:
            limpio["promedio_notas"] = limpio["promedio_notas"] / 10.0
            limpio.attrs["nota_convertida_de_100_a_10"] = True

    if requiere_objetivo:
        limpio[COLUMNA_OBJETIVO] = _normalizar_objetivo(limpio[COLUMNA_OBJETIVO])

    return limpio


def puntaje_academico_heuristico(df: pd.DataFrame, config: ConfiguracionRiesgo = CONFIG_DEFAULT) -> pd.Series:
    config.validar()
    limpio = validar_dataset_estudiantes(df, permitir_academicos_faltantes=True)
    asistencia = limpio["porcentaje_asistencia"].astype(float)
    notas = limpio["promedio_notas"].astype(float)
    riesgo_asistencia = ((config.umbral_asistencia - asistencia) / (config.umbral_asistencia - config.asistencia_critica)).clip(0, 1)
    riesgo_notas = ((config.nota_aprobacion - notas) / (config.nota_aprobacion - config.nota_critica)).clip(0, 1)
    return (0.5 * riesgo_asistencia + 0.5 * riesgo_notas).rename("riesgo_academico")


def puntaje_checklist(df: pd.DataFrame) -> pd.Series:
    salida = df.copy()
    acumulado = pd.Series(0.0, index=salida.index)
    for criterio in CRITERIOS_CHECKLIST:
        if criterio not in salida.columns:
            continue
        informado = salida[criterio].notna()
        positivos = salida[criterio].map(_normalizar_booleano)
        acumulado = acumulado + positivos * PESOS_DIAGNOSTICO[criterio] * informado.astype(float)
    return acumulado.rename("riesgo_checklist_ponderado")


def peso_checklist_disponible(df: pd.DataFrame) -> pd.Series:
    peso = pd.Series(0.0, index=df.index)
    for criterio in CRITERIOS_CHECKLIST:
        if criterio in df.columns:
            peso = peso + df[criterio].notna().astype(float) * PESOS_DIAGNOSTICO[criterio]
    return peso.rename("peso_checklist_disponible")


def cobertura_diagnostico(df: pd.DataFrame) -> pd.Series:
    cobertura = pd.Series(0.0, index=df.index)
    for criterio in FEATURES_ACADEMICAS:
        if criterio in df.columns:
            cobertura = cobertura + df[criterio].notna().astype(float) * PESOS_DIAGNOSTICO[criterio]
    cobertura = cobertura + peso_checklist_disponible(df)
    return cobertura.clip(0, 1).rename("cobertura_diagnostico")


def estado_checklist(df: pd.DataFrame) -> pd.Series:
    columnas = [c for c in CRITERIOS_CHECKLIST if c in df.columns]
    if not columnas:
        return pd.Series("INFORMACION_INSUFICIENTE", index=df.index, name="estado_checklist")
    informados = df[columnas].notna().sum(axis=1)
    total = len(CRITERIOS_CHECKLIST)
    return informados.map(
        lambda n: "COMPLETO" if n == total else "INFORMACION_INSUFICIENTE"
    ).rename("estado_checklist")


def campos_sin_informacion(df: pd.DataFrame) -> pd.Series:
    campos = FEATURES_ACADEMICAS + CRITERIOS_CHECKLIST
    disponibles = [c for c in campos if c in df.columns]
    faltantes_por_columna = [c for c in campos if c not in df.columns]
    valores = []
    for idx in df.index:
        faltantes = list(faltantes_por_columna)
        for c in disponibles:
            if pd.isna(df.loc[idx, c]):
                faltantes.append(c)
        valores.append(", ".join(faltantes))
    return pd.Series(valores, index=df.index, name="campos_sin_informacion")


def estado_informacion(df: pd.DataFrame) -> pd.Series:
    faltantes = campos_sin_informacion(df)
    return faltantes.map(lambda x: "COMPLETA" if not x else "INFORMACION_INSUFICIENTE").rename("estado_informacion")


def clasificar_riesgo(score: float, config: ConfiguracionRiesgo = CONFIG_DEFAULT) -> str:
    if pd.isna(score):
        return "INFORMACION_INSUFICIENTE"
    if score >= config.corte_riesgo_alto:
        return "ALTO"
    if score >= config.corte_riesgo_medio:
        return "MEDIO"
    return "BAJO"


def factores_detectados(fila: pd.Series) -> str:
    encontrados: list[str] = []
    for criterio in CRITERIOS_CHECKLIST:
        if criterio in fila.index and not pd.isna(fila[criterio]) and _normalizar_booleano(fila[criterio]):
            encontrados.append(criterio)
    return ", ".join(encontrados) if encontrados else "Ninguno reportado"


def factores_academicos_detectados(fila: pd.Series, config: ConfiguracionRiesgo = CONFIG_DEFAULT) -> str:
    factores = []
    asistencia = fila.get("porcentaje_asistencia")
    nota = fila.get("promedio_notas")
    if pd.notna(asistencia) and float(asistencia) < config.umbral_asistencia:
        factores.append("Baja asistencia")
    if pd.notna(nota) and float(nota) < config.nota_aprobacion:
        factores.append("Bajo rendimiento académico")
    return ", ".join(factores) if factores else "Sin alerta académica"


def entrenar_modelo(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.25,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Entrenamiento básico conservado para compatibilidad. El flujo oficial usa advanced_training.py."""
    from src.advanced_training import train_models
    return train_models(df, random_state=random_state, test_size=test_size)


def guardar_modelo(modelo: Pipeline, metadata: dict, directorio: str | Path, config: ConfiguracionRiesgo = CONFIG_DEFAULT) -> None:
    ruta = Path(directorio)
    ruta.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta / "modelo_abandono_estudiantil.joblib")
    contenido = dict(metadata)
    contenido["configuracion_riesgo"] = asdict(config)
    (ruta / "metadata_modelo_abandono.json").write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")


def cargar_modelo(ruta: str | Path) -> Pipeline:
    return joblib.load(ruta)


def diagnosticar_estudiantes(
    df: pd.DataFrame,
    modelo: Pipeline | None = None,
    config: ConfiguracionRiesgo = CONFIG_DEFAULT,
) -> pd.DataFrame:
    """Diagnostica el salón completo.

    Los campos vacíos se consideran información insuficiente. Los datos académicos
    son indispensables para producir un nivel de riesgo; si falta asistencia o nota,
    el estudiante queda en INFORMACION_INSUFICIENTE y no se inventa un resultado.
    Los campos sociales vacíos no se consideran "NO": el riesgo puede calcularse con
    los criterios disponibles, pero el registro queda marcado como incompleto.
    """
    config.validar()
    df = normalizar_columnas_entrada(df)
    limpio = validar_dataset_estudiantes(df, permitir_academicos_faltantes=True)
    salida = df.copy()

    academicos_completos = limpio[FEATURES_ACADEMICAS].notna().all(axis=1)

    riesgo_modelo = pd.Series(np.nan, index=limpio.index, dtype=float)
    if modelo is not None and academicos_completos.any():
        riesgo_modelo.loc[academicos_completos] = modelo.predict_proba(
            limpio.loc[academicos_completos, FEATURES_ACADEMICAS]
        )[:, 1]
        metodo = "modelo_supervisado_calibrado" if getattr(modelo, "_modelo_calibrado", False) else "modelo_supervisado"
    else:
        riesgo_modelo.loc[academicos_completos] = puntaje_academico_heuristico(limpio.loc[academicos_completos], config)
        metodo = "reglas_heuristicas_sin_etiqueta_historica"

    asistencia = limpio["porcentaje_asistencia"].astype(float)
    notas = limpio["promedio_notas"].astype(float)
    riesgo_asistencia = ((config.umbral_asistencia - asistencia) / (config.umbral_asistencia - config.asistencia_critica)).clip(0, 1)
    riesgo_notas = ((config.nota_aprobacion - notas) / (config.nota_aprobacion - config.nota_critica)).clip(0, 1)

    riesgo_checklist = puntaje_checklist(salida)
    peso_social = peso_checklist_disponible(salida)
    cobertura = cobertura_diagnostico(limpio)

    score_ponderado = (
        PESOS_DIAGNOSTICO["porcentaje_asistencia"] * riesgo_asistencia.fillna(0)
        + PESOS_DIAGNOSTICO["promedio_notas"] * riesgo_notas.fillna(0)
        + riesgo_checklist
    ).clip(0, 1)
    score_final = (score_ponderado / cobertura.replace(0, np.nan)).clip(0, 1)
    score_final.loc[~academicos_completos] = np.nan

    salida["riesgo_academico"] = riesgo_modelo.round(4)
    salida["riesgo_asistencia"] = riesgo_asistencia.round(4)
    salida["riesgo_notas"] = riesgo_notas.round(4)
    salida["aporte_asistencia"] = (PESOS_DIAGNOSTICO["porcentaje_asistencia"] * riesgo_asistencia).round(4)
    salida["aporte_promedio_notas"] = (PESOS_DIAGNOSTICO["promedio_notas"] * riesgo_notas).round(4)
    for criterio in CRITERIOS_CHECKLIST:
        if criterio in salida.columns:
            salida[f"aporte_{criterio}"] = (
                salida[criterio].map(_normalizar_booleano)
                * PESOS_DIAGNOSTICO[criterio]
                * salida[criterio].notna().astype(float)
            ).round(4)
        else:
            salida[f"aporte_{criterio}"] = 0.0

    salida["riesgo_checklist"] = riesgo_checklist.round(4)
    salida["peso_checklist_aplicado"] = peso_social.round(4)
    salida["cobertura_diagnostico"] = cobertura.round(4)
    salida["estado_checklist"] = estado_checklist(salida)
    salida["campos_sin_informacion"] = campos_sin_informacion(limpio)
    salida["estado_informacion"] = estado_informacion(limpio)
    salida["factores_academicos_detectados"] = salida.apply(lambda fila: factores_academicos_detectados(fila, config), axis=1)
    salida["score_riesgo_ponderado"] = score_ponderado.round(4)
    salida["score_riesgo_final"] = score_final.round(4)
    salida["porcentaje_riesgo_final"] = (score_final * 100).round(2)
    salida["nivel_riesgo"] = score_final.map(lambda x: clasificar_riesgo(x, config))
    salida["factores_checklist_detectados"] = salida.apply(factores_detectados, axis=1)
    salida["metodo_diagnostico"] = metodo
    salida["requiere_revision"] = salida["nivel_riesgo"].isin(["MEDIO", "ALTO"])
    salida["requiere_completar_informacion"] = salida["estado_informacion"].eq("INFORMACION_INSUFICIENTE")
    salida["prioridad_seguimiento"] = salida["nivel_riesgo"].map({"ALTO": "ALTA", "MEDIO": "MEDIA", "BAJO": "BAJA", "INFORMACION_INSUFICIENTE": "PENDIENTE DE DATOS"})
    return salida


def diagnosticar_estudiante(df: pd.DataFrame | dict, modelo: Pipeline | None = None, config: ConfiguracionRiesgo = CONFIG_DEFAULT) -> pd.DataFrame:
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("diagnosticar_estudiante requiere un DataFrame o un diccionario.")
    return diagnosticar_estudiantes(df, modelo=modelo, config=config)


def columnas_esperadas() -> list[str]:
    return ["id_estudiante", "grado", "fecha_corte", *CRITERIOS_DIAGNOSTICO, "abandono (solo histórico para entrenamiento; opcional)"]
