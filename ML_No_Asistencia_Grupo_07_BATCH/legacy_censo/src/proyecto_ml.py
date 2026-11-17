from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

COLUMNAS_ORIGINALES = [
    "departamento", "municipio", "distrito", "grupo_etario",
    "ha_asistido_hombres", "ha_asistido_mujeres", "ha_asistido_total",
    "nunca_asistio_hombres", "nunca_asistio_mujeres", "nunca_asistio_total",
    "no_sabe_hombres", "no_sabe_mujeres", "no_sabe_total", "poblacion_total",
]
COLUMNAS_CONTEO = COLUMNAS_ORIGINALES[4:]
GRUPOS_ETARIOS_ALCANCE = ["10 a 14 años", "15 a 19 años"]

FEATURES_MODELO = [
    "departamento", "municipio", "nivel_educativo_aproximado",
]
FEATURES_CATEGORICAS = [
    "departamento", "municipio", "nivel_educativo_aproximado",
]
FEATURES_NUMERICAS: list[str] = []
COLUMNAS_PROHIBIDAS_MODELO = {
    "ha_asistido_hombres", "ha_asistido_mujeres", "ha_asistido_total",
    "nunca_asistio_hombres", "nunca_asistio_mujeres", "nunca_asistio_total",
    "no_sabe_hombres", "no_sabe_mujeres", "no_sabe_total", "poblacion_total",
    "tasa_nunca_asistio", "alta_no_asistencia_6to_bachillerato",
}


def localizar_raiz(inicio: str | Path | None = None) -> Path:
    inicio_path = Path(inicio or Path.cwd()).resolve()
    for candidato in [inicio_path, *inicio_path.parents]:
        if (candidato / "data" / "raw" / "TAB_EDU_1.xlsx").exists():
            return candidato
    raise FileNotFoundError(
        "No se encontró data/raw/TAB_EDU_1.xlsx. Ejecute desde la raíz o notebooks/."
    )


def cargar_tabla_original(ruta_excel: str | Path) -> pd.DataFrame:
    df = pd.read_excel(
        ruta_excel, sheet_name="Table 1", skiprows=5, names=COLUMNAS_ORIGINALES
    )
    es_fila_tabla = df["departamento"].astype(str).str.match(r"^\d{2}\s*-", na=False)
    df = df.loc[es_fila_tabla].copy()
    for columna in COLUMNAS_CONTEO:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    return df.reset_index(drop=True)


def filtrar_nivel_distrito(df: pd.DataFrame) -> pd.DataFrame:
    mascara = (
        df["departamento"].ne("00 - República de El Salvador")
        & df["municipio"].ne("TOTAL")
        & df["distrito"].ne("TOTAL")
        & df["grupo_etario"].ne("TOTAL")
    )
    return df.loc[mascara].copy().reset_index(drop=True)


def filtrar_sexto_bachillerato(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva únicamente los grupos de 10–14 y 15–19 años.

    Es una aproximación al rango de sexto grado a bachillerato porque la fuente
    no contiene grado cursado ni edades simples; usa grupos quinquenales.
    """
    return df.loc[df["grupo_etario"].isin(GRUPOS_ETARIOS_ALCANCE)].copy().reset_index(drop=True)


def verificar_blancos_como_cero(df: pd.DataFrame) -> dict[str, float]:
    prueba = df.copy()
    prueba[COLUMNAS_CONTEO] = prueba[COLUMNAS_CONTEO].fillna(0)
    return {
        "error_max_ha_asistido": float((prueba["ha_asistido_hombres"] + prueba["ha_asistido_mujeres"] - prueba["ha_asistido_total"]).abs().max()),
        "error_max_nunca_asistio": float((prueba["nunca_asistio_hombres"] + prueba["nunca_asistio_mujeres"] - prueba["nunca_asistio_total"]).abs().max()),
        "error_max_no_sabe": float((prueba["no_sabe_hombres"] + prueba["no_sabe_mujeres"] - prueba["no_sabe_total"]).abs().max()),
        "error_max_poblacion": float((prueba["ha_asistido_total"] + prueba["nunca_asistio_total"] + prueba["no_sabe_total"] - prueba["poblacion_total"]).abs().max()),
    }


def _extraer_edades(texto: str) -> tuple[int, int, float]:
    numeros = [int(x) for x in re.findall(r"\d+", str(texto))]
    if len(numeros) >= 2:
        minimo, maximo = numeros[0], numeros[1]
    elif numeros:
        minimo = maximo = numeros[0]
    else:
        minimo = maximo = 0
    return minimo, maximo, (minimo + maximo) / 2


def preparar_dataset_analitico(df_alcance: pd.DataFrame) -> pd.DataFrame:
    df = df_alcance.copy()
    # Los blancos en columnas de conteo son tratados como cero después de validar
    # las identidades hombres + mujeres = total y categorías = población.
    df[COLUMNAS_CONTEO] = df[COLUMNAS_CONTEO].fillna(0).astype("int64")

    edades = df["grupo_etario"].apply(_extraer_edades)
    df[["edad_min", "edad_max", "edad_media"]] = pd.DataFrame(edades.tolist(), index=df.index)
    df["nivel_educativo_aproximado"] = np.where(
        df["grupo_etario"].eq("10 a 14 años"),
        "Sexto a noveno grado (aprox.)",
        "Bachillerato (aprox.)",
    )
    df["es_bachillerato_aproximado"] = df["grupo_etario"].eq("15 a 19 años").astype("int8")
    df["codigo_departamento"] = df["departamento"].str.extract(r"^(\d{2})")[0]
    df["grupo_geografico"] = (
        df["departamento"].astype(str) + " | " + df["municipio"].astype(str)
        + " | " + df["distrito"].astype(str)
    )
    denominador = df["poblacion_total"].replace(0, np.nan)
    df["tasa_nunca_asistio"] = (df["nunca_asistio_total"] / denominador).fillna(0.0)
    return df


def crear_preprocesador() -> ColumnTransformer:
    categorico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("categoricas", categorico, FEATURES_CATEGORICAS),
    ], remainder="drop")


def validar_ausencia_fuga(columnas: Iterable[str]) -> None:
    filtradas = set(columnas).intersection(COLUMNAS_PROHIBIDAS_MODELO)
    if filtradas:
        raise ValueError(f"Variables con fuga de información: {sorted(filtradas)}")
