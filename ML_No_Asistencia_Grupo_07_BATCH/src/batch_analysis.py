from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.riesgo_abandono import (
    COLUMNA_OBJETIVO,
    FEATURES_ACADEMICAS,
    CRITERIOS_CHECKLIST,
    diagnosticar_estudiantes,
    normalizar_columnas_entrada,
)


EXTENSIONES_VALIDAS = {".xlsx", ".xls", ".csv"}


def listar_archivos_entrada(root: Path) -> list[Path]:
    """Busca archivos tabulares colocados por el usuario en el proyecto.

    Se revisan:
    - data/entrada/
    - raíz del proyecto

    Se ignoran carpetas legacy, venv, cache y outputs.
    """
    candidatos: list[Path] = []
    carpetas = [root / "data" / "entrada", root]
    excluidas = {"legacy_censo", "venv", ".git", ".pytest_cache", "__pycache__", "outputs"}

    for carpeta in carpetas:
        if not carpeta.exists():
            continue
        for archivo in carpeta.iterdir():
            if not archivo.is_file():
                continue
            if archivo.suffix.lower() not in EXTENSIONES_VALIDAS:
                continue
            if archivo.name.startswith("~$"):
                continue
            if any(parte in excluidas for parte in archivo.parts):
                continue
            if archivo not in candidatos:
                candidatos.append(archivo)

    return sorted(candidatos, key=lambda p: p.stat().st_mtime, reverse=True)


def nombres_hojas(origen) -> list[str]:
    """Devuelve las hojas disponibles para un Excel."""
    if hasattr(origen, "name") and str(origen.name).lower().endswith(".csv"):
        return ["CSV"]

    if isinstance(origen, (str, Path)):
        sufijo = Path(origen).suffix.lower()
    else:
        sufijo = Path(getattr(origen, "name", "")).suffix.lower()

    if sufijo in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(origen)
        return excel.sheet_names

    return ["CSV"]


def leer_archivo_tabular(origen, hoja: str | int | None = None) -> pd.DataFrame:
    """Carga CSV/XLSX/XLS y normaliza encabezados."""
    nombre = getattr(origen, "name", str(origen))
    sufijo = Path(nombre).suffix.lower()

    if sufijo == ".csv":
        df = pd.read_csv(origen)
    elif sufijo in {".xlsx", ".xls"}:
        if hoja is None:
            hoja = 0
        df = pd.read_excel(origen, sheet_name=hoja)
    else:
        raise ValueError("Formato no soportado. Use CSV, XLSX o XLS.")

    df = df.dropna(how="all")
    df = normalizar_columnas_entrada(df)

    if df.empty:
        raise ValueError("El archivo no contiene registros.")

    faltantes = [c for c in FEATURES_ACADEMICAS if c not in df.columns]
    if faltantes:
        raise ValueError(
            "El archivo no contiene las columnas académicas requeridas: "
            + ", ".join(faltantes)
        )

    return df


def tiene_historico_entrenamiento(df: pd.DataFrame) -> bool:
    """Indica si el archivo contiene una etiqueta de abandono con ambas clases."""
    if COLUMNA_OBJETIVO not in df.columns:
        return False

    valores = df[COLUMNA_OBJETIVO].dropna().astype(str).str.strip().str.lower()
    positivos = {"1", "si", "sí", "s", "yes", "true", "verdadero"}
    negativos = {"0", "no", "n", "false", "falso"}
    normalizados = valores.map(
        lambda x: 1 if x in positivos else (0 if x in negativos else None)
    ).dropna()

    return normalizados.nunique() == 2


def resumen_diagnostico(resultado: pd.DataFrame) -> pd.DataFrame:
    total = len(resultado)
    alto = int((resultado["nivel_riesgo"] == "ALTO").sum())
    medio = int((resultado["nivel_riesgo"] == "MEDIO").sum())
    bajo = int((resultado["nivel_riesgo"] == "BAJO").sum())
    revision = int(resultado["requiere_revision"].sum())

    resumen = pd.DataFrame(
        [
            ["Total de estudiantes", total],
            ["Riesgo alto", alto],
            ["Riesgo medio", medio],
            ["Riesgo bajo", bajo],
            ["Requieren revisión", revision],
            ["Porcentaje con riesgo alto", round(alto / total * 100, 2) if total else 0],
            ["Porcentaje con riesgo medio o alto", round(revision / total * 100, 2) if total else 0],
        ],
        columns=["indicador", "valor"],
    )
    return resumen


def exportar_diagnostico_excel(resultado: pd.DataFrame, destino: str | Path | None = None) -> bytes | Path:
    """Genera un Excel con diagnóstico, resumen, casos prioritarios y metodología."""
    resumen = resumen_diagnostico(resultado)
    prioridad = resultado[
        resultado["nivel_riesgo"].isin(["ALTO", "MEDIO"])
    ].copy()

    metodologia = pd.DataFrame(
        [
            ["Variables académicas", "porcentaje_asistencia, promedio_notas"],
            ["Peso académico", "70% cuando existe checklist informado; 100% cuando no existe"],
            ["Checklist mayor peso", "Trabajo infantil, migración, embarazo adolescente, pobreza (peso 2)"],
            ["Checklist peso normal", "Vive lejos, bullying (peso 1)"],
            ["Riesgo medio", "0.40 a 0.6499"],
            ["Riesgo alto", ">= 0.65"],
            ["Nota metodológica", "La ausencia de información social no se interpreta como ausencia del factor."],
        ],
        columns=["elemento", "descripcion"],
    )

    if destino is None:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            resultado.to_excel(writer, sheet_name="Diagnostico_Alumnos", index=False)
            resumen.to_excel(writer, sheet_name="Resumen", index=False)
            prioridad.to_excel(writer, sheet_name="Casos_Prioritarios", index=False)
            metodologia.to_excel(writer, sheet_name="Metodologia", index=False)
        buffer.seek(0)
        return buffer.getvalue()

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(destino, engine="openpyxl") as writer:
        resultado.to_excel(writer, sheet_name="Diagnostico_Alumnos", index=False)
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        prioridad.to_excel(writer, sheet_name="Casos_Prioritarios", index=False)
        metodologia.to_excel(writer, sheet_name="Metodologia", index=False)
    return destino


def aplicar_checklist_desde_editor(
    resultado: pd.DataFrame,
    checklist_editado: pd.DataFrame,
    modelo=None,
) -> pd.DataFrame:
    """Convierte la tabla editable de Streamlit en un diagnóstico actualizado."""
    df = resultado.copy()
    for criterio in CRITERIOS_CHECKLIST:
        if criterio not in checklist_editado.columns:
            continue
        serie = checklist_editado[criterio].map(
            {"SI": True, "NO": False, "NO INFORMADO": pd.NA}
        )
        df[criterio] = serie

    return diagnosticar_estudiantes(df, modelo=modelo)
