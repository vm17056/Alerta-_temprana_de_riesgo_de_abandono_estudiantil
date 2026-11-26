from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.riesgo_abandono import (
    COLUMNA_OBJETIVO,
    FEATURES_ACADEMICAS,
    CRITERIOS_CHECKLIST,
    CRITERIOS_PESO_1,
    CRITERIOS_PESO_2,
    CRITERIOS_PESO_3,
    DESCRIPCIONES_CRITERIOS,
    PESOS_DIAGNOSTICO,
    diagnosticar_estudiantes,
    normalizar_columnas_entrada,
)
from src.medidas_preventivas import FACTOR_A_DOCUMENTO, FACTOR_A_CARPETA, FACTOR_A_NOMBRE, ruta_documento_factor


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
    insuficiente = int((resultado.get("estado_informacion", pd.Series(index=resultado.index)) == "INFORMACION_INSUFICIENTE").sum())
    completar = int(resultado.get("requiere_completar_informacion", pd.Series(False, index=resultado.index)).sum())

    resumen = pd.DataFrame(
        [
            ["Total de estudiantes", total],
            ["Riesgo alto", alto],
            ["Riesgo medio", medio],
            ["Riesgo bajo", bajo],
            ["Información insuficiente", insuficiente],
            ["Requieren revisión por riesgo", revision],
            ["Requieren completar información", completar],
            ["Porcentaje con riesgo alto", round(alto / total * 100, 2) if total else 0],
            ["Porcentaje con riesgo medio o alto", round(revision / total * 100, 2) if total else 0],
        ],
        columns=["indicador", "valor"],
    )
    return resumen


def tabla_criterios_diagnostico() -> pd.DataFrame:
    filas = []
    for criterio in [*CRITERIOS_PESO_1, *CRITERIOS_PESO_2, *CRITERIOS_PESO_3]:
        if criterio in CRITERIOS_PESO_1:
            peso = 1
        elif criterio in CRITERIOS_PESO_2:
            peso = 2
        else:
            peso = 3
        filas.append({
            "variable": criterio,
            "descripcion": DESCRIPCIONES_CRITERIOS.get(criterio, criterio.replace("_", " ").title()),
            "peso": peso,
            "porcentaje_diagnostico": round(PESOS_DIAGNOSTICO[criterio] * 100, 2),
        })
    filas.append({
        "variable": "TOTAL",
        "descripcion": "Total del diagnóstico",
        "peso": None,
        "porcentaje_diagnostico": 100.0,
    })
    return pd.DataFrame(filas)


def medidas_sugeridas(resultado: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Genera un inventario de documentos sugeridos por estudiante y factor detectado."""
    filas = []
    from src.medidas_preventivas import factores_para_estudiante
    for idx, fila in resultado.iterrows():
        if str(fila.get("nivel_riesgo", "BAJO")) not in {"MEDIO", "ALTO"}:
            continue
        for factor in factores_para_estudiante(fila):
            ruta = ruta_documento_factor(root, factor)
            filas.append({
                "id_estudiante": fila.get("id_estudiante", idx),
                "grado": fila.get("grado", ""),
                "nivel_riesgo": fila.get("nivel_riesgo", ""),
                "factor_detectado": factor,
                "factor": FACTOR_A_NOMBRE.get(factor, factor),
                "categoria": FACTOR_A_CARPETA.get(factor, ""),
                "documento": FACTOR_A_DOCUMENTO.get(factor, ""),
                "documento_disponible": ruta.exists(),
                "ruta_relativa": str(ruta.relative_to(Path(root))) if ruta.exists() else "",
            })
    return pd.DataFrame(filas)


def exportar_diagnostico_excel(
    resultado: pd.DataFrame,
    destino: str | Path | None = None,
    root: str | Path | None = None,
) -> bytes | Path:
    """Genera un Excel con diagnóstico, resumen, criterios y medidas sugeridas."""
    resumen = resumen_diagnostico(resultado)
    prioridad = resultado[resultado["nivel_riesgo"].isin(["ALTO", "MEDIO"])].copy()
    criterios = tabla_criterios_diagnostico()
    medidas = medidas_sugeridas(resultado, root or Path.cwd())

    metodologia = pd.DataFrame(
        [
            ["Variables de peso 1", "porcentaje_asistencia y promedio_notas", "20% total (10% cada una)"],
            ["Variables de peso 2", ", ".join(CRITERIOS_PESO_2), "50% total (7.14% cada una aprox.)"],
            ["Variables de peso 3", ", ".join(CRITERIOS_PESO_3), "30% total (6% cada una)"],
            ["Riesgo medio", "score_riesgo_final >= 0.40 y < 0.65", "Requiere revisión"],
            ["Riesgo alto", "score_riesgo_final >= 0.65", "Requiere revisión prioritaria"],
            ["Campos vacíos", "Se consideran información insuficiente y se identifican en campos_sin_informacion", "Nunca se interpretan como NO"],
            ["Checklist incompleto", "El riesgo puede calcularse con la información disponible, pero el registro queda marcado como INSUFICIENTE", "Completar el checklist antes de una decisión institucional"],
            ["Medidas preventivas", "Se seleccionan según factores detectados y archivos de medidas_preventivas/", "Los DOCX pueden reemplazarse sin cambiar código"],
        ],
        columns=["elemento", "descripcion", "regla"],
    )

    def escribir(writer):
        resultado.to_excel(writer, sheet_name="Diagnostico_Alumnos", index=False)
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        prioridad.to_excel(writer, sheet_name="Casos_Prioritarios", index=False)
        criterios.to_excel(writer, sheet_name="Criterios_Diagnostico", index=False)
        medidas.to_excel(writer, sheet_name="Medidas_Sugeridas", index=False)
        metodologia.to_excel(writer, sheet_name="Metodologia", index=False)

    if destino is None:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            escribir(writer)
        buffer.seek(0)
        return buffer.getvalue()

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(destino, engine="openpyxl") as writer:
        escribir(writer)
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
