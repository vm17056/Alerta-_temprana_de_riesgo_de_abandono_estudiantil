from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from docx import Document


ROOT_FOLDER_NAME = "medidas_preventivas"

FACTOR_A_DOCUMENTO = {
    "porcentaje_asistencia": "bajo_porcentaje_asistencia.docx",
    "promedio_notas": "bajo_promedio_notas.docx",
    "trabajo_infantil": "trabajo_infantil.docx",
    "migracion": "migracion.docx",
    "embarazo_adolescente": "embarazo_adolescente.docx",
    "pobreza": "pobreza.docx",
    "bullying": "bullying_acoso_escolar.docx",
    "repitencia_escolar": "repitencia_escolar.docx",
    "dificultades_aprendizaje": "dificultades_aprendizaje.docx",
    "antecedentes_abandono_escolar": "antecedentes_abandono_escolar.docx",
    "violencia_intrafamiliar": "violencia_intrafamiliar.docx",
    "falta_transporte_accesible": "falta_transporte_accesible.docx",
    "distancia_vivienda_centro_educativo": "distancia_vivienda_centro_educativo.docx",
    "limitaciones_conectividad_recursos_tecnologicos": "limitaciones_conectividad_recursos_tecnologicos.docx",
}

FACTOR_A_CARPETA = {
    "porcentaje_asistencia": "factores_academicos",
    "promedio_notas": "factores_academicos",
    "repitencia_escolar": "factores_academicos",
    "dificultades_aprendizaje": "factores_academicos",
    "antecedentes_abandono_escolar": "factores_academicos",
    "trabajo_infantil": "factores_economicos_sociales",
    "migracion": "factores_economicos_sociales",
    "embarazo_adolescente": "factores_economicos_sociales",
    "pobreza": "factores_economicos_sociales",
    "bullying": "factores_violencia_proteccion",
    "violencia_intrafamiliar": "factores_violencia_proteccion",
    "falta_transporte_accesible": "factores_geograficos_tecnologicos",
    "distancia_vivienda_centro_educativo": "factores_geograficos_tecnologicos",
    "limitaciones_conectividad_recursos_tecnologicos": "factores_geograficos_tecnologicos",
}

FACTOR_A_NOMBRE = {
    "porcentaje_asistencia": "Bajo porcentaje de asistencia",
    "promedio_notas": "Bajo promedio de notas",
    "trabajo_infantil": "Trabajo infantil",
    "migracion": "Migración",
    "embarazo_adolescente": "Embarazo adolescente",
    "pobreza": "Pobreza",
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
class MedidaPreventiva:
    factor: str
    nombre: str
    categoria: str
    ruta: Path


def ruta_raiz_medidas(root: str | Path) -> Path:
    return Path(root) / ROOT_FOLDER_NAME


def ruta_documento_factor(root: str | Path, factor: str) -> Path:
    carpeta = FACTOR_A_CARPETA[factor]
    nombre = FACTOR_A_DOCUMENTO[factor]
    return ruta_raiz_medidas(root) / carpeta / nombre


def listar_medidas_disponibles(root: str | Path) -> list[MedidaPreventiva]:
    resultado = []
    for factor, nombre in FACTOR_A_DOCUMENTO.items():
        ruta = ruta_documento_factor(root, factor)
        if ruta.exists():
            resultado.append(
                MedidaPreventiva(
                    factor=factor,
                    nombre=FACTOR_A_NOMBRE[factor],
                    categoria=FACTOR_A_CARPETA[factor],
                    ruta=ruta,
                )
            )
    return resultado


def factores_para_estudiante(fila) -> list[str]:
    factores: list[str] = []
    asistencia = float(fila.get("porcentaje_asistencia", 100))
    nota = float(fila.get("promedio_notas", 10))
    if asistencia < 85:
        factores.append("porcentaje_asistencia")
    if nota < 6:
        factores.append("promedio_notas")

    for factor in FACTOR_A_DOCUMENTO:
        if factor in {"porcentaje_asistencia", "promedio_notas"}:
            continue
        if factor in fila.index:
            valor = fila[factor]
            if str(valor).strip().lower() in {"si", "sí", "true", "1", "1.0", "x", "yes"} or valor is True:
                factores.append(factor)
    return factores


def leer_documento(ruta: str | Path) -> str:
    """Extrae texto del DOCX para mostrar una vista previa en Streamlit."""
    doc = Document(ruta)
    partes: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            partes.append(para.text.strip())
    for tabla in doc.tables:
        for fila in tabla.rows:
            partes.append(" | ".join(celda.text.strip().replace("\n", " / ") for celda in fila.cells))
    return "\n\n".join(partes)


def medidas_para_factores(root: str | Path, factores: Iterable[str]) -> list[MedidaPreventiva]:
    resultado = []
    for factor in factores:
        if factor not in FACTOR_A_DOCUMENTO:
            continue
        ruta = ruta_documento_factor(root, factor)
        if ruta.exists():
            resultado.append(
                MedidaPreventiva(
                    factor=factor,
                    nombre=FACTOR_A_NOMBRE[factor],
                    categoria=FACTOR_A_CARPETA[factor],
                    ruta=ruta,
                )
            )
    return resultado


def documento_combinado_medidas(
    root: str | Path,
    factores: Iterable[str],
    identificador: str = "Estudiante",
    nivel_riesgo: str = "",
) -> bytes:
    """Genera un DOCX editable con todas las medidas detectadas para un estudiante.

    El documento se reconstruye a partir del texto de los DOCX configurados en
    `medidas_preventivas/`, por lo que no conserva hipervínculos externos de las
    fuentes. Los archivos individuales siguen siendo editables y reemplazables.
    """
    from io import BytesIO

    documento = Document()
    documento.add_heading("Medidas preventivas y de mitigación", level=0)
    documento.add_paragraph(f"Estudiante: {identificador}")
    if nivel_riesgo:
        documento.add_paragraph(f"Nivel de riesgo: {nivel_riesgo}")
    documento.add_paragraph(
        "Este documento reúne las medidas asociadas a los factores detectados por el sistema. "
        "La aplicación de las medidas debe ser validada por el personal educativo responsable."
    )

    medidas = medidas_para_factores(root, factores)
    for indice, medida in enumerate(medidas, start=1):
        documento.add_heading(f"{indice}. {medida.nombre}", level=1)
        documento.add_paragraph(f"Categoría: {medida.categoria.replace('_', ' ').title()}")
        texto = leer_documento(medida.ruta)
        if texto:
            for bloque in texto.split("\n\n"):
                if bloque.strip():
                    documento.add_paragraph(bloque.strip())

    salida = BytesIO()
    documento.save(salida)
    salida.seek(0)
    return salida.getvalue()
