from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advanced_training import train_models
from src.batch_analysis import (
    CRITERIOS_CHECKLIST,
    aplicar_checklist_desde_editor,
    exportar_diagnostico_excel,
    listar_archivos_entrada,
    leer_archivo_tabular,
    nombres_hojas,
    resumen_diagnostico,
    tiene_historico_entrenamiento,
)
from src.riesgo_abandono import cargar_modelo, diagnosticar_estudiantes


st.set_page_config(
    page_title="Análisis de Riesgo de Abandono - Salón",
    page_icon="🎓",
    layout="wide",
)

st.title("Análisis masivo del salón")
st.write(
    "Carga un Excel con todos los estudiantes del salón. El sistema analizará "
    "cada estudiante utilizando asistencia, rendimiento académico y, cuando estén "
    "disponibles, los factores del checklist social."
)

# -------------------------------------------------------------------------
# Modelo disponible
# -------------------------------------------------------------------------
modelo_path = ROOT / "models" / "modelo_abandono_estudiantil.joblib"
modelo = None

if modelo_path.exists():
    try:
        modelo = cargar_modelo(modelo_path)
        st.success("Modelo supervisado cargado correctamente.")
    except Exception as exc:
        st.warning(
            "No se pudo cargar el modelo supervisado. Se utilizará el diagnóstico "
            f"académico heurístico. Detalle: {exc}"
        )
else:
    st.info(
        "No hay modelo histórico entrenado. El análisis se realizará mediante "
        "reglas heurísticas basadas en asistencia y notas."
    )

# -------------------------------------------------------------------------
# 1. Archivo del salón
# -------------------------------------------------------------------------
st.header("1. Archivo del salón")

archivo_subido = st.file_uploader(
    "Subir archivo del salón",
    type=["xlsx", "xls", "csv"],
    help=(
        "Debe contener al menos porcentaje_asistencia y promedio_notas. "
        "También puede contener las columnas del checklist social."
    ),
    key="archivo_salon",
)

origen_salon = None
nombre_origen = None

if archivo_subido is not None:
    origen_salon = archivo_subido
    nombre_origen = archivo_subido.name
else:
    archivos = listar_archivos_entrada(ROOT)
    if archivos:
        etiquetas = [str(p.relative_to(ROOT)) for p in archivos]
        seleccionado = st.selectbox(
            "O seleccionar un archivo colocado en la carpeta del proyecto",
            etiquetas,
            key="archivo_salon_carpeta",
        )
        origen_salon = ROOT / seleccionado
        nombre_origen = seleccionado
    else:
        st.caption(
            "También puedes colocar el Excel en data/entrada/. "
            "El sistema lo detectará automáticamente."
        )

hoja_salon = None
if origen_salon is not None:
    try:
        hojas = nombres_hojas(origen_salon)
        if len(hojas) > 1:
            hoja_salon = st.selectbox("Hoja del Excel", hojas, key="hoja_salon")
        else:
            hoja_salon = hojas[0]
    except Exception as exc:
        st.error(f"No fue posible leer las hojas del archivo: {exc}")

# -------------------------------------------------------------------------
# 2. Histórico de entrenamiento opcional
# -------------------------------------------------------------------------
st.header("2. Histórico de entrenamiento (opcional)")
st.caption(
    "Si dispones de un histórico con la columna 'abandono', puedes entrenar el "
    "modelo antes de analizar el salón. Si no lo tienes, se utilizará el diagnóstico "
    "académico heurístico."
)

historico = st.file_uploader(
    "Subir histórico de entrenamiento",
    type=["xlsx", "xls", "csv"],
    key="historico_entrenamiento",
)

if historico is not None and st.button("Entrenar modelo con el histórico"):
    try:
        df_historico = leer_archivo_tabular(historico)
        if not tiene_historico_entrenamiento(df_historico):
            st.error(
                "El histórico debe contener la columna 'abandono' con casos de "
                "abandono y no abandono."
            )
        else:
            with st.spinner(
                "Entrenando modelos, ejecutando validación cruzada, GridSearchCV "
                "y construyendo el ensemble..."
            ):
                metricas_cv, metricas_finales = train_models(df_historico)
                modelo = cargar_modelo(modelo_path)

            st.success("Modelo avanzado entrenado y guardado correctamente.")
            st.write("**Resultados de validación cruzada:**")
            st.dataframe(metricas_cv, use_container_width=True)
            st.write("**Resultados del conjunto de prueba final:**")
            st.json(
                {
                    k: v
                    for k, v in metricas_finales.items()
                    if k != "classification_report"
                }
            )
    except Exception as exc:
        st.error(f"No fue posible entrenar el modelo avanzado: {exc}")

# -------------------------------------------------------------------------
# 3. Ejecutar análisis masivo
# -------------------------------------------------------------------------
st.header("3. Ejecutar análisis del salón")

if origen_salon is None:
    st.warning("Carga o selecciona primero el archivo del salón.")
else:
    st.write(
        f"**Archivo seleccionado:** `{nombre_origen}`"
        + (f" | **Hoja:** `{hoja_salon}`" if hoja_salon else "")
    )

    if st.button("Analizar todos los alumnos del salón", type="primary"):
        try:
            with st.spinner("Analizando todos los estudiantes..."):
                df_salon = leer_archivo_tabular(origen_salon, hoja_salon)
                resultado_salon = diagnosticar_estudiantes(
                    df_salon,
                    modelo=modelo,
                )

            st.session_state["resultado_salon"] = resultado_salon
            st.session_state["nombre_origen_salon"] = nombre_origen

        except Exception as exc:
            st.error(f"No fue posible analizar el salón: {exc}")

# -------------------------------------------------------------------------
# 4. Resultados
# -------------------------------------------------------------------------
if "resultado_salon" in st.session_state:
    resultado_salon = st.session_state["resultado_salon"].copy()

    st.header("4. Resultados del análisis")

    resumen = resumen_diagnostico(resultado_salon)
    total = len(resultado_salon)
    alto = int((resultado_salon["nivel_riesgo"] == "ALTO").sum())
    medio = int((resultado_salon["nivel_riesgo"] == "MEDIO").sum())
    bajo = int((resultado_salon["nivel_riesgo"] == "BAJO").sum())
    revision = int(resultado_salon["requiere_revision"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Estudiantes", total)
    c2.metric("Riesgo alto", alto)
    c3.metric("Riesgo medio", medio)
    c4.metric("Riesgo bajo", bajo)
    c5.metric("Requieren revisión", revision)

    estado_checklist = resultado_salon["estado_checklist"].value_counts().to_dict()
    if estado_checklist.get("NO INFORMADO", 0) == total:
        st.info(
            "El Excel del salón no contiene datos del checklist. En este escenario "
            "el resultado se basa en asistencia y notas. La ausencia de información "
            "social no se interpreta como que el estudiante no presenta esos factores."
        )
    elif estado_checklist.get("NO INFORMADO", 0) > 0:
        st.warning(
            "El checklist está incompleto para algunos estudiantes. El peso social "
            "se aplica proporcionalmente a la información disponible."
        )
    else:
        st.success("El checklist social está completo para todos los estudiantes.")

    columnas_mostrar = [
        c
        for c in [
            "id_estudiante",
            "grado",
            "porcentaje_asistencia",
            "promedio_notas",
            "riesgo_academico",
            "riesgo_checklist",
            "estado_checklist",
            "score_riesgo_final",
            "nivel_riesgo",
            "prioridad_seguimiento",
            "factores_academicos_detectados",
            "factores_checklist_detectados",
            "requiere_revision",
        ]
        if c in resultado_salon.columns
    ]

    tabla = resultado_salon.sort_values(
        ["score_riesgo_final", "nivel_riesgo"],
        ascending=[False, True],
    )

    st.dataframe(
        tabla[columnas_mostrar],
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------------------
    # 5. Casos prioritarios
    # ---------------------------------------------------------------------
    st.header("5. Estudiantes que requieren mayor atención")
    prioritarios = tabla[tabla["nivel_riesgo"].isin(["ALTO", "MEDIO"])]
    if prioritarios.empty:
        st.success("No se detectaron estudiantes en riesgo medio o alto.")
    else:
        st.dataframe(
            prioritarios[columnas_mostrar],
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------------------
    # 6. Checklist manual opcional
    # ---------------------------------------------------------------------
    st.header("6. Completar checklist manualmente (opcional)")
    st.write(
        "Si el Excel no contiene información social, puedes completar el checklist "
        "para cada estudiante. Selecciona SI, NO o NO INFORMADO y actualiza el diagnóstico."
    )

    checklist_editor = resultado_salon.copy()
    for criterio in CRITERIOS_CHECKLIST:
        if criterio not in checklist_editor.columns:
            checklist_editor[criterio] = pd.NA
        checklist_editor[criterio] = checklist_editor[criterio].map(
            lambda x: "NO INFORMADO" if pd.isna(x) else ("SI" if bool(x) else "NO")
        )

    columnas_editor = [
        c
        for c in [
            "id_estudiante",
            "grado",
            "porcentaje_asistencia",
            "promedio_notas",
        ]
        if c in checklist_editor.columns
    ] + CRITERIOS_CHECKLIST

    editor = st.data_editor(
        checklist_editor[columnas_editor],
        use_container_width=True,
        hide_index=True,
        column_config={
            criterio: st.column_config.SelectboxColumn(
                criterio.replace("_", " ").title(),
                options=["NO INFORMADO", "SI", "NO"],
                required=True,
            )
            for criterio in CRITERIOS_CHECKLIST
        },
        disabled=[
            c
            for c in [
                "id_estudiante",
                "grado",
                "porcentaje_asistencia",
                "promedio_notas",
            ]
            if c in columnas_editor
        ],
        key="editor_checklist",
    )

    if st.button("Actualizar diagnóstico con checklist"):
        try:
            actualizado = aplicar_checklist_desde_editor(
                resultado_salon,
                editor,
                modelo=modelo,
            )
            st.session_state["resultado_salon"] = actualizado
            st.success(
                "Diagnóstico actualizado. El riesgo final ahora considera los factores "
                "sociales informados."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"No fue posible actualizar el diagnóstico: {exc}")

    # ---------------------------------------------------------------------
    # 7. Exportar resultados
    # ---------------------------------------------------------------------
    st.header("7. Exportar resultados")
    archivo_excel = exportar_diagnostico_excel(resultado_salon)

    st.download_button(
        label="Descargar reporte completo en Excel",
        data=archivo_excel,
        file_name="diagnostico_abandono_salon.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
