from pathlib import Path
import json
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
    tabla_criterios_diagnostico,
    tiene_historico_entrenamiento,
)
from src.medidas_preventivas import (
    FACTOR_A_NOMBRE,
    documento_combinado_medidas,
    factores_para_estudiante,
    leer_documento,
    medidas_para_factores,
)
from src.riesgo_abandono import cargar_modelo, diagnosticar_estudiantes


st.set_page_config(
    page_title="Sistema de Alerta Temprana de Abandono Escolar",
    page_icon="🎓",
    layout="wide",
)

st.title("Sistema de Alerta Temprana de Abandono Escolar")
st.write(
    "Análisis masivo de estudiantes de sexto grado a bachillerato mediante "
    "asistencia, rendimiento académico y checklist de factores de riesgo."
)
st.caption(
    "El sistema es una herramienta de apoyo. Una alerta no significa que el estudiante necesariamente abandonará sus estudios."
)

# -------------------------------------------------------------------------
# Modelo disponible
# -------------------------------------------------------------------------
modelo_path = ROOT / "models" / "modelo_abandono_estudiantil.joblib"
metadata_path = ROOT / "models" / "metadata_modelo_abandono.json"
modelo = None
metadata_modelo = {}

if modelo_path.exists():
    try:
        modelo = cargar_modelo(modelo_path)
        if metadata_path.exists():
            metadata_modelo = json.loads(metadata_path.read_text(encoding="utf-8"))
        st.success("Modelo supervisado disponible. El modelo utiliza únicamente asistencia y promedio de notas.")
    except Exception as exc:
        st.warning(f"No se pudo cargar el modelo supervisado. Se utilizará el diagnóstico heurístico. Detalle: {exc}")
else:
    st.info(
        "No hay un histórico supervisado disponible. El sistema utilizará el diagnóstico heurístico basado en asistencia y notas."
    )

# -------------------------------------------------------------------------
# Criterios
# -------------------------------------------------------------------------
with st.expander("Ver criterios, ponderación y reglas de información"):
    st.write(
        "La ponderación completa suma 100%. Peso 1 = 20%, peso 2 = 50% y peso 3 = 30%. "
        "Los campos vacíos se consideran información insuficiente y nunca se interpretan como NO."
    )
    st.dataframe(tabla_criterios_diagnostico(), width="stretch", hide_index=True)
    st.markdown(
        "**Regla de información:** si asistencia o promedio de notas están vacíos, el estudiante no recibe un nivel de riesgo y queda como `INFORMACION_INSUFICIENTE`. "
        "Si faltan campos del checklist, el riesgo académico/contextual disponible se puede calcular, pero el registro queda marcado como `INFORMACION_INSUFICIENTE` y debe completarse antes de una decisión institucional."
    )

# -------------------------------------------------------------------------
# 1. Archivo del salón
# -------------------------------------------------------------------------
st.header("1. Archivo del salón")

archivo_subido = st.file_uploader(
    "Subir archivo del salón",
    type=["xlsx", "xls", "csv"],
    help=(
        "Debe contener porcentaje_asistencia y promedio_notas. También puede contener los 12 factores del checklist. "
        "Los campos vacíos se tratarán como información insuficiente."
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
        st.caption("También puedes colocar el Excel en data/entrada/. El sistema lo detectará automáticamente.")

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
# 2. Entrenamiento histórico opcional
# -------------------------------------------------------------------------
st.header("2. Histórico de entrenamiento (opcional)")
st.caption(
    "Solo utilice esta sección cuando exista un histórico real con la columna 'abandono'. "
    "El entrenamiento optimiza hiperparámetros, compara modelos, construye un ensemble, "
    "realiza validación cruzada y reserva una prueba final."
)

historico = st.file_uploader(
    "Subir histórico de entrenamiento",
    type=["xlsx", "xls", "csv"],
    key="historico_entrenamiento",
)

if historico is not None and st.button("Entrenar modelo supervisado", type="secondary"):
    try:
        df_historico = leer_archivo_tabular(historico)
        if not tiene_historico_entrenamiento(df_historico):
            st.error("El histórico debe contener 'abandono' con casos de abandono y no abandono.")
        else:
            with st.spinner("Entrenando, optimizando y validando el modelo..."):
                metricas_cv, metricas_finales = train_models(df_historico)
                modelo = cargar_modelo(modelo_path)
                metadata_modelo = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

            st.success("Modelo supervisado entrenado, calibrado y guardado correctamente.")
            st.subheader("Validación cruzada")
            st.dataframe(metricas_cv, width="stretch", hide_index=True)
            st.subheader("Prueba final")
            resumen_final = {
                k: v for k, v in metricas_finales.items()
                if k not in {"classification_report", "confusion_matrix", "best_rf_params"}
            }
            st.json(resumen_final)
            st.write("**Mejores hiperparámetros del Random Forest:**")
            st.json(metricas_finales.get("best_rf_params", {}))
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
                resultado_salon = diagnosticar_estudiantes(df_salon, modelo=modelo)

            st.session_state["resultado_salon"] = resultado_salon
            st.session_state["nombre_origen_salon"] = nombre_origen
            st.success("Análisis del salón completado correctamente.")
        except Exception as exc:
            st.error(f"No fue posible analizar el salón: {exc}")

# -------------------------------------------------------------------------
# 4. Resultados
# -------------------------------------------------------------------------
if "resultado_salon" in st.session_state:
    resultado_salon = st.session_state["resultado_salon"].copy()
    st.header("4. Resultados del análisis")

    total = len(resultado_salon)
    alto = int((resultado_salon["nivel_riesgo"] == "ALTO").sum())
    medio = int((resultado_salon["nivel_riesgo"] == "MEDIO").sum())
    bajo = int((resultado_salon["nivel_riesgo"] == "BAJO").sum())
    info_insuficiente = int((resultado_salon["nivel_riesgo"] == "INFORMACION_INSUFICIENTE").sum())
    revision = int(resultado_salon["requiere_revision"].sum())
    completar = int(resultado_salon["requiere_completar_informacion"].sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Estudiantes", total)
    c2.metric("Riesgo alto", alto)
    c3.metric("Riesgo medio", medio)
    c4.metric("Riesgo bajo", bajo)
    c5.metric("Información insuficiente", info_insuficiente)
    c6.metric("Requieren completar datos", completar)

    cobertura_promedio = float(resultado_salon["cobertura_diagnostico"].mean() * 100)
    st.metric("Cobertura promedio del diagnóstico", f"{cobertura_promedio:.1f}%")

    if completar:
        st.warning(
            f"Hay {completar} estudiante(s) con uno o más campos vacíos. Los campos vacíos se consideran información insuficiente. "
            "No se interpretan como una respuesta negativa."
        )

    columnas_mostrar = [
        c for c in [
            "id_estudiante", "grado", "fecha_corte", "porcentaje_asistencia", "promedio_notas",
            "riesgo_asistencia", "riesgo_notas", "riesgo_academico", "riesgo_checklist",
            "cobertura_diagnostico", "porcentaje_riesgo_final", "estado_informacion",
            "campos_sin_informacion", "nivel_riesgo", "prioridad_seguimiento",
            "factores_academicos_detectados", "factores_checklist_detectados",
            "requiere_revision", "requiere_completar_informacion",
        ] if c in resultado_salon.columns
    ]

    tabla = resultado_salon.sort_values(
        ["score_riesgo_final", "nivel_riesgo"], ascending=[False, True], na_position="last"
    )
    st.dataframe(tabla[columnas_mostrar], width="stretch", hide_index=True)

    # ---------------------------------------------------------------------
    # 5. Casos prioritarios
    # ---------------------------------------------------------------------
    st.header("5. Estudiantes que requieren mayor atención")
    prioritarios = tabla[tabla["nivel_riesgo"].isin(["ALTO", "MEDIO"])].copy()
    if prioritarios.empty:
        st.success("No se detectaron estudiantes con riesgo medio o alto.")
    else:
        st.dataframe(prioritarios[columnas_mostrar], width="stretch", hide_index=True)

    # ---------------------------------------------------------------------
    # 6. Medidas preventivas y de mitigación
    # ---------------------------------------------------------------------
    st.header("6. Medidas preventivas y de mitigación")
    st.write(
        "Seleccione un estudiante prioritario. El sistema mostrará de una sola vez todas las medidas "
        "asociadas a los factores detectados. No es necesario seleccionar cada factor individualmente. "
        "Los DOCX pueden sustituirse en la carpeta correspondiente sin modificar el código."
    )

    if prioritarios.empty:
        st.info("No hay estudiantes prioritarios para asociar medidas en este momento.")
    else:
        opciones = {}
        for idx, fila in prioritarios.iterrows():
            ident = fila.get("id_estudiante", idx)
            grado = fila.get("grado", "")
            nivel = fila.get("nivel_riesgo", "")
            opciones[idx] = f"{ident} | {grado} | Riesgo {nivel}"

        seleccionado_idx = st.selectbox(
            "Seleccione un estudiante",
            list(opciones.keys()),
            format_func=lambda x: opciones[x],
            key="estudiante_medidas",
        )
        fila = prioritarios.loc[seleccionado_idx]
        factores = factores_para_estudiante(fila)

        if fila.get("estado_informacion") == "INFORMACION_INSUFICIENTE":
            st.warning(
                "El estudiante tiene información incompleta. Las medidas mostradas corresponden únicamente a factores "
                "que sí fueron informados o señales académicas detectables. Complete los campos faltantes antes de tomar decisiones."
            )

        if not factores:
            st.info("No se identificaron factores con respuesta afirmativa para asociar medidas.")
        else:
            medidas = medidas_para_factores(ROOT, factores)
            st.subheader(f"Todas las medidas detectadas ({len(medidas)})")

            # Documento combinado para el caso seleccionado.
            identificador = str(fila.get("id_estudiante", seleccionado_idx))
            doc_combinado = documento_combinado_medidas(
                ROOT,
                factores,
                identificador=identificador,
                nivel_riesgo=str(fila.get("nivel_riesgo", "")),
            )
            st.download_button(
                "Descargar todas las medidas en un solo DOCX",
                data=doc_combinado,
                file_name=f"medidas_preventivas_{identificador}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"descargar_todas_{seleccionado_idx}",
            )

            for medida in medidas:
                with st.expander(medida.nombre, expanded=True):
                    st.caption(f"Categoría: {medida.categoria.replace('_', ' ').title()}")
                    st.caption(f"Archivo editable: {medida.ruta.relative_to(ROOT)}")
                    try:
                        contenido = leer_documento(medida.ruta)
                        st.text(contenido)
                    except Exception as exc:
                        st.warning(f"No fue posible mostrar la vista previa: {exc}")
                    st.download_button(
                        "Descargar esta medida",
                        data=medida.ruta.read_bytes(),
                        file_name=medida.ruta.name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"descargar_medida_{seleccionado_idx}_{medida.factor}",
                    )

    # ---------------------------------------------------------------------
    # 7. Completar checklist manualmente
    # ---------------------------------------------------------------------
    st.header("7. Completar checklist manualmente")
    st.write(
        "Los campos vacíos se muestran como `NO INFORMADO`. Seleccione SI o NO cuando exista información verificable. "
        "Un valor NO es una respuesta válida y no se considera información insuficiente."
    )

    checklist_editor = resultado_salon.copy()
    for criterio in CRITERIOS_CHECKLIST:
        if criterio not in checklist_editor.columns:
            checklist_editor[criterio] = pd.NA
        checklist_editor[criterio] = checklist_editor[criterio].map(
            lambda x: "NO INFORMADO" if pd.isna(x) else ("SI" if bool(x) else "NO")
        )

    columnas_editor = [
        c for c in ["id_estudiante", "grado", "porcentaje_asistencia", "promedio_notas"]
        if c in checklist_editor.columns
    ] + CRITERIOS_CHECKLIST

    editor = st.data_editor(
        checklist_editor[columnas_editor],
        width="stretch",
        hide_index=True,
        column_config={
            criterio: st.column_config.SelectboxColumn(
                criterio.replace("_", " ").title(),
                options=["NO INFORMADO", "SI", "NO"],
                required=True,
            )
            for criterio in CRITERIOS_CHECKLIST
        },
        disabled=[c for c in ["id_estudiante", "grado", "porcentaje_asistencia", "promedio_notas"] if c in columnas_editor],
        key="editor_checklist",
    )

    if st.button("Actualizar diagnóstico con checklist"):
        try:
            actualizado = aplicar_checklist_desde_editor(resultado_salon, editor, modelo=modelo)
            st.session_state["resultado_salon"] = actualizado
            st.success("Diagnóstico actualizado con el checklist informado.")
            st.rerun()
        except Exception as exc:
            st.error(f"No fue posible actualizar el diagnóstico: {exc}")

    # ---------------------------------------------------------------------
    # 8. Exportar resultados
    # ---------------------------------------------------------------------
    st.header("8. Exportar resultados")
    archivo_excel = exportar_diagnostico_excel(resultado_salon, root=ROOT)
    st.download_button(
        label="Descargar reporte completo en Excel",
        data=archivo_excel,
        file_name="diagnostico_abandono_salon.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
