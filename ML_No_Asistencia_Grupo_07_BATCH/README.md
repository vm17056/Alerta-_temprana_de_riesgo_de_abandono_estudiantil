# Sistema de alerta temprana de abandono escolar

Este proyecto permite analizar de forma masiva estudiantes de sexto grado a bachillerato utilizando principalmente porcentaje de asistencia y promedio de notas, complementados con un checklist de factores sociales.

## Análisis masivo por salón

La interfaz Streamlit está enfocada exclusivamente en el análisis de un salón completo. El archivo puede:

1. Subirse desde la interfaz.
2. Colocarse en `data/entrada/`.
3. También puede colocarse en la raíz del proyecto.

El sistema detecta automáticamente archivos `.xlsx`, `.xls` y `.csv`.

### Columnas mínimas

```text
porcentaje_asistencia
promedio_notas
```

También se aceptan encabezados comunes como `Asistencia`, `Promedio`, `ID Estudiante`, etc.

### Columnas opcionales

```text
id_estudiante
grado
trabajo_infantil
migracion
embarazo_adolescente
pobreza
vive_lejos
bullying
```

## Flujo del análisis

El sistema procesa todos los estudiantes del archivo y genera un diagnóstico para cada registro.

Si existe un modelo supervisado entrenado con un histórico que contiene `abandono`, se utiliza la probabilidad estimada por el modelo. Si no existe un modelo, se utiliza el score heurístico documentado en el proyecto.

El checklist tiene:

- Peso 2: trabajo infantil, migración, embarazo adolescente y pobreza.
- Peso 1: vive lejos y bullying.

Cuando el Excel no contiene información social, el sistema marca el checklist como `NO INFORMADO` y NO interpreta esa ausencia como un `NO`. En ese caso el resultado inicial se basa en el componente académico.

La aplicación también permite completar manualmente el checklist después del análisis y recalcular el riesgo final.

## Histórico de entrenamiento

Se puede cargar un histórico desde Streamlit o entrenar desde consola:

```powershell
python entrenar_modelo.py data/entrenamiento/historico.xlsx
```

El histórico debe contener `abandono` con casos positivos y negativos.

## Ejecución de Streamlit

```powershell
streamlit run app/streamlit_app.py
```

La aplicación mostrará únicamente el flujo de análisis masivo del salón.

## Análisis masivo desde consola

```powershell
python analizar_salon.py data/entrada/salon.xlsx
```

Con modelo:

```powershell
python analizar_salon.py data/entrada/salon.xlsx --modelo models/modelo_abandono_estudiantil.joblib
```

El resultado se genera en:

```text
outputs/diagnostico_salon.xlsx
```

El Excel contiene:

- Diagnostico_Alumnos
- Resumen
- Casos_Prioritarios
- Metodologia

## Interpretación

`ALTO` y `MEDIO` significan que el estudiante requiere revisión. No significa que el estudiante necesariamente abandonará. El sistema está diseñado como herramienta de alerta temprana y apoyo a la decisión.
