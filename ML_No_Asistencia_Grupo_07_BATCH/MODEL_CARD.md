# Model Card — Riesgo de abandono estudiantil

## Propósito

Apoyar la identificación temprana de estudiantes de sexto grado a bachillerato que requieren revisión por posible riesgo de abandono y orientar medidas preventivas y de mitigación.

## Unidad de análisis

Un estudiante dentro del salón cargado.

## Variables académicas

- Porcentaje de asistencia.
- Promedio de notas.

El modelo supervisado utiliza únicamente estas variables como predictores.

## Objetivo supervisado

`abandono`: resultado histórico que indica si el estudiante abandonó posteriormente sus estudios.

Sin esta etiqueta no existe un resultado supervisado observado. En ese escenario el sistema utiliza reglas heurísticas y no debe presentar el resultado como una probabilidad aprendida por Machine Learning.

## Checklist contextual

Los doce criterios se mantienen separados del entrenamiento por la ausencia de históricos sociales confiables. Se aplican como complemento del diagnóstico.

## Ponderación

### Peso 1 — 20% total
- porcentaje_asistencia: 10%.
- promedio_notas: 10%.

### Peso 2 — 50% total
- trabajo_infantil
- migracion
- embarazo_adolescente
- pobreza
- dificultades_aprendizaje
- antecedentes_abandono_escolar
- violencia_intrafamiliar

### Peso 3 — 30% total
- bullying: 6%.
- repitencia_escolar: 6%.
- falta_transporte_accesible: 6%.
- distancia_vivienda_centro_educativo: 6%.
- limitaciones_conectividad_recursos_tecnologicos: 6%.

## Información insuficiente

Los campos vacíos, N/A, NA, NULL y NO INFORMADO se consideran información insuficiente.

Una respuesta `NO` es válida y significa que el factor fue evaluado y no fue identificado.

Si falta asistencia o promedio de notas, el estudiante queda en `INFORMACION_INSUFICIENTE` y no se inventa un nivel de riesgo.

Si faltan factores del checklist, el sistema puede calcular el riesgo con la información disponible, pero el registro se marca como incompleto y muestra los campos faltantes.

## Entrenamiento avanzado

Cuando existe histórico etiquetado, el flujo oficial incorpora:

- Regresión Logística.
- Random Forest.
- Gradient Boosting.
- XGBoost cuando está disponible.
- GridSearchCV.
- StratifiedKFold.
- VotingClassifier.
- Calibración de probabilidades.
- Separación temporal por `fecha_corte` cuando es posible.
- Prueba final independiente.
- Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC y Brier Score.

## Fairness

Las variables sensibles no se utilizan como predictores. Cuando existe información autorizada, Fairlearn permite auditar Recall, False Positive Rate y Selection Rate por grupo.

## Medidas preventivas y de mitigación

Cuando un estudiante queda en riesgo medio o alto, la aplicación relaciona los factores detectados con documentos DOCX ubicados en `medidas_preventivas/`. Las medidas pueden sustituirse manteniendo nombre y carpeta.

La interfaz muestra todas las medidas del estudiante en una sola vista y permite descargar un documento combinado.

## Limitaciones

- Asistencia y notas no capturan todas las causas del abandono.
- Sin etiqueta histórica no es posible medir capacidad predictiva real de abandono individual.
- Los factores sociales son sensibles y dependen de la calidad del reporte.
- Un checklist incompleto reduce la cobertura del diagnóstico.
- Una métrica alta sobre datos sintéticos no demuestra desempeño sobre estudiantes reales.
- El sistema no debe utilizarse para sancionar, excluir o restringir matrícula.
- La alerta requiere revisión humana.

## Uso web

La aplicación está preparada para ejecutarse como aplicación web mediante Streamlit y Docker. Un despliegue institucional debe añadir autenticación, autorización, HTTPS, gestión de datos y políticas de retención adecuadas.
