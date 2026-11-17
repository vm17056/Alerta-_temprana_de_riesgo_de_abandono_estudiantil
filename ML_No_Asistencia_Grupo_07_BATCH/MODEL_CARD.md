# Model Card — Riesgo de abandono estudiantil

## Propósito
Apoyar la identificación temprana de estudiantes de sexto grado a bachillerato que requieren revisión por posible riesgo de abandono.

## Unidad de análisis
Un estudiante individual.

## Variables del componente académico
- Porcentaje de asistencia.
- Promedio de notas.

El modelo supervisado, cuando exista histórico etiquetado, **no utiliza** variables sociales del checklist para entrenar.

## Objetivo supervisado requerido
`abandono`: indicador histórico de si el estudiante abandonó posteriormente sus estudios.

Sin esta columna no existe una variable objetivo observada y, por tanto, el sistema utiliza un score heurístico de riesgo; dicho score no debe describirse como una probabilidad aprendida por machine learning.

## Checklist complementario
Mayor peso (2): trabajo infantil, migración, embarazo adolescente y pobreza.

Peso normal (1): distancia al centro educativo y bullying.

El checklist se utiliza como señal de apoyo para intervención, no para sancionar o excluir estudiantes.

## Combinación predeterminada
- 70% riesgo académico.
- 30% checklist.

Los pesos y cortes son parámetros del prototipo y requieren calibración con datos reales y validación de especialistas educativos.

## Modelos candidatos
Cuando existe la etiqueta histórica, el entrenamiento compara:
- Regresión Logística con balanceo de clases.
- Random Forest con balanceo de clases.

La selección prioriza balanced accuracy, F1 y ROC-AUC.

## Limitaciones
- Asistencia y notas no capturan todas las causas del abandono.
- Sin etiqueta histórica no es posible medir capacidad predictiva real.
- Los factores del checklist pueden ser sensibles y dependen de la calidad del reporte.
- No se deben tomar decisiones automáticas de alto impacto basadas únicamente en la salida del sistema.
- Antes de uso real debe evaluarse el desempeño por grado, cohorte y centro educativo, incluyendo falsos positivos y falsos negativos.
