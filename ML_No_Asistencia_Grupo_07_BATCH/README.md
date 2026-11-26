# Sistema de alerta temprana de abandono escolar

Proyecto de la Tesina de Ingeniería de Sistemas Informáticos de la Universidad de El Salvador para el diagnóstico de riesgo de abandono escolar en estudiantes de sexto grado a bachillerato.

La aplicación analiza un salón completo a partir de Excel/CSV, utiliza asistencia y notas como base académica y complementa el diagnóstico con un checklist de factores contextuales. Para estudiantes con riesgo medio o alto, genera las medidas preventivas y de mitigación asociadas a los factores detectados.

## 1. Ponderación del diagnóstico

La distribución oficial es:

- **Peso 1: 20% total**
  - porcentaje_asistencia: 10%
  - promedio_notas: 10%
- **Peso 2: 50% total**
  - trabajo_infantil
  - migracion
  - embarazo_adolescente
  - pobreza
  - dificultades_aprendizaje
  - antecedentes_abandono_escolar
  - violencia_intrafamiliar
- **Peso 3: 30% total**
  - bullying: 6%
  - repitencia_escolar: 6%
  - falta_transporte_accesible: 6%
  - distancia_vivienda_centro_educativo: 6%
  - limitaciones_conectividad_recursos_tecnologicos: 6%

Los siete criterios de peso 2 utilizan internamente `0.50/7` para conservar exactamente el 50%, aunque se muestran como 7.14% redondeado.

## 2. Regla de información insuficiente

Los campos vacíos, `N/A`, `NA`, `NULL`, `NO INFORMADO` y valores equivalentes se consideran **información insuficiente**.

Una respuesta `NO` sí es información válida y no se interpreta como vacío.

### Datos académicos

Si `porcentaje_asistencia` o `promedio_notas` están vacíos:

- el estudiante queda como `INFORMACION_INSUFICIENTE`;
- no se inventa un valor;
- no se genera un nivel BAJO/MEDIO/ALTO;
- se indican los campos faltantes.

### Checklist

Si uno o más campos sociales están vacíos:

- el riesgo disponible puede calcularse utilizando los criterios informados;
- la cobertura se muestra en `cobertura_diagnostico`;
- el estudiante queda marcado como `INFORMACION_INSUFICIENTE` en el estado de información;
- los campos vacíos no se consideran `NO`;
- se recomienda completar el checklist antes de tomar una decisión institucional.

## 3. Diagnóstico heurístico vs. modelo supervisado

El proyecto distingue explícitamente dos situaciones:

### Sin histórico real de abandono

Se utiliza un diagnóstico heurístico basado en asistencia y notas. No debe presentarse como una probabilidad estadística de abandono.

### Con histórico real de abandono

Se puede entrenar un modelo supervisado. El modelo oficial utiliza únicamente:

```text
porcentaje_asistencia
promedio_notas
```

El checklist se aplica posteriormente como complemento contextual y no como predictor entrenado.

## 4. Entrenamiento avanzado

El flujo oficial de entrenamiento incluye:

- Regresión Logística como baseline.
- Random Forest.
- Gradient Boosting.
- XGBoost cuando está disponible.
- GridSearchCV para optimización de Random Forest.
- StratifiedKFold.
- Ensemble `VotingClassifier`.
- Calibración de probabilidades mediante `CalibratedClassifierCV`.
- Prueba final independiente.
- Accuracy.
- Balanced Accuracy.
- Precision.
- Recall.
- F1.
- ROC-AUC.
- PR-AUC.
- Brier Score.
- Matriz de confusión.
- Curva de calibración.

Si el histórico contiene `fecha_corte` válida, el sistema intenta utilizar una separación temporal para reservar los registros más recientes como prueba final. Si no es posible, utiliza una división estratificada.

Los resultados se guardan en:

```text
reports/training/
├── metricas_validacion_cruzada.csv
├── metricas_prueba_final.csv
├── mejores_hiperparametros.json
├── matriz_confusion.csv
├── calibracion.csv
└── resumen_entrenamiento.json
```

El modelo se guarda en:

```text
models/modelo_abandono_estudiantil.joblib
```

## 5. Análisis de fairness

Las variables sensibles no forman parte de los predictores académicos. Cuando exista información autorizada para una auditoría, se puede ejecutar:

```powershell
python evaluar_fairness.py historico.xlsx genero
```

El reporte compara por grupo:

- Recall.
- False Positive Rate.
- Selection Rate.
- Brechas entre grupos.

El análisis de fairness es una auditoría del comportamiento del modelo y no debe utilizarse para etiquetar o sancionar estudiantes.

## 6. Análisis masivo por salón

La interfaz permite:

1. subir un Excel/CSV;
2. seleccionar una hoja;
3. analizar todos los estudiantes;
4. visualizar el ranking de riesgo;
5. identificar campos insuficientes;
6. consultar los factores detectados;
7. mostrar todas las medidas preventivas de un estudiante de una sola vez;
8. descargar todas las medidas en un único DOCX;
9. descargar cada medida individualmente;
10. completar posteriormente el checklist desde la interfaz;
11. descargar un reporte completo en Excel.

## 7. Medidas preventivas y de mitigación

Los documentos se encuentran en:

```text
medidas_preventivas/
├── factores_academicos/
├── factores_economicos_sociales/
├── factores_violencia_proteccion/
└── factores_geograficos_tecnologicos/
```

Para modificar una medida, sustituya el DOCX correspondiente manteniendo su nombre y carpeta. No es necesario modificar Python.

La aplicación genera además un documento combinado por estudiante con todas las medidas detectadas. El documento combinado se reconstruye sin hipervínculos externos.

## 8. Ejecución local

Crear entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Ejecutar la aplicación:

```powershell
streamlit run app\streamlit_app.py
```

Pruebas:

```powershell
python -m pytest -q
```

## 9. Entrenamiento desde consola

```powershell
python entrenar_modelo.py data/templates/plantilla_historico_entrenamiento.csv
```

La plantilla histórica incluida es **sintética y únicamente sirve para demostrar técnicamente el flujo de entrenamiento**. No representa abandono real de estudiantes salvadoreños y no debe utilizarse como evidencia de desempeño del proyecto.

Para probar el nuevo manejo de información insuficiente se incluye `data/templates/plantilla_salon_informacion_insuficiente.xlsx`, que contiene campos sociales vacíos y algunos registros con asistencia o notas vacías.

## 10. Análisis desde consola

```powershell
python analizar_salon.py data/entrada/salon.xlsx
```

Con modelo supervisado:

```powershell
python analizar_salon.py data/entrada/salon.xlsx --modelo models/modelo_abandono_estudiantil.joblib
```

## 11. Despliegue web

Streamlit ya proporciona una interfaz web. El proyecto incluye `Dockerfile` y `WEB_DEPLOYMENT.md` para facilitar su despliegue en un servidor interno o plataforma de contenedores.

Para uso institucional se recomienda autenticación, HTTPS, control de acceso y almacenamiento protegido. No se recomienda publicar información identificable de estudiantes directamente en Internet.

## 12. Interpretación responsable

`ALTO` y `MEDIO` indican prioridad de revisión, no una sentencia de abandono.

`INFORMACION_INSUFICIENTE` indica que el sistema no dispone de información suficiente para producir un diagnóstico completo, especialmente cuando falta asistencia o promedio de notas.

La herramienta es un apoyo para priorizar acompañamiento educativo. Las decisiones finales corresponden al personal educativo y a las instituciones competentes.
