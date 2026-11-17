# Guía de ejecución

## 1. Instalar dependencias

```powershell
pip install -r requirements.txt
```

## 2. Ejecutar la interfaz

```powershell
streamlit run app/streamlit_app.py
```

La interfaz está enfocada exclusivamente en el análisis masivo del salón.

## 3. Analizar un salón

En Streamlit se puede:

- subir un Excel/CSV;
- seleccionar un archivo colocado en `data/entrada/`.

El sistema analiza todos los registros del archivo.

## 4. Formato mínimo del Excel

```text
porcentaje_asistencia | promedio_notas
```

Se recomienda incluir:

```text
id_estudiante | grado | porcentaje_asistencia | promedio_notas
```

Los nombres de columnas comunes son normalizados automáticamente.

## 5. Checklist

Si el Excel contiene las columnas sociales, se utilizan automáticamente.

Si no las contiene:

- el checklist queda `NO INFORMADO`;
- no se interpreta como ausencia del factor;
- el diagnóstico inicial se basa en asistencia y notas.

La aplicación permite completar manualmente el checklist después del análisis masivo.

## 6. Entrenamiento

Si se cuenta con un histórico etiquetado:

```powershell
python entrenar_modelo.py historico.xlsx
```

El histórico debe tener:

```text
porcentaje_asistencia
promedio_notas
abandono
```

La etiqueta `abandono` debe contener casos de abandono y no abandono.

## 7. Análisis por consola

```powershell
python analizar_salon.py salon.xlsx
```

Con modelo:

```powershell
python analizar_salon.py salon.xlsx --modelo models/modelo_abandono_estudiantil.joblib
```

## 8. Salida

Se genera un Excel con:

- diagnóstico de todos los estudiantes;
- resumen del salón;
- casos de riesgo medio/alto;
- metodología utilizada.

## 9. Interpretación

El sistema genera una alerta de riesgo, no una sentencia sobre el estudiante. La intervención final debe ser realizada por personal educativo.
