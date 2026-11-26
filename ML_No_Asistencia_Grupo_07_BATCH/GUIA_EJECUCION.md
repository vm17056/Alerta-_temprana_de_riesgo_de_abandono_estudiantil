# Guía de ejecución y operación

## 1. Instalar dependencias

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Ejecutar la interfaz web

```powershell
streamlit run app\streamlit_app.py
```

La interfaz está enfocada exclusivamente en el análisis masivo del salón.

## 3. Cargar el salón

El Excel puede:

- subirse desde Streamlit;
- colocarse en `data/entrada/`;
- colocarse en la raíz del proyecto.

Se admiten `.xlsx`, `.xls` y `.csv`.

## 4. Columnas académicas

```text
porcentaje_asistencia
promedio_notas
```

Ambas son necesarias para producir un nivel de riesgo. Si alguna está vacía, el estudiante queda en `INFORMACION_INSUFICIENTE`.

Las notas pueden venir en escala 0-10 o 0-100; la aplicación normaliza 0-100 a 0-10.

Para probar el manejo de campos vacíos utiliza `data/templates/plantilla_salon_informacion_insuficiente.xlsx`.

## 5. Checklist

```text
trabajo_infantil
migracion
embarazo_adolescente
pobreza
bullying
repitencia_escolar
dificultades_aprendizaje
antecedentes_abandono_escolar
violencia_intrafamiliar
falta_transporte_accesible
distancia_vivienda_centro_educativo
limitaciones_conectividad_recursos_tecnologicos
```

### Regla de datos vacíos

- vacío = `INFORMACION_INSUFICIENTE`;
- `NO` = respuesta válida;
- `SI` = factor detectado;
- `NO INFORMADO` = información insuficiente.

El sistema no convierte un vacío en `NO`.

## 6. Ponderación

- Peso 1: 20%.
- Peso 2: 50%.
- Peso 3: 30%.

## 7. Medidas preventivas

Después del análisis, seleccione un estudiante de riesgo medio o alto.

La aplicación mostrará de una sola vez todos los documentos asociados a sus factores detectados y permitirá:

- visualizar cada documento;
- descargar cada documento;
- descargar todas las medidas del estudiante en un solo DOCX.

Los documentos fuente se pueden modificar directamente en `medidas_preventivas/`.

## 8. Completar información faltante

La sección de edición permite cambiar `NO INFORMADO` a `SI` o `NO` cuando exista información verificada.

Después de actualizar el checklist, presione:

```text
Actualizar diagnóstico con checklist
```

El sistema vuelve a calcular el diagnóstico.

## 9. Entrenamiento supervisado

Solo debe utilizarse con un histórico real que tenga una etiqueta posterior de abandono.

```powershell
python entrenar_modelo.py historico.xlsx
```

El flujo realiza optimización, validación cruzada, ensemble, calibración y prueba final.

La plantilla incluida:

```text
data/templates/plantilla_historico_entrenamiento.csv
```

es sintética y únicamente sirve para comprobar el funcionamiento técnico.

## 10. Fairness

Cuando exista una variable sensible autorizada:

```powershell
python evaluar_fairness.py historico.xlsx genero
```

No se recomienda usar datos sensibles como predictores del modelo. Se utilizan para auditoría de resultados por grupo.

## 11. Pruebas

```powershell
python -m pytest -q
```

## 12. Despliegue web

Para instrucciones de despliegue con Docker consulte `WEB_DEPLOYMENT.md`.

En producción deben existir controles de autenticación, autorización, HTTPS, respaldo y protección de datos.
