# Despliegue web del sistema

El proyecto utiliza Streamlit, por lo que la interfaz ya está construida como una aplicación web. Esta versión agrega un `Dockerfile` para facilitar su ejecución en un servidor interno o servicio de contenedores.

## Ejecución web local

```powershell
streamlit run app/streamlit_app.py --server.address 0.0.0.0
```

Desde otro equipo de la misma red se puede acceder utilizando la IP del servidor y el puerto 8501, siempre que el firewall lo permita.

## Ejecución con Docker

Construir la imagen:

```powershell
docker build -t alerta-abandono-escolar .
```

Ejecutar:

```powershell
docker run --rm -p 8501:8501 alerta-abandono-escolar
```

La aplicación queda disponible en `http://localhost:8501`.

## Recomendación para uso institucional

No se recomienda publicar la aplicación directamente en Internet cuando contenga información identificable de estudiantes. La publicación institucional debe utilizar HTTPS, autenticación y autorización, control de acceso por centro educativo, respaldo, registro de actividad y políticas de retención de datos.

Una arquitectura apropiada es colocar Streamlit detrás de un proxy o servicio de identidad institucional. La autenticación no debe implementarse con contraseñas escritas directamente en el código.

## Datos y privacidad

- No almacenar archivos de estudiantes dentro del repositorio.
- Mantener `data/entrada/` fuera del control de versiones.
- No publicar modelos entrenados con información personal sin autorización.
- Utilizar identificadores internos en lugar de nombres completos cuando sea posible.
- Limitar el acceso a los documentos de medidas y reportes.
- Mantener revisión humana para toda alerta.

## Alcance del despliegue

El Dockerfile demuestra que el prototipo es técnicamente desplegable como aplicación web. Esto no equivale a certificar un despliegue institucional ni a validar su seguridad para producción.
