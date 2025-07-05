---
title: "Clasificador de Madera"
emoji: "🌲"
colorFrom: "green"
colorTo: "yellow"
sdk: "docker"
sdk_version: "1.0.0"
app_file: app.py
pinned: false
---

# 🌲 Clasificador de Tipos de Madera con FastAPI

Este proyecto implementa una API REST que clasifica imágenes de madera utilizando un modelo entrenado con PyTorch. Está diseñado para ser desplegado en Hugging Face Spaces usando Docker.

## 🚀 Cómo usar la API

### Endpoint: `/predict`
- Método: `POST`
- Formato: `multipart/form-data`
- Campo: `file` (imagen a clasificar)

### Respuesta de ejemplo
```json
{
  "tipo": "pino"
}
```
## 🌲 Tipos de Madera identificables

- `Caoba, cerezo, ébano, nogal, olivo, pino`

## 🎯 Precisión

- `73.33%`

## 📂 Archivos incluidos

- `app.py`: FastAPI con endpoint `/predict`
- `model.pt`: Modelo entrenado con PyTorch
- `wood_db.imagenes.csv`: Etiquetas para reconstruir clases
- `Dockerfile`: Para ejecutar en Hugging Face
- `requirements.txt`: Dependencias necesarias

## 🧪 Prueba local

```bash
docker build -t clasificador-madera .
docker run -p 7860:7860 clasificador-madera
```

Abre tu navegador en [http://localhost:7860](http://localhost:7860)

## 📱 Despliegue

1. Crea un Space tipo **Docker** en Hugging Face
2. Sube todos estos archivos
3. Espera a que se construya el contenedor y accede a tu API desde el móvil
