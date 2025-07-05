FROM python:3.10-slim

# Evitar preguntas interactivas
ENV DEBIAN_FRONTEND=noninteractive

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY . /app

# Instalar dependencias
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Exponer el puerto por defecto de Uvicorn
EXPOSE 7860

# Comando para ejecutar FastAPI con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
