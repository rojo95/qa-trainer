# Usa la imagen oficial de TensorFlow con Python y pip
FROM tensorflow/tensorflow:latest

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias
COPY requirements.txt ./

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copia tu aplicación
COPY ./app /app

# Comando por defecto: ejecutar el entrenamiento
CMD ["python", "train.py"]
