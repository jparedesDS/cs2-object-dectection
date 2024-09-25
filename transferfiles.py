import cv2
import os
from pathlib import Path
import shutil
import subprocess

# Directorio de imágenes y etiquetas
directorio_imagenes = 'data/valid/nolabels'  # Carpeta de imágenes
directorio_labels = 'data/valid/nolabels2'  # Carpeta de etiquetas
directorio_sin_label = 'data/valid/nolabelsinlabel/'  # Carpeta para imágenes y etiquetas sin labels

# Crear la carpeta 'nolabels' si no existe
os.makedirs(directorio_sin_label, exist_ok=True)

# Procesar todas las imágenes y etiquetas
for archivo_imagen in os.listdir(directorio_imagenes):
    img_path = os.path.join(directorio_imagenes, archivo_imagen)
    label_path = os.path.join(directorio_labels, Path(archivo_imagen).stem + '.txt')

    # Verificar si el archivo de etiqueta existe
    if os.path.exists(label_path):
        # Leer el archivo de etiqueta
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Si el archivo de etiqueta está vacío o no contiene etiquetas, mover imagen y label
        if len(labels) == 0:
            print(f"Moviéndo {archivo_imagen} y su label correspondiente a la carpeta 'nolabels'")
            shutil.move(img_path, os.path.join(directorio_sin_label, archivo_imagen))
            shutil.move(label_path, os.path.join(directorio_sin_label, Path(label_path).name))
    else:
        # Si no existe el archivo de etiqueta, mover solo la imagen
        print(f"No se encontró el label para {archivo_imagen}, moviendo a la carpeta 'nolabels'")
        shutil.move(img_path, os.path.join(directorio_sin_label, archivo_imagen))

print("Proceso completo. Las imágenes y etiquetas sin labels han sido movidas a la carpeta 'nolabels'.")