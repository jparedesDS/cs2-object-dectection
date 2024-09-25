import os
from pathlib import Path
import subprocess
import time
import cv2

# Directorio de imágenes y etiquetas
directorio_imagenes = ('data/train/images')  # Carpeta de imágenes
directorio_labels = ('data/train/labels')  # Carpeta de etiquetas

# Diccionario para las clases (ajústalo según tus clases)
clases = {0: 'ct', 1: 'cthead', 2: 't', 3: 'thead'}  # Ajusta esto según tus clases

def visualizar_labels_en_imagen(imagen_path, label_path):
    img = cv2.imread(imagen_path)
    img_height, img_width = img.shape[:2]

    # Leer el archivo de etiquetas
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # Dibujar cada etiqueta en la imagen
    for label in labels:
        clase_id, x_centro, y_centro, ancho, alto = map(float, label.split())

        # Desnormalizar las coordenadas y dimensiones (convertir a píxeles)
        x_centro *= img_width
        y_centro *= img_height
        ancho *= img_width
        alto *= img_height

        # Obtener las esquinas del bounding box
        x1 = int(x_centro - ancho / 2)
        y1 = int(y_centro - alto / 2)
        x2 = int(x_centro + ancho / 2)
        y2 = int(y_centro + alto / 2)

        # Dibujar el bounding box en la imagen
        color = (0, 255, 0)  # Verde
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Añadir el nombre de la clase
        text = clases[int(clase_id)]
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar la imagen con los bounding boxes
    cv2.imshow("Imagen con Labels", img)
    cv2.waitKey(0)  # Presiona cualquier tecla para cerrar la ventana
    cv2.destroyAllWindows()