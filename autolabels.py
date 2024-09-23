import torch
import cv2
import os
from pathlib import Path
from ultralytics import YOLO  # YOLOv8

# Cargar el modelo YOLOv10 preentrenado
model = YOLO('cs2_yolov10s.pt')  # Reemplaza con la ruta a tu modelo YOLOv8

# Ruta a las imágenes no etiquetadas
directorio_imagenes = 'data/train/images/'

# Directorio de salida labels
directorio_salida = 'data/train/labels/'
os.makedirs(directorio_salida, exist_ok=True)


def guardar_labels_yolo(imagen, predicciones, directorio_salida):
    nombre_imagen = Path(imagen).stem
    with open(f"{directorio_salida}/{nombre_imagen}.txt", 'w') as f:
        for x1, y1, x2, y2, conf, clase in predicciones:
            if conf > 0.7:  # Solo guardar predicciones con alta confianza
                # Convertir las coordenadas a formato YOLO (normalizado)
                x_centro = (x1 + x2) / 2.0
                y_centro = (y1 + y2) / 2.0
                ancho = x2 - x1
                alto = y2 - y1

                # Normalizar usando el tamaño de la imagen
                img = cv2.imread(imagen)
                img_width = img.shape[1]
                img_height = img.shape[0]
                x_centro /= img_width
                y_centro /= img_height
                ancho /= img_width
                alto /= img_height

                # Guardar en formato YOLO
                f.write(f"{int(clase)} {x_centro} {y_centro} {ancho} {alto}\n")


# Realizar inferencia en todas las imágenes no etiquetadas
for archivo_imagen in os.listdir(directorio_imagenes):
    img_path = os.path.join(directorio_imagenes, archivo_imagen)
    img = cv2.imread(img_path)

    # Realizar inferencia con el modelo YOLOv8
    resultados = model(img)

    # Extraer los valores de las cajas, confianza y clases
    boxes = resultados[0].boxes  # Obtenemos las cajas de predicción
    predicciones = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Coordenadas de la caja
        conf = box.conf.cpu().numpy()[0]  # Confianza de la predicción
        clase = box.cls.cpu().numpy()[0]  # Clase predicha
        predicciones.append([x1, y1, x2, y2, conf, clase])

    # Guardar las pseudo-labels en formato YOLO
    guardar_labels_yolo(img_path, predicciones, directorio_salida)

print(f"Pseudo-labels generados y guardados en {directorio_salida}")