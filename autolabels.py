import os
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO  # Asegúrate de que la librería esté instalada

# Cargar tu propio modelo YOLO
model = YOLO('models/cs2_cv-woacm-noparams-yolo10m-1200epoch.pt')  # Reemplaza con tu modelo

# Ruta a las imágenes no etiquetadas:
directorio_imagenes = 'data/train/images'
# Directorio de salida para imágenes con nuevos labels
directorio_salida = 'data/train/imagesNUEVOSLABELS'
os.makedirs(directorio_salida, exist_ok=True)
# Directorio de salida para imágenes sin labels
directorio_sin_labels = 'data/train/imagesSINLABELS'
os.makedirs(directorio_sin_labels, exist_ok=True)
# Directorio de salida para etiquetas vacías
directorio_labels_sin_labels = 'data/test/labelsSINLABELS'
os.makedirs(directorio_labels_sin_labels, exist_ok=True)

# Función para guardar las pseudo-labels en formato YOLO
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

                # Guardar en formato YOLO .txt
                f.write(f"{int(clase)} {x_centro} {y_centro} {ancho} {alto}\n")

# Realizar inferencia en todas las imágenes no etiquetadas
for archivo_imagen in os.listdir(directorio_imagenes):
    img_path = os.path.join(directorio_imagenes, archivo_imagen)
    img = cv2.imread(img_path)

    # Realizar inferencia con el modelo YOLOvX
    resultados = model(img)

    # Extraer los valores de las cajas, confianza y clases
    boxes = resultados[0].boxes  # Obtenemos las cajas de predicción
    predicciones = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Coordenadas de la caja
        conf = box.conf.cpu().numpy()[0]  # Confianza de la predicción
        clase = box.cls.cpu().numpy()[0]  # Clase predicha
        predicciones.append([x1, y1, x2, y2, conf, clase])

    if len(predicciones) > 0:  # Si hay predicciones, guardamos las labels
        guardar_labels_yolo(img_path, predicciones, directorio_salida)
    else:  # Si no hay predicciones, movemos la imagen a la carpeta de imágenes sin labels
        shutil.move(img_path, os.path.join(directorio_sin_labels, archivo_imagen))

        # Crear un archivo .txt vacío y moverlo a la carpeta de labels sin labels
        nombre_imagen = Path(img_path).stem
        empty_label_path = os.path.join(directorio_labels_sin_labels, f"{nombre_imagen}.txt")
        open(empty_label_path, 'w').close()

print(f"Proceso completado. Imágenes sin labels movidas a {directorio_sin_labels}. Labels generados y guardados en {directorio_salida}.")




"""
# Directorio que contiene tanto las imágenes como las etiquetas
directorio = 'data/test/imagesSINLABELS'  # Cambia esto según tu ruta
directorio_labels = 'data/test/labelsSINLABELS2'  # Cambia esto según tu ruta

# Procesar todas las imágenes y etiquetas
for archivo_imagen in os.listdir(os.path.join(directorio,)):
    img_path = os.path.join(directorio, archivo_imagen)
    label_path = os.path.join(directorio_labels, Path(archivo_imagen).stem + '.txt')

    # Comprobar si el archivo de etiqueta está vacío
    if not os.path.exists(label_path) or os.stat(label_path).st_size == 0:
        print(f"Abrir LabelImg para etiquetar {archivo_imagen}")

        # Abrir LabelImg para esa imagen
        if os.name == 'nt':  # Windows
            subprocess.run(['labelImg', img_path])
        else:  # Linux o macOS
            subprocess.run(['python', 'labelImg.py', img_path])

        # Esperar hasta que el usuario termine de etiquetar antes de continuar
        input("Presiona Enter cuando hayas terminado de etiquetar la imagen en LabelImg...")
    else:
        print(f"Etiqueta ya existente para {archivo_imagen}.")
"""