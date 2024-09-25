import cv2
import os
from pathlib import Path
import numpy as np
import random

# Función para rotar la imagen
def rotar_imagen(imagen, angulo):
    (alto, ancho) = imagen.shape[:2]
    centro = (ancho // 2, alto // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (ancho, alto))
    return imagen_rotada

# Función para ajustar el brillo de la imagen
def ajustar_brillo(imagen, valor_brillo=30):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)  # Convertir la imagen a HSV
    h, s, v = cv2.split(hsv)
    # Aumentar o disminuir el valor del brillo
    v = cv2.add(v, valor_brillo)
    v[v > 255] = 255  # Limitar el valor máximo
    v[v < 0] = 0  # Limitar el valor mínimo
    imagen_brillante = cv2.merge((h, s, v))
    imagen_brillante = cv2.cvtColor(imagen_brillante, cv2.COLOR_HSV2BGR)
    return imagen_brillante

# Función para ajustar el contraste de la imagen
def ajustar_contraste(imagen, factor_contraste=1.2):
    # Multiplicar los valores de los píxeles por el factor de contraste
    imagen_contraste = cv2.convertScaleAbs(imagen, alpha=factor_contraste, beta=0)
    return imagen_contraste

# Función para escalar la imagen
def escalar_imagen(imagen, factor_escala=1.2):
    return cv2.resize(imagen, None, fx=factor_escala, fy=factor_escala, interpolation=cv2.INTER_LINEAR)

# Funciones para las transformaciones
def rotar_imagen(imagen, angulo):
    (alto, ancho) = imagen.shape[:2]
    centro = (ancho // 2, alto // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(imagen, matriz_rotacion, (ancho, alto))

def ajustar_brillo(imagen, valor_brillo=30):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, valor_brillo)
    v[v > 255] = 255
    v[v < 0] = 0
    imagen_brillante = cv2.merge((h, s, v))
    return cv2.cvtColor(imagen_brillante, cv2.COLOR_HSV2BGR)

def ajustar_contraste(imagen, factor_contraste=1.2):
    return cv2.convertScaleAbs(imagen, alpha=factor_contraste, beta=0)

def aplicar_gaussian_blur(imagen, kernel_size=(5,5)):
    return cv2.GaussianBlur(imagen, kernel_size, 0)

def agregar_ruido_gaussiano(imagen, mean=0, var=0.1):
    row, col, ch = imagen.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    return np.clip(imagen + gauss.reshape(row, col, ch), 0, 255).astype(np.uint8)

def transformar_perspectiva(imagen):
    altura, ancho = imagen.shape[:2]
    pts1 = np.float32([[0,0], [ancho,0], [0,altura], [ancho,altura]])
    pts2 = np.float32([[50,50], [ancho-50,50], [50, altura-50], [ancho-50, altura-50]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(imagen, M, (ancho, altura))

# Rotación con un rango de ángulos completo
def rotar_imagen_extremo(imagen):
    angulo = random.uniform(0, 360)  # Rotación extrema en cualquier ángulo
    (alto, ancho) = imagen.shape[:2]
    centro = (ancho // 2, alto // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(imagen, matriz_rotacion, (ancho, alto))

# Ajuste de brillo extremo
def ajustar_brillo_extremo(imagen, valor_brillo=100):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, valor_brillo)
    v[v > 255] = 255
    v[v < 0] = 0
    imagen_brillante = cv2.merge((h, s, v))
    return cv2.cvtColor(imagen_brillante, cv2.COLOR_HSV2BGR)

# Ajuste de contraste extremo
def ajustar_contraste_extremo(imagen, factor_contraste=2.0):
    return cv2.convertScaleAbs(imagen, alpha=factor_contraste, beta=0)

# Desenfoque gaussiano extremo
def aplicar_gaussian_blur_extremo(imagen, kernel_size=(15, 15)):
    return cv2.GaussianBlur(imagen, kernel_size, 0)

# Ruido gaussiano extremo
def agregar_ruido_gaussiano_extremo(imagen, mean=0, var=0.5):
    row, col, ch = imagen.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    return np.clip(imagen + gauss.reshape(row, col, ch), 0, 255).astype(np.uint8)

# Transformación de perspectiva extrema
def transformar_perspectiva_extremo(imagen):
    altura, ancho = imagen.shape[:2]
    margen = int(min(ancho, altura) * 0.3)  # Aumentar la distorsión
    pts1 = np.float32([[0, 0], [ancho, 0], [0, altura], [ancho, altura]])
    pts2 = np.float32([[margen, margen], [ancho - margen, margen], [margen, altura - margen], [ancho - margen, altura - margen]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(imagen, M, (ancho, altura))


# Lista de directorios a procesar
directorios = [('data/train/images', 'data/train/imagesOPENCV'),
              ('data/valid/images', 'data/valid/imagesOPENCV'),
              ('data/test/images', 'data/test/imagesOPENCV')]

# Procesar cada directorio
for directorio_imagenes, directorio_modificado in directorios:
    # Crear la carpeta de salida si no existe
    os.makedirs(directorio_modificado, exist_ok=True)

    # Procesar todas las imágenes del directorio
    for archivo_imagen in os.listdir(directorio_imagenes):
        img_path = os.path.join(directorio_imagenes, archivo_imagen)
        imagen = cv2.imread(img_path)
        if imagen is None:
            print(f"No se pudo cargar la imagen {archivo_imagen}.")
            continue
        print(f"Procesando imagen: {archivo_imagen}")

        # Rotaciones
        for angulo in [35, 65, 90, 180, 270]:
            imagen_rotada = rotar_imagen(imagen, angulo)
            output_path = os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_rot{angulo}.jpg")
            cv2.imwrite(output_path, imagen_rotada)

        # Volteo
        imagen_volteada_h = cv2.flip(imagen, 1)  # Volteo horizontal
        imagen_volteada_v = cv2.flip(imagen, 0)  # Volteo vertical
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_flipH.jpg"), imagen_volteada_h)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_flipV.jpg"), imagen_volteada_v)

        # Ajuste de brillo
        imagen_brillo_aumentado = ajustar_brillo(imagen, 85)
        imagen_brillo_aumentado_2 = ajustar_brillo(imagen, 50)
        imagen_brillo_reducido = ajustar_brillo(imagen, -50)
        imagen_brillo_reducido_2 = ajustar_brillo(imagen, -25)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_brilloAumentado.jpg"), imagen_brillo_aumentado)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_brilloAumentado_2.jpg"), imagen_brillo_aumentado_2)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_brilloReducido.jpg"), imagen_brillo_reducido)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_brilloReducido_2.jpg"), imagen_brillo_reducido_2)

        # Ajuste de contraste
        imagen_contraste_aumentado = ajustar_contraste(imagen, 1.9)
        imagen_contraste_aumentado_2 = ajustar_contraste(imagen, 1.35)
        imagen_contraste_reducido = ajustar_contraste(imagen, 0.7)
        imagen_contraste_reducido_2 = ajustar_contraste(imagen, 0.45)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_contrasteAumentado.jpg"), imagen_contraste_aumentado)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_contrasteAumentado_2.jpg"), imagen_contraste_aumentado_2)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_contrasteReducido.jpg"), imagen_contraste_reducido)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_contrasteReducido_2.jpg"), imagen_contraste_reducido_2)

        # Escalado
        imagen_escalada = escalar_imagen(imagen, 1.5)
        imagen_escalada_2 = escalar_imagen(imagen, 2.5)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_escalado.jpg"), imagen_escalada)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_escalado_2.jpg"), imagen_escalada_2)

        # Aplicar transformaciones extremas
        imagen_rotada_extremo = rotar_imagen_extremo(imagen)
        imagen_brillo_extremo = ajustar_brillo_extremo(imagen, 100)  # Aumentar brillo en 100
        imagen_contraste_extremo = ajustar_contraste_extremo(imagen, 2.0)  # Duplicar el contraste
        imagen_blur_extremo = aplicar_gaussian_blur_extremo(imagen)  # Desenfoque extremo con un kernel mayor
        imagen_ruido_extremo = agregar_ruido_gaussiano_extremo(imagen)  # Ruido extremo con alta varianza
        imagen_perspectiva_extremo = transformar_perspectiva_extremo(imagen)  # Perspectiva extrema
        imagen_rotada = rotar_imagen(imagen, random.uniform(0, 360))
        imagen_blur = aplicar_gaussian_blur(imagen)
        imagen_ruido = agregar_ruido_gaussiano(imagen)
        imagen_perspectiva = transformar_perspectiva(imagen)

        # Guardar imágenes modificadas
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_rotada_extrema.jpg"), imagen_rotada_extremo)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_brillo_extremo.jpg"), imagen_brillo_extremo)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_contraste_extremo.jpg"), imagen_contraste_extremo)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_blur_extremo.jpg"), imagen_blur_extremo)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_ruido_extremo.jpg"), imagen_ruido_extremo)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_perspectiva_extrema.jpg"), imagen_perspectiva_extremo)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_rotada.jpg"), imagen_rotada)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_blur.jpg"), imagen_blur)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_ruido.jpg"), imagen_ruido)
        cv2.imwrite(os.path.join(directorio_modificado, f"{Path(archivo_imagen).stem}_perspectiva.jpg"), imagen_perspectiva)
        print("YOLO! (¡Guardada con éxito!)")

print("EXITO.")