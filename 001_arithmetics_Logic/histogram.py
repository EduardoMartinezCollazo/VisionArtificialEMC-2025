import cv2 
# Importa la biblioteca OpenCV para procesamiento de imágenes.

import numpy as np
# Importa NumPy para operaciones numéricas, especialmente con arrays.

import matplotlib.pyplot as plt
# Importa la librería Matplotlib para generar gráficas y visualizaciones.

# Cargar las imágenes
img1 = cv2.imread('001_arithmetics_Logic/fotos/bill.jpg')
# Lee la imagen 'bill.jpg' y la almacena en img1.

img2 = cv2.imread('001_arithmetics_Logic/fotos/gravy.jpg')
# Lee la imagen 'gravy.jpg' y la almacena en img2.

img3 = cv2.imread('001_arithmetics_Logic/fotos/py.png')
img4 = cv2.imread('001_arithmetics_Logic/fotos/py.png')
# Lee dos veces la imagen 'py.png'.

add = img3 + img4
# Realiza la suma de las dos imágenes, píxel a píxel.

# Superponer img2 sobre img1
rows, cols, channels = img2.shape
# Obtiene el número de filas, columnas y canales de img2.

roi = img1[0:rows, 0:cols ]
# Define la región de interés (ROI) en img1 del mismo tamaño que img2.

# Crear máscaras
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Convierte img2 a escala de grises.

ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
# Aplica una umbralización binaria inversa para crear una máscara de las áreas oscuras.

mask_inv = cv2.bitwise_not(mask)
# Invierte la máscara para seleccionar las zonas claras.

# Preparar regiones para la combinación
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# Aplica la máscara invertida sobre la región de fondo de img1.

img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
# Aplica la máscara original sobre img2 para extraer la región de primer plano.

dst = cv2.add(img1_bg, img2_fg)
# Suma las dos regiones para formar la imagen combinada.

img1[0:rows, 0:cols] = dst
# Inserta la región combinada de nuevo en img1.

# Convertir la imagen final a escala de grises para histogramas
gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# Convierte la imagen combinada final a escala de grises.

# Calcular histogramas
hist_original = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
# Calcula el histograma de la imagen original en escala de grises.

img_eq = cv2.equalizeHist(gray_img)
# Aplica ecualización de histograma para mejorar el contraste.

hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
# Calcula el histograma de la imagen ecualizada.

# Mostrar los resultados en una ventana con Matplotlib
plt.figure(figsize=(10, 6))
# Crea una figura de 10x6 pulgadas para contener las subgráficas.

plt.subplot(2, 2, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Imagen Original (Gris)')
plt.axis('off')
# Muestra la imagen original en grises.

plt.subplot(2, 2, 2)
plt.plot(hist_original, color='blue')
plt.title('Histograma Original')
plt.xlim([0, 256])
# Muestra el histograma correspondiente a la imagen original.

plt.subplot(2, 2, 3)
plt.imshow(img_eq, cmap='gray')
plt.title('Imagen Ecualizada')
plt.axis('off')
# Muestra la imagen luego de la ecualización.

plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='green')
plt.title('Histograma Ecualizado')
plt.xlim([0, 256])
# Muestra el histograma de la imagen ecualizada.

plt.tight_layout()
# Ajusta el diseño para que no se sobrepongan los elementos.

plt.show()
# Muestra todas las gráficas en una sola ventana.