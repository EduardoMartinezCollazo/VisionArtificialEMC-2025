import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
img = cv2.imread('001_arithmetics_Logic/fotos/bill.jpg')

# Convertir la imagen a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calcular el histograma de la imagen original
hist_original = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

# Aplicar ecualización de histograma
img_eq = cv2.equalizeHist(gray_img)

# Calcular el histograma de la imagen ecualizada
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# Mostrar los resultados en una sola ventana con Matplotlib
plt.figure(figsize=(12, 6))

# Subplot 1: Imagen Original (Gris)
plt.subplot(1, 4, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original (Gris)')
plt.axis('off')

# Subplot 2: Histograma Original
plt.subplot(1, 4, 2)
plt.plot(hist_original, color='blue')
plt.title('Histograma Original')
plt.xlabel('Nivel de Intensidad')
plt.ylabel('Frecuencia')
plt.xlim([0, 256])

# Subplot 3: Imagen Ecualizada
plt.subplot(1, 4, 3)
plt.imshow(img_eq, cmap='gray')
plt.title('Ecualizada')
plt.axis('off')

# Subplot 4: Histograma Ecualizado
plt.subplot(1, 4, 4)
plt.plot(hist_eq, color='green')
plt.title('Histograma Ecualizado')
plt.xlabel('Nivel de Intensidad')
plt.ylabel('Frecuencia')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()