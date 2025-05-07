import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen original
img_color = cv2.imread('001_arithmetics_Logic/fotos/py.png')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Ecualización del histograma
img_eq = cv2.equalizeHist(img_gray)

# Calcular histogramas
hist_orig = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# Mostrar los resultados
plt.figure(figsize=(10, 8))

# Imagen original en escala de grises
plt.subplot(2, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Imagen Original (Grises)')
plt.axis('off')

# Histograma de la imagen original
plt.subplot(2, 2, 2)
plt.plot(hist_orig, color='black')
plt.title('Histograma Original')
plt.xlim([0, 256])

# Imagen ecualizada
plt.subplot(2, 2, 3)
plt.imshow(img_eq, cmap='gray')
plt.title('Imagen Ecualizada')
plt.axis('off')

# Histograma de la imagen ecualizada
plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='black')
plt.title('Histograma Ecualizado')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
