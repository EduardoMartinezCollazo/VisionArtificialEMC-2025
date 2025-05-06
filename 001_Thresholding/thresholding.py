import numpy
# Importa la librería NumPy para operaciones numéricas eficientes.
import matplotlib
# Importa la librería Matplotlib (aunque no se usa directamente en este código).
import cv2
# Importa la librería OpenCV (cv2) para el procesamiento de imágenes.

img = cv2.imread('001_Thresholding/fotos/bookpage.jpg')
# Lee una imagen y la almacena en la variable 'img'.
retval, threshold = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
# Aplica un umbral global binario a la imagen 'img'. Los píxeles con valor menor que 12 se establecen en 0 (negro),
#  y los mayores se establecen en 255 (blanco). 'retval' contiene el valor del umbral utilizado.
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Convierte la imagen a escala de grises y la almacena en 'grayscaled'.
retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)
# Aplica un umbral global binario a la imagen en escala de grises 'grayscaled'. Los píxeles con valor menor que 12 se 
# establecen en 0, y los mayores en 255.
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2. ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# Aplica un umbral adaptativo gaussiano a la imagen en escala de grises. El valor del píxel se compara con un 
# promedio ponderado de su vecindad (tamaño de bloque 115x115), y se le resta una constante (1).
retval3, otsu = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Aplica un umbral global binario a la imagen en escala de grises, utilizando el método de Otsu para 
# determinar automáticamente el valor óptimo del umbral (el valor 125 aquí podría ser ignorado por Otsu).
retval2, threshold3 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.imshow('threshold2',threshold2)
cv2.imshow('gaus',gaus)
cv2.imshow('otsu',otsu)
cv2.imshow('threshold3',threshold3)

cv2.waitKey(0)
cv2.destroyAllWindows()