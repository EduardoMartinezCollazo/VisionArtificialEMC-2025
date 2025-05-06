import numpy as np # Importa la librería NumPy para operaciones numéricas eficientes.
import cv2 # Importa la librería OpenCV (cv2) para el procesamiento de imágenes.

img = cv2.imread('001_DRAWIN_WRITING/fotos/bill.jpg',cv2.IMREAD_COLOR)
cv2.line(img,(0,0),(200,300),(0,155,0),15)
# Dibuja una línea blanca gruesa (15 píxeles) desde la coordenada (0,0) hasta (200,300) sobre la imagen 'img'.
cv2.rectangle(img,(200,250),(300,400),(0,0,255),5)
# Dibuja un rectángulo rojo con un grosor de borde de 5 píxeles, con la esquina superior izquierda en (200,250) y la esquina inferior derecha en (300,400) sobre 'img'.
cv2.circle(img,(247,63), 63, (0,255,0), -1)
# Dibuja un círculo verde relleno (el -1 indica relleno) con centro en (247,63) y radio de 63 píxeles sobre 'img'.
pts = np.array([[100,50],[200,300],[100,200],[500,100]], np.int32)
# Crea un array NumPy de puntos (vértices) con coordenadas enteras para definir un polígono.
pts = pts.reshape((-1,1,2))
# Remodela el arreglo de puntos para que tenga la forma requerida por la función polylines (una lista de contornos).
# Siendo el -1 la deteccion automatica del tamaño del arreglo
cv2.polylines(img, [pts], True, (0,255,255), 3)
# Dibuja un polígono amarillo (True indica que se cierra la figura uniendo el último punto con el primero) con un grosor de línea de 3 píxeles utilizando los puntos definidos en 'pts' sobre 'img'.
font = cv2.FONT_HERSHEY_SIMPLEX
# Define la fuente de texto que se utilizará (en este caso, una fuente simple).
cv2.putText(img,'HOLA MUNDO!',(10,500), font, 2, (200,255,155), 20, cv2.LINE_AA)
# Escribe el texto 'HOLA MUNDO!' en la imagen 'img' en la posición (10,500), usando la fuente definida, con un tamaño de 2, color verde claro y un grosor de 20 píxeles con antialiasing.
cv2.imshow('image',img)
# Muestra la imagen modificada en una ventana con el título 'image'.
cv2.waitKey(0)
# Espera indefinidamente a que se presione una tecla.
cv2.destroyAllWindows()
# Cierra todas las ventanas de OpenCV creadas.