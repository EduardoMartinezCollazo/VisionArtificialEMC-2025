import cv2
import numpy as np

# Importa la librería OpenCV para el procesamiento de imágenes.
# Importa la librería NumPy para operaciones numéricas eficientes.

cap = cv2.VideoCapture(0)
# Inicializa la captura de video desde la cámara predeterminada (índice 0).

#hsv - 

while(1):

    _, frame = cap.read()
    # Lee un fotograma de la cámara. '_' descarta el valor de retorno (booleano indicando éxito), 'frame' contiene el fotograma capturado.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Convierte el fotograma del espacio de color BGR (predeterminado de OpenCV) al espacio de color HSV (Hue, Saturation, Value).

    lower_red = np.array([0,150,150])
    upper_red = np.array([20,250,250])
    # Define los límites inferior y superior para el color rojo en el espacio de color HSV.

    mask = cv2.inRange(hsv, lower_red, upper_red)
    # Crea una máscara binaria. Los píxeles de la imagen HSV que caen dentro del rango definido por lower_red y upper_red se establecen en blanco (255), y los demás en negro (0).
    res = cv2.bitwise_and(frame,frame, mask= mask)
    # Realiza una operación bitwise AND entre el fotograma original y sí mismo, utilizando la máscara. Esto resulta en mostrar solo las regiones del fotograma original donde la máscara es blanca.

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    # Muestra el resultado de la operación bitwise AND (solo las regiones rojas detectadas) .

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    # Si la tecla presionada es Escape, sale del bucle.

cv2.destroyAllWindows()
# Cierra todas las ventanas de OpenCV creadas.
cap.release()
# Libera los recursos de la cámara.