import numpy as np
import cv2

# --- 1. Inicializar la captura de video desde la webcam ---
# El '0' se refiere a la cámara predeterminada del sistema.
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara. Asegúrate de que no esté en uso y esté conectada.")
    exit() # Salir del programa si la cámara no está disponible

print("Cámara iniciada. Presiona 'Esc' para salir.")

# --- 2. Bucle principal para procesar fotogramas de la webcam ---
while True:
    # Leer un fotograma de la cámara.
    # 'ret' es True si el fotograma se leyó correctamente, 'frame' es el fotograma.
    ret, frame = cap.read()

    # Si no se pudo leer el fotograma, salimos del bucle.
    if not ret:
        print("No se pudo leer el fotograma. Finalizando...")
        break

    # --- Aquí 'img' ahora será el 'frame' actual de la webcam ---
    # La variable 'img' ahora contiene el fotograma de la cámara en cada iteración.
    img = frame.copy() # Hacemos una copia para dibujar en ella y mantener el 'frame' original si es necesario.

    # Convierte el fotograma (img) a escala de grises.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convierte la imagen gris a tipo flotante, requerido por goodFeaturesToTrack.
    gray = np.float32(gray)
    
    # Detecta las esquinas usando el algoritmo Shi-Tomasi.
    # - 100: Número máximo de esquinas.
    # - 0.01: Calidad mínima de la esquina.
    # - 2: Distancia euclidiana mínima entre esquinas.
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 2)
    
    # Si se detectaron esquinas, las procesamos y dibujamos.
    if corners is not None:
        # Convierte las coordenadas de las esquinas a enteros.
        corners = np.int0(corners)
        
        # Dibuja un círculo en cada esquina detectada.
        for corner in corners:
            x, y = corner.ravel() # Obtiene las coordenadas x, y del array de esquina
            # Dibuja un círculo rojo (0,0,255 en BGR) relleno (-1) en cada esquina.
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1) 
    
    # Muestra el fotograma con las esquinas dibujadas en una ventana.
    cv2.imshow('Corner Detection (Webcam)', img)

    # --- Control de salida: Esperar por la tecla 'Esc' ---
    # Espera 1 milisegundo por una pulsación de tecla para mantener el video fluido.
    # Si se presiona la tecla 'Esc' (ASCII 27), el bucle se rompe.
    k = cv2.waitKey(1) & 0xFF
    if k == 27: # 27 es el código ASCII para la tecla 'Esc'
        break

# --- Liberar recursos al salir del bucle ---
cap.release() # Libera el objeto de captura de la cámara
cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV