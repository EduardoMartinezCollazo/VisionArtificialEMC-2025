import numpy as np
import cv2

# --- CONFIGURACIÓN ---
# 1. Ruta a tu imagen de plantilla (ej. el reloj)
# Asegúrate de que esta imagen esté en la misma carpeta que tu script o proporciona la ruta completa.
template_path = '001_MOGandORB/recursos/reloj.jpeg' # ¡CAMBIA ESTO A LA RUTA DE TU IMAGEN DE RELOJ!

# 2. Configuración de la captura de video
# 0 para la webcam predeterminada, o la ruta a un archivo de video como 'videos/people-walking.mp4'
video_source = 0
# video_source = '001_MOGBackground/videos/people-walking.mp4' # Para usar un archivo de video

# Umbral de coincidencias para considerar que el objeto fue encontrado
# Un número mayor significa que se requieren más puntos de coincidencia para la detección.
MIN_MATCH_COUNT = 10 # Puedes ajustar este valor

# --- INICIALIZACIÓN ---

# Cargar la imagen de la plantilla en escala de grises
img_template = cv2.imread(template_path, 0)

if img_template is None:
    print(f"Error: No se pudo cargar la imagen de la plantilla desde '{template_path}'. Verifica la ruta.")
    exit()

# Inicializar el detector y descriptor ORB
orb = cv2.ORB_create(
    nfeatures=1000, # Número máximo de características a detectar
    scaleFactor=1.2, # Factor de escala para la pirámide de imágenes
    nlevels=8 # Número de niveles de la pirámide
)

# Detectar y computar keypoints y descriptores para la plantilla
kp_template, des_template = orb.detectAndCompute(img_template, None)

# Si la plantilla no tiene suficientes keypoints, no podemos hacer matching
if des_template is None or len(kp_template) < MIN_MATCH_COUNT:
    print("Error: La imagen de la plantilla no tiene suficientes keypoints detectables.")
    print("Intenta usar una imagen con más textura o ajusta los parámetros de ORB.")
    exit()

# Crear el emparejador de fuerza bruta (BFMatcher) con validación cruzada
# NORM_HAMMING es adecuado para descriptores binarios como ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Inicializar la captura de video
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: No se pudo abrir la fuente de video. Verifica la webcam o la ruta del archivo.")
    exit()

# Crear el objeto para la sustracción de fondo MOG2
# Es adaptable a cambios lentos en la iluminación.
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,  # Número de fotogramas anteriores para construir el modelo de fondo
    varThreshold=16, # Umbral de varianza para la detección de primer plano
    detectShadows=True # Detectar y marcar sombras (0.5 en la máscara)
)

print(f"Buscando '{template_path}' en el video en vivo. Presiona 'Esc' para salir.")
print("Ventana 'Original Frame': Muestra el video con las detecciones.")
print("Ventana 'Foreground Mask': Muestra la máscara de movimiento (blanco: movimiento, negro: fondo, gris: sombra).")
print("Ventana 'Foreground Only': Muestra solo los objetos en movimiento sobre fondo negro.")

# --- BUCLE PRINCIPAL DE PROCESAMIENTO DE VIDEO ---
while True:
    ret, frame = cap.read()

    # Si no se pudo leer el fotograma (fin del video o error), salimos del bucle
    if not ret:
        print("Fin del video o error de lectura. Saliendo...")
        break

    # --- PASO 1: Sustracción de Fondo MOG2 ---
    # Aplica la sustracción de fondo al fotograma actual
    # fgmask: Máscara binaria donde 255 es primer plano, 0 es fondo, y a veces 127 para sombras.
    fgmask = fgbg.apply(frame)

    # --- Opcional: Eliminar sombras de la máscara de movimiento ---
    # Convertimos los píxeles de sombra (127) a fondo (0) para un primer plano más limpio.
    ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    # --- PASO 2: Aislar el Primer Plano (reducción de fondo para Feature Matching) ---
    # Creamos una imagen donde solo el primer plano detectado por MOG2 es visible.
    # El fondo se vuelve negro. Aquí es donde se aplica la "reducción de fondo".
    foreground_only_frame = cv2.bitwise_and(frame, frame, mask=fgmask)

    # --- PASO 3: Detección de Similitudes (Feature Matching con ORB) ---
    # Convertir el primer plano aislado a escala de grises para ORB
    # Aunque ya es casi "negro y el objeto", ORB necesita un solo canal.
    gray_fg_only = cv2.cvtColor(foreground_only_frame, cv2.COLOR_BGR2GRAY)

    # Detectar y computar keypoints y descriptores en el frame actual (solo en el primer plano)
    kp_frame, des_frame = orb.detectAndCompute(gray_fg_only, None)

    # Si se detectaron descriptores en el frame actual:
    if des_frame is not None and len(kp_frame) >= MIN_MATCH_COUNT:
        # Realizar el emparejamiento de descriptores
        matches = bf.match(des_template, des_frame)

        # Ordenar las coincidencias por su distancia (las más pequeñas son las mejores)
        matches = sorted(matches, key = lambda x:x.distance)

        # --- Verificación de suficientes buenas coincidencias ---
        # Si tenemos un número suficiente de buenas coincidencias, se considera el objeto encontrado.
        if len(matches) > MIN_MATCH_COUNT:
            # Extraer los keypoints correspondientes a las mejores coincidencias
            src_pts = np.float32([ kp_template[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

            # Encontrar la matriz de homografía (transformación que alinea los puntos)
            # RANSAC es robusto a valores atípicos (outliers)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Si se encuentra una homografía válida:
            if M is not None:
                # Obtener las dimensiones de la plantilla
                h,w = img_template.shape

                # Definir las 4 esquinas de la plantilla
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                
                # Transformar las esquinas de la plantilla a la perspectiva del frame actual
                dst = cv2.perspectiveTransform(pts,M)

                # Dibujar un polígono alrededor del objeto detectado en el frame original
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0,255,255), 3, cv2.LINE_AA)
                
                # Opcional: También puedes dibujar las 10 mejores coincidencias en una imagen separada
                # (Esto puede ralentizar un poco si tienes muchos keypoints/matches)
                # img_matches = cv2.drawMatches(img_template, kp_template, frame, kp_frame, matches[:MIN_MATCH_COUNT], None, flags=2)
                # cv2.imshow('Matches Found', img_matches)
            else:
                # Si no se encontró una homografía robusta
                # (Esto puede ocurrir si las coincidencias son insuficientes o muy ruidosas)
                pass # No hacer nada, no dibujar nada
        # else:
            # No hay suficientes buenas coincidencias para considerar que el objeto fue encontrado
            # print("No suficientes coincidencias para encontrar el objeto.")
    
    # --- VISUALIZACIÓN DE RESULTADOS EN TIEMPO REAL ---
    cv2.imshow('Original Frame (with Detections)', frame)      # Fotograma original con el objeto resaltado
    cv2.imshow('Foreground Mask', fgmask)                      # Máscara de movimiento
    cv2.imshow('Foreground Only', foreground_only_frame)       # Solo el primer plano (fondo negro)

    # --- CONTROL DE SALIDA ---
    # Espera 30ms para cada fotograma y captura la tecla presionada
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Si se presiona 'Esc' (ASCII 27)
        break

# --- LIBERAR RECURSOS AL FINALIZAR ---
cap.release()
cv2.destroyAllWindows()