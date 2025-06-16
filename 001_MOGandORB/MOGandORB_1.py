import numpy as np
import cv2

template_path = '001_MOGandORB/recursos/reloj.jpeg'

video_source = 0

MIN_MATCH_COUNT = 10

img_template = cv2.imread(template_path, 0)

if img_template is None:
    print(f"Error: No se pudo cargar la imagen de la plantilla desde '{template_path}'. Verifica la ruta.")
    exit()

orb = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8
)

kp_template, des_template = orb.detectAndCompute(img_template, None)

if des_template is None or len(kp_template) < MIN_MATCH_COUNT:
    print("Error: La imagen de la plantilla no tiene suficientes keypoints detectables.")
    print("Intenta usar una imagen con más textura o ajusta los parámetros de ORB.")
    exit()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: No se pudo abrir la fuente de video. Verifica la webcam o la ruta del archivo.")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

print(f"Buscando '{template_path}' en el video en vivo. Presiona 'Esc' para salir.")
print("Ventana 'Original Frame': Muestra el video con las detecciones.")
print("Ventana 'Foreground Mask': Muestra la máscara de movimiento (blanco: movimiento, negro: fondo, gris: sombra).")
print("Ventana 'Foreground Only': Muestra solo los objetos en movimiento sobre fondo negro.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Fin del video o error de lectura. Saliendo...")
        break

    fgmask = fgbg.apply(frame)

    ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    foreground_only_frame = cv2.bitwise_and(frame, frame, mask=fgmask)

    gray_fg_only = cv2.cvtColor(foreground_only_frame, cv2.COLOR_BGR2GRAY)

    kp_frame, des_frame = orb.detectAndCompute(gray_fg_only, None)

    if des_frame is not None and len(kp_frame) >= MIN_MATCH_COUNT:
        matches = bf.match(des_template, des_frame)

        matches = sorted(matches, key = lambda x:x.distance)

        if len(matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp_template[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h,w = img_template.shape

                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                
                dst = cv2.perspectiveTransform(pts,M)

                frame = cv2.polylines(frame, [np.int32(dst)], True, (0,255,255), 3, cv2.LINE_AA)
            else:
                pass
    
    cv2.imshow('Original Frame (with Detections)', frame)
    cv2.imshow('Foreground Mask', fgmask)
    cv2.imshow('Foreground Only', foreground_only_frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()