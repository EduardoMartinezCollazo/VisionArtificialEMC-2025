import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 2)
    corners = np.int0(corners)
    for corner in corners:
        x,y = corner.ravel()
    cv2.circle(frame,(x,y),3,255,-1)    
    cv2.imshow('Corner',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break