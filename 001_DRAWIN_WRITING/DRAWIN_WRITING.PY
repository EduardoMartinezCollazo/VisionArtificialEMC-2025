import numpy as np
import cv2

img = cv2.imread('001_DRAWIN_WRITING/fotos/bill.jpg',cv2.IMREAD_COLOR)
cv2.line(img,(0,0),(200,300),(255,255,255),15)
cv2.rectangle(img,(200,250),(300,400),(0,0,255),5)
cv2.circle(img,(247,63), 63, (0,255,0), -1)
pts = np.array([[100,50],[200,300],[100,200],[500,100]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'HOLA MUNDO!',(10,500), font, 2, (200,255,155), 20, cv2.LINE_AA)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()