import numpy as np
import cv2

img = cv2.imread('001_Image_Operations/fotos/bill.jpg', cv2.IMREAD_COLOR)

img [55,55] = [255,255,255]
px = img[55,55]

img[100:250, 100:250] = [255,255,255]

bill_sec = img[37:111, 107:194]
img[0:74, 0:87] = bill_sec

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
