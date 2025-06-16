import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('001_GrabCut_Foreground/fotos/flores.jpg')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (0,0,1400,1400)  #x-horizontal desde la izq, y-vertical desde arriba, ancho, alto
#rect = (100,140,350,480)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()
