import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('001_FeatureMatching/fotos/opencv-feature-matching-template.jpg',0)
img2 = cv2.imread('001_FeatureMatching/fotos/opencv-feature-matching-image.jpg',0)
#img1 = cv2.imread('001_FeatureMatching/fotos/wally.jpg')
#img2 = cv2.imread('001_FeatureMatching/fotos/wally-pan.jpg')
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   #BFMatcher object
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)    #matches[:10] numero de coincidencias
plt.imshow(img3)
plt.show()