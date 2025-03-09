#librerias
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('fotos/bill.jpg',cv2.IMREAD_GRAYSCALE) #asignar a la variable img los datos de bill que se encuenta en la carpeta fotos, donde se va a hacer la lectura a base de grises

#otros tipos de interpretacion de las imagenes
#IMREAD_COLOR = 1
#IMREAD_UNCHANGED = -1

#nombre de la ventana y variable que se va a mostrar con la funcion show de cv2
#cv2.imshow('holam mundo',img)
#cv2.waitKey(0)  #espera de alguna tecla 
#cv2.destroyAllWindows() #cerrar el ejecutable

#imprecion de img con interpolacion 
plt.imshow(img, cmap='plasma',interpolation='bicubic') #mostrar la imagen con una referencia de color(como si fuera un filto)
plt.plot([275,50],[280,50],'c',linewidth=5)    #incertar en la imagen una linea en las cordenadas y con line de grosor 5
plt.show()

#cv2.imwrite('fotos/bill mejorado.png',img)