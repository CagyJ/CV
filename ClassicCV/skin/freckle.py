import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def skin(src):
    '''
    Dest =Src * (1 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity ;
    '''

    dst = np.zeros_like(src)
    #int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
    v1 = 3
    v2 = 1
    dx = v1 * 5 # 双边滤波参数之一 
    fc = v1 * 12.5 # 双边滤波参数之一 
    p = 0.1
   
    temp4 = np.zeros_like(src)
    
    temp1 = cv2.bilateralFilter(src,dx,fc,fc)
    temp2 = cv2.subtract(temp1,src)
    temp2 = cv2.add(temp2, (10,10,10,128))
    temp3 = cv2.GaussianBlur(temp2,(2*v2 - 1,2*v2-1),0)
    temp4 = cv2.subtract(cv2.add(cv2.add(temp3, temp3), src), (10, 10, 10, 255))
    
    dst = cv2.addWeighted(src,p,temp4,1-p,0.0)
    dst = cv2.add(dst, (10, 10, 10,255))
    return dst


face = cv2.imread("face1.png")
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

median_face = cv2.medianBlur(face, 11)
gauss_face = cv2.GaussianBlur(face, (21, 21), 2)
bilater_face = cv2.bilateralFilter(face, 0, 49, 20)
complex_face = skin(face)

plt.subplot(151)
plt.imshow(face)
plt.subplot(152)
plt.imshow(median_face)
plt.subplot(153)
plt.imshow(gauss_face)
plt.subplot(154)
plt.imshow(bilater_face)
plt.subplot(155)
plt.imshow(complex_face)
plt.show()