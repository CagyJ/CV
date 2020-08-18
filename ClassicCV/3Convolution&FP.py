import cv2
import matplotlib.pyplot as plt
import numpy as np

#image convolution

# first-order derivative: prewitt operator and sobel operator
def prewitt_x(img):
    x_kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    x_img = cv2.filter2D(img,-1,x_kernel)
    return x_img    


def prewitt_y(img):
    y_kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    y_img = cv2.filter2D(img,-1,y_kernel)
    return y_img


def sobel_x(img):
    x_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    x_img = cv2.filter2D(img,-1,x_kernel)
    return x_img


def sobel_y(img):
    y_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    y_img = cv2.filter2D(img,-1,y_kernel)
    return y_img


# second-order derivative: laplacian operator

def laplacian_weak(img):
    kernel_weak = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    lap_img = cv2.filter2D(img,-1,kernel_weak)
    return lap_img


def laplacian_strong(img):
    kernel_strong = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    lap_img = cv2.filter2D(img,-1,kernel_strong)
    return lap_img


#Gaussian Kernel (Blur)
g_img = cv2.GaussianBlur(img,(11,11),2) #(11,11) is kernel and 2 is variance
#way 2
def gauss_blur(img,kernel,var):
    kernel = cv2.getGaussianKernel(11,200)
    g_img = cv2.sepFilter2D(img,-1,kernel,kernel)
    return g_img

# Image sharpening
def img_sharpen(img,mod):
    if(mod == 0):
        kernel_rui = np.array([[1,1,1],[1,-7,1],[1,1,1]])
    else:
        kernel_rui = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    
    lap_rui_img = cv2.filter2D(img,-1,kernel_rui)

# Median Blur: more comfortable than gaussian blur and can reduce salt&pepper
md_img = cv2.medianBlur(img,7)


# Harris Corner
def fp_out(src,thresh_p): #thresh percent: like 0.03
    img = cv2.imread(src)
    gray = np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    harris = cv2.cornerHarris(gray,2,3,0.03)
    harris = cv2.dilate(harris,None)
    threshold = np.max(harris)*thresh_p #域值越大，特征点变少
    img[harris>threshold] = [0,0,255]
    return img


# SIFT(opencv before 3.4.2)
def sift(img,mod=0):
    s = cv2.xfeatures2d.SIFT_create()
    kp = s.detect(img)
    kp, des = s.compute(img,kp)
    if(mod==0):
        f = cv2.DRAW_MATCHES_FLAGS_DEFAULT
    else:
        f = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    img_sift = cv2.drawKeypoints(img,kp,outImage=np.array([]),flags=f)
    return img_sift