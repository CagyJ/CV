import cv2
import matplotlib.pyplot as plt
import numpy as np


def crop_img(img, x1, y1, x2, y2):
    if (x1==x2 and y1 == y2):
        shape=img.shape
        return img[0:shape[0],0:shape[1]]
    return img[y1:y2, x1:x2]

def img_flip(img,op):
    new_img = cv2.flip(img,op)
    return new_img
#op:
#0    上下翻转
#1    左右翻转
#-1   上下左右翻转

def rotation(img, angle): 
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def resize_img(img, width):
    ratio = width/img.shape[1]
    dimension = (int(width), int(img.shape[0]*ratio))
    return cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)

def img_cooler(img,b_increase,r_decrease):
    B,G,R = cv2.split(img)
    b_lim = 255 - b_increase 
    B[B>b_lim] = 255
    B[B<=b_lim] = (b_increase + B[B<=b_lim]).astype(img.dtype) 
    
    r_lim = r_decrease 
    R[R<r_lim] = 0
    R[R>r_lim] = (R[R>r_lim] - r_decrease).astype(img.dtype)
    return cv2.merge((B,G,R))


def img_warmer(img, r_increase, b_decrease):
    B,G,R = cv2.split(img)
    b_lim = b_decrease 
    B[B<b_lim] = 0
    B[B>b_lim] = (B[B>b_lim] - b_decrease).astype(img.dtype) 
    
    r_lim = 255 - r_increase 
    R[R>r_lim] = 255
    R[R<=r_lim] = (R[R<=r_lim] + r_increase).astype(img.dtype)
    return cv2.merge((B,G,R))


def adjust_gamma(img,gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i/255.0)**invGamma)*255) 
    table = np.array(table).astype('uint8')
    return cv2.LUT(img,table)


def color_shift(img, b, g, r):
    B,G,R = cv2.split(img)

    if(b>0):
        b_lim = 255 - b
        B[B>b_lim] = 255
        B[B<=b_lim] = (B[B<=b_lim] + b).astype(img.dtype)
    elif(b<0):
        b_lim = abs(b)
        B[B<=b_lim] = 0
        B[B>b_lim] = (B[B>b_lim] - b).astype(img.dtype)
    else:
        pass

    if(g>0):
        g_lim = 255 - g
        G[G>g_lim] = 255
        G[G<=g_lim] = (G[G<=g_lim] + g).astype(img.dtype)
    elif(g<0):
        g_lim = abs(g)
        G[G<=g_lim] = 0
        G[G>g_lim] = (G[G>g_lim] - g).astype(img.dtype)
    else:
        pass

    if(r>0):
        r_lim = 255 - r
        R[R>r_lim] = 255
        R[R<=r_lim] = (R[R<=r_lim] + r).astype(img.dtype)
    elif(r<0):
        r_lim = abs(r)
        R[R<=r_lim] = 0
        R[R>r_lim] = (R[R>r_lim] - r).astype(img.dtype)
    else:
        pass
    
    return cv2.merge((B,G,R))