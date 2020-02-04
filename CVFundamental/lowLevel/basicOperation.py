import cv2
import matplotlib.pyplot as pyplot
import numpy as np


def crop_img(img, x1, y1, x2, y2):
    if (x1==x2 and y1 == y2):
        shape=img.shape
        return img[0:shape[0],0:shape[1]]
    return img[y1:y2, x1:x2]


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
    
    r_lim = r_increase 
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

