import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io 
from skimage import transform as tf 

# similarity transform
def img_similarityTransform(img,translation,rotation,scale):
    rows, cols = img_ori.shape[:2]
    M_translation = np.float32([[1,0,translation[0]],[0,1,translation[1]]])
    translation_img = cv2.warpAffine(img,M_translation,(cols,rows))
    M = cv2.getRotationMatrix2D((rows/2,cols/2),rotation,scale)
    new_img = cv2.warpAffine(translation_img,M,(cols,rows))
    return new_img

def st(img, scale=1.8, rotation = np.deg2rad(9), translation=(-3,-100)):
    tform = tf.SimilarityTransform(scale, rotation, translation)
    st_img = tf.warp(img, tform)
    io.imshow(st_img)


#affine transform
rows, cols, ch = img_ori.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.7, rows * 0.2], [cols * 0.1, rows * 0.9]])
 
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img_ori, M, (cols, rows))

def img_affineTransform(img,pts1,pts2):
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img,M,(img.shape[0],img.shape[1]))


#perspective transform
pts1 = np.float32([[0,0],[0,500],[500,0],[500,500]]) 
pts2 = np.float32([[5,19],[19,460],[460,7],[410,420]])

M = cv2.getPerspectiveTransform(pts1,pts2) 
img_wart = cv2.warpPerspective(img_ori,M,(500,500)) 

def img_perspectiveTransform(img, pts1, pts2, size=(500,500)):
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,size)


#only for black-white image
#erode
erode_writing = cv2.erode(img_writing,None,iterations=2)


#dilate
dilate_img = cv2.dilate(img_writing,None,iterations=1)