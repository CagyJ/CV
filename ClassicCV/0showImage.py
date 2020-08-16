import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_show_img(img, name='image'):
    cv2.imshow(name, img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


def show_color_img(img, size=(3, 3)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()



def show_gray_img(img, size=(10, 8)):
    plt.figure(figsize=size)
    plt.imshow(img, cmap='gray')
    plt.show()

def show_hsv_img(img, size=(3, 3)):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.figure(figsize=size)
    plt.imshow(img_hsv, cmap='hsv')
    plt.show()

img = cv2.imread("../images/lenna.jpg")
show_hsv_img(img)
