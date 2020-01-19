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


def show_gray_show(img, size=(10, 8)):
    plt.figure(figsize=size)
    plt.imshow(img, cmap='gray')
    plt.show()


