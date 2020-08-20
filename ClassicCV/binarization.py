import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import collections

def easy_binarization(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray[img_gray>127] = 255
    img_gray[img_gray<=127] = 0
    plt.imshow(img_gray, cmap='gray')
    plt.show()
    return img_gray


def mean_binarization(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = np.mean(img_gray)
    img_gray[img_gray>threshold] = 255
    img_gray[img_gray<=threshold] = 0
    plt.imshow(img_gray, cmap='gray')
    plt.show()
    return img_gray


def hist_binarization(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = img_gray.flatten()
    plt.subplot(121)
    plt.hist(hist, 256)

    cnt_hist = collections.Counter(hist)
    begin, end = cnt_hist.most_common(2)[0][0], cnt_hist.most_common(2)[1][0]
    
    cnt = np.iinfo(np.int16).max
    threshold = 0
    for i in range(begin, end+1):
        if cnt_hist[i] < cnt:
            cnt = cnt_hist[i]
            threshold = i 
    print(f'{threshold}: {cnt}')
    img_gray[img_gray>threshold] = 255
    img_gray[img_gray<=threshold] = 0

    plt.subplot(122)
    plt.imshow(img_gray, cmap='gray')
    plt.show()
    return img_gray


def otsu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    threshold_t = 0
    max_g = 0
    
    for t in range(255):
        front = img[img < t]
        back = img[img >= t]
        front_p = len(front) / (h*w)
        back_p = len(back) / (h*w)
        front_mean = np.mean(front) if len(front)>0 else 0.
        back_mean = np.mean(back) if len(back)>0 else 0.
        
        g = front_p*back_p*((front_mean - back_mean)**2)
        if g > max_g:
            max_g = g
            threshold_t = t
    print(f"threshold = {threshold_t}")
    img[img < threshold_t] = 0
    img[img >= threshold_t] = 255

    plt.imshow(img, cmap='gray')
    plt.show()
    return img


img = cv2.imread('../images/lenna.jpg')
otsu(img)
