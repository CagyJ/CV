import cv2


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


