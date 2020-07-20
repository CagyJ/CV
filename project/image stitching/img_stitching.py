import numpy as np 
import matplotlib.pyplot as plt 
import cv2


def show_color_img(img, size=(3, 3)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def detect_KP(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用SIFT检测角点
    sift = cv2.xfeatures2d.SIFT_create()
    # 获取关键点和描述符
    kps, des = sift.detectAndCompute(img, None)
    # kps为SIFT特征点，features为128维向量
    # kps = np.float32([kp.pt for kp in kps])  # 将结果转换成NumPy数组, if need it.
    # img_kp = cv2.drawKeypoints(img,kp,None)  #绘制关键点
    return kps,des


def flann_match_KP(des_le, des_ri, error):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_le, des_ri, k=2)
    good = []
    for i,j in matches:
        if i.distance < j.distance*error:
            good.append(i)
    
    return good


def draw_match(img_le, kps_le, img_ri, kps_ri, good):
    draw_params = dict(matchColor = (255,0,0), singlePointColor = None, flags = 2)
    img = cv2.drawMatches(img_le, kps_le, img_ri, kps_ri, good, None, **draw_params)
    return img


def find_M(img_le, img_ri, kps_le, kps_ri, good):
    pts_le = np.float32([kps_le[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts_ri = np.float32([kps_ri[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(pts_le, pts_ri, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    
    h,w,d = img_le.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    
    img_ri = cv2.polylines(img_ri, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    
    warpImg = cv2.warpPerspective(img_ri, np.linalg.inv(M), (img_le.shape[1]+img_ri.shape[1], img_ri.shape[0]))
    
    return M, warpImg


def stitching(img_le, img_ri, warpImg):
    
    direct = warpImg.copy()
    direct[0:img_le.shape[0], 0:img_le.shape[1]] = img_le
    
    rows, cols = img_le.shape[:2]
    
    for col in range(0, cols):
        if img_le[:, col].any() and warpImg[:, col].any(): # find the left side of the stitching image
            left = col
            break
        
    for col in range(cols-1, 0, -1):
        if img_le[:, col].any() and warpImg[:, col].any(): # right side
            right = col
            break
    
    res = np.zeros([rows, cols, 3], np.uint8)
    
    for row in range(0, rows):
        for col in range(0, cols):
            if not img_le[row, col].any():
                res[row, col] = warpImg[row, col]
            elif not warpImg[row, col].any():
                res[row, col] = img_le[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(img_le[row, col] * (1-alpha) + warpImg[row, col] * alpha, 0, 255)
    warpImg[0:img_le.shape[0], 0:img_le.shape[1]] = res[:,:,:]
    
    return warpImg


def img_stiching(img_le, img_ri):
    kps_le, des_le = detect_KP(img_le)
    kps_ri, des_ri = detect_KP(img_ri)
    left_sift = cv2.drawKeypoints(img_le, kps_le, outImage=np.array([]))
    right_sift = cv2.drawKeypoints(img_ri, kps_ri, outImage=np.array([]))

    show_color_img(left_sift, (10,10))
    show_color_img(right_sift, (10,10))

    good = flann_match_KP(des_le, des_ri, 0.8)
    img_match = draw_match(img_le, kps_le, img_ri, kps_ri, good)

    show_color_img(img_match, (10,10))

    M, warpImg = find_M(img_le, img_ri, kps_le, kps_ri, good)

    res_img = stitching(img_le, img_ri, warpImg)

    return res_img




left = cv2.imread('./imgs//left2.jpg')
right = cv2.imread('./imgs/right2.jpg')

out = img_stiching(left, right)
show_color_img(out, (15,15))