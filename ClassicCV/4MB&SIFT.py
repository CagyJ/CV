def gray_medianBlur(img):
    h,w = img.shape[:2]
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    dst = np.zeros((h,w,3),dtype=int)
    collect = np.zeros(9,dtype=int)
    for i in range(1,h-1):
        for j in range(1,w-1):
            k = 0                         #  index
            for m in range(-1,2):
                for n in range(-1,2):
                    gray = img_gray[i+m,j+n]
                    collect[k] = gray
                    k = k + 1
            for k in range(0,9):
                p1 = collect[k]
                for t in range(k+1,9):
                    if p1<collect[t]:
                        tmp = collect[t]
                        collect[t] = p1
                        p1 = tmp
            dst[i,j] = collect[4]
    return dst

