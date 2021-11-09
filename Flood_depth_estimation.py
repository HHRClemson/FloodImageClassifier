import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import skimage.viewer
import imutils

BLUE = (255,0,0)
#Image Path
original_img = ("./Images/flood_181.jpg")

def find_skyline(x1, y1, x2, y2, img):
    k = 1.0 * (y2 - y1) / (x2 - x1)
    d = 1.0 * (y1*x2 - y2*x1) / (x2 - x1)
    return(k,d)
def vignette_filter(img, pixels_falloff = 0, types = 0):
    height, width = img.shape
    radius = max(width, height) / 2.0 * 0.95
    radius = min(width, height) / 2.0 * 0.95
    #pixels_falloff = 0.1
    row_ctr = height / 2
    col_ctr = width / 2
    max_img_rad = math.sqrt(row_ctr * row_ctr + col_ctr * col_ctr)
    res = img.copy()
    
    if types:
        trow = pixels_falloff
        lcol = pixels_falloff
        brow = img.shape[0] - pixels_falloff * 2
        rcol = img.shape[1] - pixels_falloff * 2
    for i in range(height):
        for j in range(width):
            dh = abs(i - row_ctr)
            dw = abs(j - col_ctr)
            if not types:
                dis = math.sqrt(dh * dh + dw * dw)
                if dis > radius:
                    if dis > radius + pixels_falloff:
                        res[i, j] = img [i, j] * (dis) / radius
                        # cv2.imshow('res2', res)
                    else:
                        sigma = (dis - radius) / pixels_falloff
                        res[i, j] = img [i, j] * (1 - sigma * sigma)
                        # cv2.imshow('res3', res)
                else:
                    pass
            else:
                print('not here')
                dis1 = min(abs(i - trow), abs(i - brow))
                dis2 = min(abs(j - lcol), abs(j - rcol))
                if i <= brow and i >= trow and j >= lcol and j <= rcol:
                    pass
                else: 
                    sigma = (dis1 + dis2) * (dis1 + dis2) / (dis1 * dis1 + dis2 * dis2)
                    res[i, j] = img [i, j] * sigma
    return res

def affline_rotate(img, pts1, pts2):
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

if __name__ == "__main__":
    img = cv2.imread(original_img, 0)
    viewer = skimage.viewer.ImageViewer(image=img)
    viewer.show()
    img2 = vignette_filter(img, 0.3)
    viewer = skimage.viewer.ImageViewer(image=img2)
    viewer.show()
    
    img = cv2.blur(img, (5, 5))
    img2 = cv2.blur(img2, (10,10))
    
    img = cv2.addWeighted(img, 0.80, img2, 0.20, 1)
    #cv2.imshow('after vignette_filter', img)
    
    img = cv2.blur(img, (15,15))
    clahe = cv2.createCLAHE(clipLimit = 2.00, tileGridSize = (11,11))
    img = clahe.apply(img)
    ret2, detected_edges = cv2.threshold(img, 10, 230, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('de', detected_edges)
    
    edges = cv2.Canny(detected_edges, 0.2, 1.8, apertureSize = 3)
    #cv2.imshow('ee', edges)
    
    dst = cv2.bitwise_and(img, img, mask = edges)
    #cv2.imshow('dst', dst)
    
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 10, np.pi/180, minLineLength, maxLineGap)
    a1, b1, a2, b2 = (0,0,0,0)
    dis = 0
    
    print(lines[0])
    for x1,y1,x2,y2 in lines[0]:
        if (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) > dis:
            a1,b1,a2,b2 = (x1,y1,x2,y2)
    (k, d) = find_skyline(a1, b1, a2, b2, img)
    
    print("line : ", k, d)
    
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            pos = int(k * j + d)
            if i <= pos + 4:
                dst[i, j] = 0
    viewer = skimage.viewer.ImageViewer(image=dst)
    viewer.show()
    
    original = cv2.imread(original_img)
    for row in range(original.shape[0]):
        for bt in range(original.shape[1]):
            if dst[row, bt] == 0:
                pass
            else:
                original[row, bt] = BLUE
    
    srcy1 = img.shape[1]
    srcx1 = k * srcy1 + d
    srcx2 = img.shape[0]
    srcy2 = 0
    srcx3 = img.shape[0]
    srcy3 = img.shape[1]
    pts1 = np.float32([[int(srcy1), int(srcx1)], [srcy2, srcx2], [srcy3, srcx3]])
    pts2 = np.float32([[img.shape[1] * 0.9, 0], [0, img.shape[0] / 7], [img.shape[1] * 0.75, img.shape[0]]])

    img = cv2.imread(original_img, 0)
    img = affline_rotate(img,pts1, pts2)
    detected_edges = affline_rotate(detected_edges, pts1, pts2)
    result = affline_rotate(original, pts1,pts2)
    viewer = skimage.viewer.ImageViewer(image=img)
    viewer.show()
    viewer = skimage.viewer.ImageViewer(image=detected_edges)
    viewer.show()
    viewer = skimage.viewer.ImageViewer(image=result)
    viewer.show()

            
    contours = cv2.findContours(detected_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    bottle_clone = img.copy()
    cv2.drawContours(bottle_clone, contours, -1, (255, 0, 0), 2)
    viewer = skimage.viewer.ImageViewer(image=bottle_clone)
    viewer.show()
    
    areas = [cv2.contourArea(contour) for contour in contours]
    (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))
    print(areas[-1])
    # print contour with largest area
    bottle_clone = img.copy()
    cv2.drawContours(bottle_clone, [contours[-1]], -1, (255, 0, 0), 2)
    cv2.drawContours(bottle_clone, [contours[-2]], -1, (255, 0, 0), 2)
    (x, y, w, h) = cv2.boundingRect(contours[-1])
    cv2.putText(bottle_clone, "Area : " + str(areas[-1]), (x + 45 , y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    viewer = skimage.viewer.ImageViewer(image=bottle_clone)
    viewer.show()
    
    bottle_clone = img.copy()
    (x, y, w, h) = cv2.boundingRect(contours[-1])
    aspectRatio = w / float(h)
    print(aspectRatio)
    if aspectRatio < 0.4:
        cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(bottle_clone, "Full", (x + 45, y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    else:
        cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(bottle_clone, "Low", (x + 45, y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    viewer = skimage.viewer.ImageViewer(image=bottle_clone)
    viewer.show()
        
