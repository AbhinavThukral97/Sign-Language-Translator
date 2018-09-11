import cv2
import numpy as np
import math
import sys

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(cap.isOpened()):
    ret, img = cap.read()
    img = cv2.flip(img,1)
    start = 200
    end = 450

    cv2.rectangle(img, (start,start), (end,end), (102,185,255),5)
    crop_img = img[start:end, start:end]
    contour_img = img[start:end, start:end]

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    _, thresh1 = cv2.threshold(blurred, 135, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernal = np.ones((10,10),np.uint8)
    erosion = cv2.erode(thresh1,kernal,iterations=1)
    dilation = cv2.dilate(erosion,kernal,iterations=1)
    image, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(contour_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(contour_img, far, 1, [0,0,255], -1)
        dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(contour_img,start, end, [0,255,0], 2)
        cv2.circle(contour_img,far,5,[0,0,255],-1)

    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, contour_img))
    cv2.imshow('Thresholded', thresh1)
    cv2.imshow('Dilated', dilation)
    cv2.imshow('Contours', all_img)
    cv2.imshow('Grayscale', blurred)

    k = cv2.waitKey(10)
    if k == 27:
        break