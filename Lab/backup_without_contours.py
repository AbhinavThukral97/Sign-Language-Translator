import cv2
import numpy as np
import math
import sys
import os

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def image_resize(image, height = 50, inter = cv2.INTER_AREA):
    resized = cv2.resize(image, (height,height), interpolation = inter)
    return resized

while(True):
    ret, img = cap.read()
    img = cv2.flip(img,1)
    start = 200
    end = 450

    cv2.rectangle(img, (start,start), (end,end), (102,185,255),5)
    crop_img = img[start:end, start:end]

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
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    dilation_rgb = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)

    all_img = np.hstack((drawing, crop_img,dilation_rgb))
    cv2.imshow('WebCam', img)
    cv2.imshow('All', all_img)

    k = cv2.waitKey(10)

    if k == 27:
        break