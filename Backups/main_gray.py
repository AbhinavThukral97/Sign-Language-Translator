import cv2
import numpy as np
import pandas as pd
import math
import sys
import os
import tensorflow as tf 
from keras.models import load_model

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def image_resize(image, height = 50, inter = cv2.INTER_AREA):
    resized = cv2.resize(image, (height,height), interpolation = inter)
    return resized

model = load_model('trained_gray.h5')

encoding_chart = pd.read_csv('label_encoded.csv')
encoding_values = encoding_chart['Encoded'].values
encoding_labels = encoding_chart['Label'].values
int_to_label = dict(zip(encoding_values,encoding_labels))

font = cv2.FONT_HERSHEY_SIMPLEX

history = []
history_length = 20

while(True):
    ret, img = cap.read()
    img = cv2.flip(img,1)
    start = 200
    end = 450
    cv2.rectangle(img, (start,start), (end,end), (102,185,255),5)
    crop_img = img[start:end, start:end]

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    resized = image_resize(grey)
    predicted = model.predict(np.array([np.reshape(resized,(np.shape(resized)[0],np.shape(resized)[1],1))]))
    predicted_char = int_to_label[np.argmax(predicted)]
    
    if(len(history)>=history_length):
        history.clear()
    if(predicted_char != 'None'):
        history.append(predicted_char)

    cv2.putText(img, "".join(history), (start,end+50),font,1,(102,185,255),4)

    cv2.imshow('WebCam', img)
    k = cv2.waitKey(10)
    if k == 27:
        break