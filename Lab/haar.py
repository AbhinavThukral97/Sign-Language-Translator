import cv2
from math import ceil
import sys
import os
import numpy as np
#Face Detection using Haar Cascade
cascPath = 'hand.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)\

while True:
	ret,frame = video_capture.read()
	frame = cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.1, 10, minSize = (50,50))
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		test = gray[x:x+w,y:y+h]
	cv2.imshow('Video', frame)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break