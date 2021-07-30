# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 05:43:11 2021

@author: ACER
"""

import cv2, time
from PIL import Image
camera = 0
video = cv2.VideoCapture(camera,cv2.CAP_DSHOW)
a = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetect = cv2.CascadeClassifier('face-detect.xml')
recognizer.read('c://capture/training/training.yml')
id = 0
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,0,0)
while True:
  a = a + 1  
  check, frame = video.read()  
  print(check)
  print(frame)  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces=faceDetect.detectMultiScale(gray,1.3,5);
  for (x,y,w,h) in faces:
      cv2.imwrite("DataSet/User."+str(id)+"."+str(a)+".jpg", gray[y:y+h,x:x+w])
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
      id, conf=recognizer.predict(gray[y:y+h,x:x+w])
      if (id == 1) :
         id = "murnia"   
      cv2.putText(frame,str(id),(x+w,y+h),fontFace,fontScale,fontColor)
  cv2.imshow("wajah",frame)
  if (cv2.waitKey(1)==ord('q')):
       break
  print(a)
video.release()
cv2.destroyAllWindows()
