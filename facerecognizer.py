import numpy as np
import cv2

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
roc=cv2.face.LBPHFaceRecognizer_create()
roc.read("recognizer/trainingdata.yml")
id=0;
# font for the text written on image
font = cv2.FONT_HERSHEY_SIMPLEX

#fontColor =(255, 255, 255)

while(True):
    ret, img =  cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=roc.predict(gray[y:y+h,x:x+w])
        #cv2.putText(img,str,(x,y-10),font,0.55,(0,255,0),1)
        cv2.putText(img,str(id),(x,y-20),font,0.55,(0,0,255),2)
        #cv2.putText(img)
    cv2.imshow('frame',img)
    cv2.waitKey(10)
cap.release()
cv2.destroyAllWindows()                       
