import numpy as np
import cv2

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
id =input("enter user id");
sampnum=0;
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        sampnum=sampnum+1;
        cv2.imwrite("dataset/"+id+"."+str(sampnum)+".jpg",gray[y:y+h,x:x+w]);
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('frame',img)
    cv2.waitKey(100)
    if(sampnum>20):
         break
    
cap.release()
cv2.destroyAllWindows()
