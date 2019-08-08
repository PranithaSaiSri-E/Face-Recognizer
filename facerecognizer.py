import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataset'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f)for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(imagePath.split(".")[0].split("\\")[1])
        faces.append(faceNp)
        print(imagePath.split(".")[0].split("\\")[1])
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(20)
    return IDs,faces
IDs,faces=getImagesWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('recognizer/trainingdata.yml');
cv2.destroyAllWindows()
