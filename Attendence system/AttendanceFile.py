import cv2 
import numpy as np 
import face_recognition
import os
from datetime import datetime

path='Images'
Images=[]
ClassNames=[]
mylist=os.listdir(path)
print(mylist)
#getting the list of the students
for cl in mylist:
    currentimage=cv2.imread(f'{path}/{cl}')
    Images.append(currentimage)
    ClassNames.append(os.path.splitext(cl)[0])
print(ClassNames)
#finding encoding of thr images
def findencodings(Images):
    encodelist=[]
    for img in Images:
        #converting image from BGR format to RGB format
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markattendance(name):
    with open('attendance.csv','r+') as f:  #opening the file in write/read mode
        myDataList=f.readlines()
        NameList=[]
        for line in myDataList:
            entry=line.split(',')
            NameList.append(entry[0])
        if name not in NameList:
            now=datetime.now()
            datetimestring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datetimestring}')

encodelistknown=findencodings(Images)         
print("ENCODING COMPLETE....")
#finding the matches with the encodings
webcamcapture=cv2.VideoCapture(0)  #initializing web cam
#getting each frame one by one
while True:
    success,img=webcamcapture.read()
    #as we are doing this in real time we are reducing the size of the image in order to speed up the process
    redimg=cv2.resize(img,(0,0),None,0.25,0.25)  #reducing image size by 4 times
    #converting image from BGR format to RGB format
    redimg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #in the webcam image we might actually find multiple faces so for that we going to find the loacations of the faces first and then send it to encoding function
    facesincurrentframe=face_recognition.face_locations(redimg)
    #getting the encodings of the image from the webcam
    encodingsofcurrentframe=face_recognition.face_encodings(redimg,facesincurrentframe)

    #finding matches
    #itterate through all the faces we found in our current frame and campare to the encodings of stored images
    for encodeface,faceloc in zip(encodingsofcurrentframe,facesincurrentframe):  #as we want both the lists in the same loop we use zip
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        faceDistance=face_recognition.face_distance(encodelistknown,encodeface)
        #lowest of the distances will be our best match
        #print(faceDistance)
        MatchIndex=np.argmin(faceDistance)  #getting the minimum distance from the distance list
        #getting the best match
        if matches[MatchIndex]:
            name=ClassNames[MatchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceloc
           # y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4  #enlarging the image again
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2 )
            markattendance(name)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)
