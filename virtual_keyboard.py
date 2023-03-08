import cv2
import numpy as np
from time import sleep
from cvzone.HandTrackingModule import HandDetector


# url = 'https://192.168.0.103:8080/video'
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)
# Resizing the width of webcam to 1280
cap.set(3,1280)
# Resizing the height of webcam to 720
cap.set(4,720)
detector = HandDetector(detectionCon=0.8)


# Letters for Virtual Keyboard
letter = [['Q','W','E','R','T','Y','U','I','O','P'],['A','S',
                'D','F','G','H','J','K','L',';'],['Z','X','C','V','B','N','M',',','/','.']]
class button:
    def __init__(self, pos, text):
        self.pos = pos
        self.text = text

    def draw(self, img):
        x,y = self.pos
        # Creating a square at given postion of 50x50  
        cv2.rectangle(img, (x, y), (x+50, y+50), (255, 0, 255), cv2.FILLED)
        # Adding the Letter into the given square
        cv2.putText(img, self.text, (x+15, y+50-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        new_img = np.zeros_like(img, dtype=np.uint8)
        out = img.copy()
        alpha = 0.5
        mask = new_img.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, new_img, 1-alpha, 0)[mask]
        return out


finalText = ""
while True:
    success, img = cap.read()
    img = cv2.resize(img, (900, 600))
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img=img)
    
    cv2.rectangle(img, (80, 80), (700,280),(255,0,255),2)
    
    buttons = []
    x, y = 100, 100
    for i in range(len(letter)):
        for j in range(len(letter[i])):
            buts = button((x+(60*j), y), letter[i][j])
            buttons.append(buts)
            img = buts.draw(img)
        y+=60
    
    
    if lmList:
        for buts in buttons:
            x, y = buts.pos
            if x<lmList[8][0]<x+50 and y<lmList[8][1]<y+50:
                cv2.rectangle(img, (x, y), (x+50, y+50), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, buts.text, (x+15, y+50-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                l, _, _ = detector.findDistance(8,12, img, draw = False)
                if l<25:
                    finalText = finalText + buts.text
                    cv2.rectangle(img, (x, y), (x+50, y+50), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, buts.text, (x+15, y+50-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    sleep(0.4)

    cv2.rectangle(img, (100, 500), (800, 550), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, finalText, (115, 535), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imshow('Video',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()