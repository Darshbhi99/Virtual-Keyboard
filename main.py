import cv2
from cvzone.HandTrackingModule import HandDetector



# url = "https://192.168.0.103:8080/video"
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8)


while True:
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (900, 600))
        frame = detector.findHands(frame)
        lmList, bboxInfo = detector.findPosition(frame)
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break   

cap.release()
cv2.destroyAllWindows()
