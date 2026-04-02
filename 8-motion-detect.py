#Build a motion detection system using webcam video.


import cv2

cap = cv2.VideoCapture(0)
ret,frame1 = cap.read()
ret,frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,thresh = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)

    cv2.imshow('Motion Detection', thresh)

    frame1 = frame2
    ret,frame2 = cap.read()
    if not ret or cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
