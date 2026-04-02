#Detect multiple faces in an image and draw bounding boxes around them.

import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread("dogs.jpeg")

faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Face Detection:",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
