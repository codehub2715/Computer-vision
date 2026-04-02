#Build a license plate detection system using OpenCV.
import cv2

plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

img = cv2.imread("car.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect plates
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in plates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    plate = img[y:y+h, x:x+w]
    cv2.imwrite("plate.jpg", plate)

cv2.imshow("License Plate Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
