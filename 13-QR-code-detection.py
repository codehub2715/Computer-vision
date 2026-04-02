#Build a QR code detection and scanner using OpenCV.

import cv2

img = cv2.imread("qrcode.png")

detector = cv2.QRCodeDetector()

data, bbox, _ = detector.detectAndDecode(img)

if bbox is not None:
    for i in range(len(bbox[0])):
        pt1 = tuple(map(int, bbox[0][i]))
        pt2 = tuple(map(int, bbox[0][(i+1) % len(bbox[0])]))
        cv2.line(img, pt1, pt2, (0,255,0), 2)

    print("Decoded Data:", data)

cv2.imshow("QR Code", img)
cv2.waitKey(0)
cv2.destroyAllWindows()