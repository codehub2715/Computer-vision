#Create a vehicle detection and counting system from a traffic video.

import cv2

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic.mp4")
# Define vehicle classes (COCO dataset)
vehicle_classes = [2, 3, 5, 7] 
# 2=car, 3=motorbike, 5=bus, 7=truck

# Line position for counting
line_y = 300

count = 0
# Store detected IDs
detected_ids = set()

while True:
    ret,frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])

            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

                # Count when crossing line
                if cy > line_y - 5 and cy < line_y + 5:
                    obj_id = (cx, cy)

                    if obj_id not in detected_ids:
                        detected_ids.add(obj_id)
                        count += 1

    # Draw counting line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255,0,0), 2)

    # Display count
    cv2.putText(frame, f"Vehicle Count: {count}", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Vehical Detect and Count",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
