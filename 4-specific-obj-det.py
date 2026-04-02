#Train a custom object detection model using YOLO to detect a specific object (for example: passport, phone, or ID card).

from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Predict
results = model("ids.jpg", show=True)

# Save output
results[0].save(filename="output.jpg")

