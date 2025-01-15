from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
results = model("../YOLO/images/img.jpg", show=True)
cv2.waitKey(0)