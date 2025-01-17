# ////////////////////////
# Detecting class
# ////////////////////////

from ultralytics import YOLO
import cv2
import cvzone
import math

# Webcam setup
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# YOLO model
model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Tracking setup
limits = [500, 297, 873, 297]  # Line coordinates
totalCount = 0  # Count for objects crossing the line
trackedObjects = []  # List of previously tracked objects

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Display object name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Consider only specific classes and confidence threshold
            if currentClass in ["mouse", "cell phone"] and conf > 0.5:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Center of the object
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                # Draw bounding box and center
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(30, y1)), scale=1, thickness=2)

                # Check if the center crosses the red line
                if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
                    # Only count if the object is not already in the tracked list
                    if (cx, cy) not in trackedObjects:
                        totalCount += 1
                        trackedObjects.append((cx, cy))  # Track the object
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 15)

    # Draw the red line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cvzone.putTextRect(img, f'Count: {totalCount}', (50, 50), scale=2, thickness=2, offset=10)

    # Display the frame
    cv2.imshow("Image", img)
    cv2.waitKey(1)
