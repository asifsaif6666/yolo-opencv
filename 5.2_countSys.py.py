import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import sys
sys.path.append(r'E:\Downloads\yolo\livecctvvehiclecountingyolov8-main\sort\sort.py')
from sort import Sort

cap = cv2.VideoCapture(0)  # for webcam
#cap = cv2.VideoCapture(r"C:\Users\asifs\Downloads\cars.mp4")
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv2.imread(r"C:\Users\asifs\Downloads\mask.png", 0)  # Grayscale mask for area restriction

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
totalCounts = []

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Apply the mask to the frame
    # imgMasked = cv2.bitwise_and(img, img, mask=mask)  # Change made: Apply mask to the image
    
    # results = model(imgMasked, stream=True)  # Pass the masked image for YOLO detection
    results = model(img, stream=True)  # Pass the masked image for YOLO detection

    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["cell phone"] and conf > 0.1:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    
    resultsTracker = tracker.update(detections)
    # cv2.line(imgMasked, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        # cvzone.cornerRect(imgMasked, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # cvzone.putTextRect(imgMasked, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        
        # Center point of the object
        cx, cy = x1 + w // 2, y1 + h // 2
        # cv2.circle(imgMasked, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count vehicles crossing the line within the masked area
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if id not in totalCounts:
                totalCounts.append(id)
                # cv2.line(imgMasked, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display count
    # cvzone.putTextRect(imgMasked, f' Count: {len(totalCounts)}', (50, 50), scale=2, thickness=2, offset=10)

    # cv2.imshow("Masked Image", imgMasked)
    cvzone.putTextRect(img, f' Count: {len(totalCounts)}', (50, 50), scale=2, thickness=2, offset=10)

    cv2.imshow("Masked Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    





# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# 
# #cap = cv2.VideoCapture(0) #for webcam
# cap = cv2.VideoCapture(r"C:\Users\asifs\Downloads\cars.mp4")
# cap.set(3,1280)
# cap.set(4,720)
# 
# model = YOLO('yolov8n.pt')
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             
#             # bounding box
#             x1,y1,x2,y2=box.xyxy[0]
#             x1,y1,x2,y2=int (x1),int (y1),int (x2),int (y2)
# #             cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             
#             w,h=x2-x1,y2-y1
#             cvzone.cornerRect(img,(x1,y1,w,h))
#             # Confidence
#             conf = math.ceil((box.conf[0]*100))/100
#             # Class Name
#             cls = int(box.cls[0])
#             
#             
#             cvzone.putTextRect(img, f'{classNames[cls]} {conf}',(max(0, x1), max(35,y1)),scale=1, thickness=1)
#             
#     cv2.imshow("IMAGE",img)
#     cv2.waitKey(1)
# 
# 
