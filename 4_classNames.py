


# ////////////////////////
# Detecting class
# ////////////////////////

from ultralytics import YOLO
import cv2
import cvzone
# New
import math

# webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# cap = cv2.VideoCapture("../Videos/name.mp4")

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
              "teddy bear", "hair drier", "toothbrush"
              ]
limits = [400, 297, 673, 297]


while True:
    success, img = cap.read()
    results = model(img, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounting box
            x1,y1,x2,y2 =box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,255),3)
            #draw rectangle box. (255,0,255) colour, 3 thickness
            w,h = x2-x1,y2-y1
            bbox = x1,y1,w,h
            cvzone.cornerRect(img, (bbox))
            
            #confidence  
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cvzone.putTextRect(img, f'{conf}',(max(0,x1),max(30,y1)))
            #Display object name
            cls = int(box.cls[0])
            print()

            #1 cvzone.putTextRect(img, f'{cls} {conf}',(max(0,x1),max(30,y1)))
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',(max(0,x1),max(30,y1)),scale = 1,thickness=2)


    cv2.imshow("Image",img)
    cv2.waitKey(1)