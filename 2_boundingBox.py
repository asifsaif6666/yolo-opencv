# ////////////////////////////
# Now draw the bounting boxes


from ultralytics import YOLO
import cv2
import cvzone


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('../Yolo-Weights/yolov8n.pt')

while True:
    success, img = cap.read()
    results = model(img, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 =box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,255),3)
            #draw rectangle box. (255,0,255) colour, 3 thickness
            
            # w,h = x2-x1,y2-y1
            # bbox = int(x1), int(y1), int(w), int(h)
            # cvzone.cornerRect(img, bbox)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
