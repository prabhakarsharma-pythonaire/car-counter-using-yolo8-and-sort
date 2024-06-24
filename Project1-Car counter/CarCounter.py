import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from  sort import *



webcam_obj = cv2.VideoCapture(r"C:\Users\91920\Downloads\1721294-sd_960_540_25fps.mp4")


model = YOLO("../Yolo-weights/yolov8s.pt")


#coco dataset
classname = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
             "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
             "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
             "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
             "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
             "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
             "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
             "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
             "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
             "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# mask =cv2.imread("mask.png")  #don't created mask yet

#tracker
tracker = Sort(max_age=10,min_hits=3,iou_threshold=0.3)

while True:
    success, img = webcam_obj.read()
    results = model(img,stream=True)

    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding box
            X1, Y1, X2, Y2= box.xyxy[0]
            X1, Y1, X2, Y2 = int(X1),int(Y1),int(X2),int(Y2)
            #cv2.rectangle(img,(X1,Y1),(X2,Y2),(255,0,255),3)

            w,h = X2-X1,Y2-Y1


            #Confidencce
            conf = math.ceil(((box.conf[0]*100))/100)  #getting in two decimal places

           #class name
            cls = int(box.cls[0])

            currentclass = classname[cls]

            if currentclass=="car" or currentclass=="truck"  or currentclass=="bus" or currentclass=="motorbike" and conf>0.3:
                # cvzone.putTextRect(img,f"{classname[cls]}",(max(0,X1),max(30 ,Y1)),scale=0.7,thickness=1,offset=3)
                cvzone.cornerRect(img, (X1, Y1, w, h), l=9,rt=5)
                currentArray = np.array([X1,Y1,X2,Y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        X1,Y1,X2,Y2,id = result
        X1, Y1, X2, Y2 = int(X1), int(Y1), int(X2), int(Y2)
        print(result)
        w,h=X2-X1,Y2-Y1
        cvzone.cornerRect(img,(X1,Y1,w,h),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f"{int(id)}", (max(0, X1), max(30, Y1)), scale=2, thickness=3, offset=3)
    cv2.imshow("Image",img)
    cv2.waitKey(0)


