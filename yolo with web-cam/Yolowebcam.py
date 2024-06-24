from ultralytics import YOLO
import cv2
import cvzone
import math

# webcam_obj = cv2.VideoCapture(0)   #for webcam
webcam_obj = cv2.VideoCapture(r"C:\Users\91920\Videos\Record_2022-03-01-12-13-52_456a1fceef3bca625c7c7a8b6fa38f99.mp4")
webcam_obj.set(3,1280) #width
webcam_obj.set(4,720)  #height


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

while True:
    success, img = webcam_obj.read()
    results = model(img,stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding box
            X1, Y1, X2, Y2= box.xyxy[0]
            X1, Y1, X2, Y2 = int(X1),int(Y1),int(X2),int(Y2)
            #cv2.rectangle(img,(X1,Y1),(X2,Y2),(255,0,255),3)

            w,h = X2-X1,Y2-Y1
            cvzone.cornerRect(img,(X1,Y1,w,h))

            #Confidencce
            conf = math.ceil(((box.conf[0]*100))/100)  #getting in two decimal places

           #class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img,f"{classname[cls]}",(max(0,X1),max(30 ,Y1)),scale=0.7,thickness=1)
    cv2.imshow("Image",img)
    cv2.waitKey(1)


