from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-weights/yolov8n.pt')
results = model(r"C:\Users\91920\Downloads\Licplatesdetection_train\license_plates_detection_train\8.jpg", show=True) #show=True it'll show the probability of being thta object
cv2.waitKey(0)


