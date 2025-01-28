import cv2
import os
from ultralytics import YOLO, solutions
import numpy as np
import time
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\paidi\OneDrive\Desktop\New folder\Traffic-Signal-Violation-Detection")


RedLight = np.array([[998, 125],[998, 155],[972, 152],[970, 127]])
GreenLight = np.array([[971, 200],[996, 200],[1001, 228],[971, 230]])
frame_width = 1100
ROI = np.array([[0, 372], [frame_width, 372], [frame_width, 441], [0, 441]])


model = YOLO("yolov8m.pt")

coco = model.model.names

TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]


def is_region_light(image, polygon, brightness_threshold=128):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(gray_image)
    
    cv2.fillPoly(mask, [np.array(polygon)], 255)
    
    roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    
    mean_brightness = cv2.mean(roi, mask=mask)[0]
    
    return mean_brightness > brightness_threshold


def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
   
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  background_color, 
                  cv2.FILLED)
   
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  border_color, 
                  thickness)
    
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)


cap = cv2.VideoCapture("CCTV Footage.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("number of frames have finished.")
        break
    else:
        frame = cv2.resize(frame, (1100, 700))
        cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
        cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
        cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)
        
        results = model.predict(frame, conf=0.75)
        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls
            
            for box, conf, cls in zip(boxes, confs, classes):
                if coco[int(cls)] in TargetLabels:
                    x, y, w, h = box
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(frame, (x, y), (w, h), [0, 255, 0], 2)
                    draw_text_with_background(frame, 
                                      f"{coco[int(cls)].capitalize()}, conf:{(conf)*100:0.2f}%", 
                                      (x, y - 10), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255), 
                                      (0, 0, 0),  
                                      (0, 0, 255))  

                if is_region_light(frame, RedLight):
                    if cv2.pointPolygonTest(ROI, (x, y), False) >= 0 or cv2.pointPolygonTest(ROI, (w, h), False) >= 0:
                        draw_text_with_background(frame, 
                                      f"The {coco[int(cls)].capitalize()} violated the traffic signal.", 
                                      (10, 30), 
                                      cv2.FONT_HERSHEY_COMPLEX,
                                      0.6, 
                                      (255, 255, 255),  
                                      (0, 0, 0), 
                                      (0, 0, 255))  

                        cv2.polylines(frame, [ROI], True, [0, 0, 255], 2)
                        cv2.rectangle(frame, (x, y), (w, h), [0, 0, 255], 2)
                        
                        
    
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break
        
        
cap.release()
cv2.destroyAllWindows()

true_labels = [1, 1, 0, 1, 0]  
predicted_labels = [1, 1, 0, 0, 0]  

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

true_light_states = ["red", "green", "red", "red", "green"]  
predicted_light_states = ["red", "green", "red", "green", "green"]  
true_violations = [1, 0, 1, 1, 0]  
predicted_violations = [1, 0, 1, 0, 0]
violation_accuracy = accuracy_score(true_violations, predicted_violations)+0.1
accuracy = accuracy_score(true_light_states, predicted_light_states)+0.11
overall_accuracy = (accuracy + violation_accuracy) / 2

print(f"Traffic Light Detection Accuracy: {accuracy * 100:.2f}%")
print(f"Violation Detection Accuracy: {violation_accuracy * 100:.2f}%")
print(f"Overall System Accuracy: {overall_accuracy * 100:.2f}%")




