import cv2
import easyocr
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np


image_path = "app/img/ai.png"
model = YOLO('yolo11n.pt') 
reader = easyocr.Reader(['en'])


def test_view():
    camera=cv2.VideoCapture('any host')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 10)
    class_name=None
    while True:
        ret, image =camera.read()
        if ret:
            image = cv2.resize(image,(640,480))
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model(img_rgb,verbose=False)
            box = [box for result in results for box in result.boxes]
            
            if box:
                x1, y1, x2, y2 = map(int, box[0].xyxy[0]) 
                confidence = box[0].conf[0].item()
                cls = int(box[0].cls[0].item()) 
                class_name = model.names[cls]
                print(f"Detected: {class_name}, Confidence: {confidence:.2f}, Box: [{x1},{y1},{x2},{y2}]")
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('frame',img_rgb)
            if class_name == 'display':
                cropped_display = img_rgb[y1:y2, x1:x2]
                ocr_results = reader.readtext(cropped_display, detail=0)
                extracted_text = " ".join(ocr_results) 
                print(f"Extracted Text:{extracted_text}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('break')
            camera.release()
            cv2.destroyAllWindows()
       
    cv2.destroyAllWindows()

test_view()