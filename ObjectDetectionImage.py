import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("./yolo//yolov4.weights", "./yolo//yolov4.cfg")
classes = []
with open("./yolo//coco_names.txt", "r") as f:
    classes = f.read().splitlines()
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

def detect_objects(image_path):
    # Load image
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Object detection
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # Bounding Box            
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    for i in indexes.flatten():
        box = boxes[i]
        x, y, w, h = box
        confidence = confidences[i]
    
        if confidence <= 0.6:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Resize image
    scale_percent = 50  
    resized_img = cv2.resize(img, (int(width * scale_percent / 100), int(height * scale_percent / 100)))

    # Display result
    cv2.imshow("Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()