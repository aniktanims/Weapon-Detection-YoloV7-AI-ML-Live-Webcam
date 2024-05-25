# object_detection_app/views.py
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import time
import base64

# Load YOLO model for weapon detection
net_weapon = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net_weapon.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_weapon.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
classes_weapon = ["gun"]

# Initialize output layer names for weapon detection
output_layer_names_weapon = net_weapon.getUnconnectedOutLayersNames()
colors_weapon = np.random.uniform(0, 255, size=(len(classes_weapon), 3))

# Initialize last weapon detection time
last_detection_time_weapon = None
frame_skip_interval = 5  # Process every 5th frame to improve performance

# Define cooldown parameters
cooldown_duration = 10  # Cooldown duration in seconds
last_detection_timestamp_weapon = 0

# Define object detection logic with cooldown for weapon detection
def detect_objects_weapon(img):
    global last_detection_time_weapon, last_detection_timestamp_weapon

    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net_weapon.setInput(blob)
    outs = net_weapon.forward(output_layer_names_weapon)

    class_ids = []
    confidences = []
    boxes = []
    detected_labels = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:  # Adjust confidence threshold
                if class_id == 0:  # Ensure the detected class is "Weapon"
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    detected_labels.append(classes_weapon[class_id])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)  # Adjust NMS threshold

    font = cv2.FONT_HERSHEY_PLAIN
    weapon_detected = False

    current_time = time.time()

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = detected_labels[i]
            color = colors_weapon[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

            # Check if detected object is actually a weapon and cooldown period has passed
            if label == "Weapon" and current_time - last_detection_timestamp_weapon > cooldown_duration:
                weapon_detected = True
                last_detection_timestamp_weapon = current_time

    if weapon_detected:
        last_detection_time_weapon = time.time()
        # Capture screenshot
        _, buffer = cv2.imencode('.jpg', img)
        screenshot = base64.b64encode(buffer).decode('utf-8')  # Convert bytes to string
        return img, screenshot
    else:
        return img, None

# Define index view
def index(request):
    return render(request, 'object_detection_app/index.html')

# Define webcam feed generator
# Define webcam feed generator with enhanced error handling
def webcam_feed(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        # Try toggling the camera index if the initial index doesn't work
        print(f"Camera index {camera_index} not found. Trying other indices.")
        for i in range(5):  # Attempt up to 5 different indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Using camera index {i}")
                break
        else:
            print("No available camera found.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Adjust resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Skip frames to improve performance
        frame_count += 3
        if frame_count % frame_skip_interval != 0:
            continue

        # Detect objects in frame
        frame, weapon_screenshot = detect_objects_weapon(frame)

        # Convert frame to JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)

        if weapon_screenshot:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                                                                          b'Content-Type: image/jpeg\r\n\r\n' + weapon_screenshot.encode() + b'\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()


# Define webcam feed view
def webcam_feed_view(request):
    camera_index = 1 # Use 1 for the secondary camera (adjust as needed)
    return StreamingHttpResponse(webcam_feed(camera_index), content_type='multipart/x-mixed-replace; boundary=frame')


@csrf_exempt
def weapon_detection_status(request):
    global last_detection_time_weapon
    if request.method == 'GET':
        if last_detection_time_weapon:
            # Weapon detected, return timestamp
            return JsonResponse({'status': 'Weapon detected', 'timestamp': last_detection_time_weapon})
        else:
            # No weapon detected
            return JsonResponse({'status': 'No weapon detected'})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
