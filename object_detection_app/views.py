from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import time
import base64
import requests
import serial
import tempfile
import threading

# Define serial port and baud rate for Arduino communication
SERIAL_PORT = 'COM3'  # Update this with the correct serial port name
BAUD_RATE = 9600

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

def send_signal_to_arduino():
    # Send signal to Arduino to produce a beep sound
    ser.write(b'b')  # Assuming 'b' indicates beep command

# Load YOLO model for weapon detection once
net_weapon = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes_weapon = ["Weapon"]
output_layer_names_weapon = net_weapon.getUnconnectedOutLayersNames()
colors_weapon = np.random.uniform(0, 255, size=(len(classes_weapon), 3))

# Initialize last weapon detection time
last_detection_time_weapon = None
last_sms_time_weapon = 0  # Timestamp for last SMS sent
frame_skip_interval = 5  # Process every 5th frame to improve performance

# Define cooldown parameters
cooldown_duration = 10  # Cooldown duration in seconds
sms_cooldown_duration = 15  # Cooldown duration for SMS in seconds
last_detection_timestamp_weapon = 0

# Define the SMS API URL and key
SMS_API_URL = "http://bulksmsbd.net/api/smsapi"
SMS_API_KEY = "ptLeqXo6ieUEAwmg4oog"
SMS_SENDER_ID = "8809617612429"
SMS_RECEIVER = "8801797622411"  # Replace with the actual receiver number

latest_screenshot_path = None  # Path to the latest screenshot

# Lock to synchronize access to the latest screenshot
screenshot_lock = threading.Lock()

def send_sms(message):
    payload = {
        'api_key': SMS_API_KEY,
        'senderid': SMS_SENDER_ID,
        'number': SMS_RECEIVER,
        'message': message,
        'type': 'text'
    }
    response = requests.post(SMS_API_URL, data=payload)
    return response.status_code == 202  # Check if the SMS was submitted successfully

# Define object detection logic with cooldown for weapon detection
def detect_objects_weapon(img):
    global last_detection_time_weapon, last_detection_timestamp_weapon, last_sms_time_weapon, latest_screenshot_path

    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
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
            if confidence > 0.35:  # Increase confidence threshold
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Adjust NMS threshold

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

    screenshot_base64 = None
    if weapon_detected:
        send_signal_to_arduino()
        last_detection_time_weapon = current_time

        # Send SMS notification if cooldown period for SMS has passed
        if current_time - last_sms_time_weapon > sms_cooldown_duration:
            send_sms("CSE499B PROJECT: Weapon detected at CAM1, Location: NSU-GRNDFLR")
            last_sms_time_weapon = current_time  # Update last SMS sent time

        # Capture screenshot
        _, buffer = cv2.imencode('.jpg', img)
        screenshot_bytes = buffer.tobytes()
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        # Save screenshot to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(screenshot_bytes)
            temp_file_path = temp_file.name

        # Store the path to the latest screenshot
        with screenshot_lock:
            latest_screenshot_path = temp_file_path

    return img, screenshot_base64

def get_latest_screenshot():
    global latest_screenshot_path
    with screenshot_lock:
        if latest_screenshot_path:
            try:
                with open(latest_screenshot_path, 'rb') as f:
                    screenshot_bytes = f.read()
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                return screenshot_base64
            except Exception as e:
                print(f"Error retrieving latest screenshot: {e}")
                return None
        else:
            return None

# Preinitialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Adjust resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Add a warm-up period for the camera
warmup_frames = 20
for _ in range(warmup_frames):
    cap.read()
    time.sleep(0.1)  # Small delay to allow the camera to stabilize

# Define index view
def index(request):
    return render(request, 'object_detection_app/index.html')

# Define webcam feed generator
def webcam_feed():
    global cap
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to improve performance
        frame_count += 1
        if frame_count % frame_skip_interval != 0:
            continue

        # Detect objects in frame
        frame, weapon_screenshot = detect_objects_weapon(frame)

        # Convert frame to JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        if weapon_screenshot:
            # Convert screenshot to JPEG format
            screenshot_bytes = base64.b64decode(weapon_screenshot)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + screenshot_bytes + b'\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Release the camera when the application is closed
import atexit
@atexit.register
def cleanup():
    cap.release()
    cv2.destroyAllWindows()

# Define webcam feed view
def webcam_feed_view(request):
    return StreamingHttpResponse(webcam_feed(), content_type='multipart/x-mixed-replace; boundary=frame')

# Define weapon detection status view
@csrf_exempt
def weapon_detection_status(request):
    global last_detection_time_weapon
    if request.method == 'GET':
        if last_detection_time_weapon:
            # Weapon detected, return timestamp and screenshot
            screenshot = get_latest_screenshot()  # Call the function to get the latest screenshot
            return JsonResponse({
                'status': 'Weapon detected',
                'timestamp': last_detection_time_weapon,
                'screenshot': screenshot  # Include screenshot in the response
            })
        else:
            # No weapon detected
            return JsonResponse({'status': 'No weapon detected'})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
