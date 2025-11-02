'''import cv2
import threading
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
import pythoncom
import win32com.client  # Windows SAPI voice
from queue import Queue

app = FastAPI()

# Load YOLO model
print("Loading YOLOv8s model...")
model = YOLO("yolov8s.pt")
print("YOLOv8s model loaded.")

cap = cv2.VideoCapture(0)

# Queue for speech messages
speech_queue = Queue()
spoken_objects = {}
SPEECH_INTERVAL = 5  # seconds per object (cooldown)
CONF_THRESHOLD = 0.6  # confidence filter
MIN_BOX_SIZE = 60     # pixels, ignore tiny boxes

def speak_worker():
    """Run SAPI.SpVoice in a separate thread."""
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    while True:
        msg = speech_queue.get()
        if msg is None:
            break
        speaker.Speak(msg, 1)  # async
        speech_queue.task_done()

# Start TTS worker thread
threading.Thread(target=speak_worker, daemon=True).start()

def get_distance(box_height, frame_height):
    """Estimate distance from box height."""
    if box_height == 0:
        return 0
    return round(2 * frame_height / box_height, 1)

def generate_frames():
    global spoken_objects
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_height, frame_width = frame.shape[:2]

        results = model(frame, verbose=False)[0]
        current_frame_objects = {}

        for det in results.boxes:
            conf = float(det.conf[0].cpu().numpy())
            if conf < CONF_THRESHOLD:
                continue  # skip low-confidence detections

            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            cls_id = int(det.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]

            box_height = y2 - y1
            if box_height < MIN_BOX_SIZE:
                continue  # skip tiny objects (noise)

            distance = get_distance(box_height, frame_height)

            # Draw detections
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {distance}m",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            # Track for speaking
            current_frame_objects[cls_name] = distance

        # Speak only new/valid detections with cooldown
        now = time.time()
        for obj, dist in current_frame_objects.items():
            last_time = spoken_objects.get(obj, 0)
            if now - last_time > SPEECH_INTERVAL:
                speech_queue.put(f"{obj} ahead at {dist} meters")
                spoken_objects[obj] = now

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def index():
    return HTMLResponse("""
    <html>
      <head><title>YOLOv8 Live Camera</title></head>
      <body>
        <h1>YOLOv8 Live Camera Feed</h1>
        <img src="/video" width="640" height="480" />
      </body>
    </html>
    """)

@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)'''


























'''
import cv2
import threading
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
import pythoncom
import win32com.client  # Windows SAPI voice
from queue import Queue

app = FastAPI()

# Load YOLO model
print("Loading YOLOv8s model...")
model = YOLO("yolov8s.pt")
print("YOLOv8s model loaded.")

cap = cv2.VideoCapture(0)

# Queue for speech messages
speech_queue = Queue()
spoken_objects = {}
SPEECH_INTERVAL = 5  # seconds per object (cooldown)
CONF_THRESHOLD = 0.6  # confidence filter
MIN_BOX_SIZE = 60     # pixels, ignore tiny boxes

def speak_worker():
    """Run SAPI.SpVoice in a separate thread."""
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    while True:
        msg = speech_queue.get()
        if msg is None:
            break
        speaker.Speak(msg, 1)  # async
        speech_queue.task_done()

# Start TTS worker thread
threading.Thread(target=speak_worker, daemon=True).start()

def get_distance(box_height, frame_height):
    """Estimate distance from box height."""
    if box_height == 0:
        return 0
    return round(2 * frame_height / box_height, 1)

def get_position(x_center, frame_width):
    """Determine Left, Center, or Right based on x coordinate."""
    if x_center < frame_width / 3:
        return "left"
    elif x_center < frame_width * 2 / 3:
        return "center"
    else:
        return "right"

def generate_frames():
    global spoken_objects
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_height, frame_width = frame.shape[:2]

        results = model(frame, verbose=False)[0]
        current_frame_objects = {}

        for det in results.boxes:
            conf = float(det.conf[0].cpu().numpy())
            if conf < CONF_THRESHOLD:
                continue  # skip low-confidence detections

            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            cls_id = int(det.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]

            box_height = y2 - y1
            if box_height < MIN_BOX_SIZE:
                continue  # skip tiny objects (noise)

            distance = get_distance(box_height, frame_height)
            x_center = (x1 + x2) / 2
            position = get_position(x_center, frame_width)

            # Draw detections
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {distance}m {position}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            # Track for speaking
            current_frame_objects[cls_name] = (distance, position)

        # Speak only new/valid detections with cooldown
        now = time.time()
        for obj, (dist, pos) in current_frame_objects.items():
            last_time = spoken_objects.get(obj, 0)
            if now - last_time > SPEECH_INTERVAL:
                speech_queue.put(f"{obj} ahead at {dist} meters on the {pos}")
                spoken_objects[obj] = now

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def index():
    return HTMLResponse("""
    <html>
      <head><title>YOLOv8 Live Camera</title></head>
      <body>
        <h1>YOLOv8 Live Camera Feed</h1>
        <img src="/video" width="640" height="480" />
      </body>
    </html>
    """)

@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)'''


















import cv2
import numpy as np
import threading
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import pythoncom
import win32com.client
from queue import Queue
from io import BytesIO
from PIL import Image

app = FastAPI()

# Allow all origins for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading YOLOv8s model...")
model = YOLO("yolov8s.pt")
print("YOLOv8s model loaded.")

# Queue for speech messages
speech_queue = Queue()
spoken_objects = {}
SPEECH_INTERVAL = 5
CONF_THRESHOLD = 0.6
MIN_BOX_SIZE = 60

def speak_worker():
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    while True:
        msg = speech_queue.get()
        if msg is None:
            break
        speaker.Speak(msg, 1)
        speech_queue.task_done()

threading.Thread(target=speak_worker, daemon=True).start()

def get_distance(box_height, frame_height):
    if box_height == 0:
        return 0
    return round(2 * frame_height / box_height, 1)

def get_position(x_center, frame_width):
    if x_center < frame_width / 3:
        return "left"
    elif x_center < frame_width * 2 / 3:
        return "center"
    else:
        return "right"

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_height, frame_width = frame.shape[:2]
    results = model(frame, verbose=False)[0]
    current_frame_objects = {}

    for det in results.boxes:
        conf = float(det.conf[0].cpu().numpy())
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        cls_id = int(det.cls[0].cpu().numpy())
        cls_name = model.names[cls_id]

        box_height = y2 - y1
        if box_height < MIN_BOX_SIZE:
            continue

        distance = get_distance(box_height, frame_height)
        x_center = (x1 + x2) / 2
        position = get_position(x_center, frame_width)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} {distance}m {position}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        current_frame_objects[cls_name] = (distance, position)

    now = time.time()
    for obj, (dist, pos) in current_frame_objects.items():
        last_time = spoken_objects.get(obj, 0)
        if now - last_time > SPEECH_INTERVAL:
            speech_queue.put(f"{obj} ahead at {dist} meters on the {pos}")
            spoken_objects[obj] = now

    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.get("/")
def index():
    return HTMLResponse("""
    <html>
    <head><title>AI Navigation Assistant</title></head>
    <body>
        <h1>AI Navigation Assistant</h1>
        <video id="video" autoplay playsinline width="640" height="480"></video>
        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        <img id="result" width="640" height="480" />
        <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const result = document.getElementById('result');

        async function initCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { exact: "environment" } } // back camera
                });
                video.srcObject = stream;
            } catch (err) {
                console.log("Back camera not found, switching to front.");
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
            }
        }

        async function sendFrame() {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg'));
            const formData = new FormData();
            formData.append("file", blob, "frame.jpg");

            const response = await fetch("/detect/", { method: "POST", body: formData });
            const blobResponse = await response.blob();
            result.src = URL.createObjectURL(blobResponse);
        }

        setInterval(sendFrame, 1000); // every 1 second
        initCamera();
        </script>
    </body>
    </html>
    """)

@app.on_event("shutdown")
def shutdown_event():
    speech_queue.put(None)
