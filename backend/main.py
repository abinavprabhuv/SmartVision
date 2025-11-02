import cv2
import numpy as np
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

app = FastAPI()

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

CONF_THRESHOLD = 0.6
MIN_BOX_SIZE = 60

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
    detected_objects = []

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

        detected_objects.append(f"{cls_name} {distance} meters {position}")

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} {distance}m {position}",
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.get("/")
def index():
    return HTMLResponse("""
    <html>
    <head>
        <title>AI Navigation Assistant</title>
    </head>
    <body style="font-family: sans-serif; text-align: center; background: #000; color: #0f0;">
        <h2>AI Navigation Assistant</h2>
        <video id="video" autoplay playsinline width="640" height="480"></video>
        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        <img id="result" width="640" height="480" />
        <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const result = document.getElementById('result');
        const synth = window.speechSynthesis;
        let lastSpoken = {};

        async function initCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { exact: "environment" } }
                });
                video.srcObject = stream;
            } catch (err) {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
            }
        }

        function speak(text) {
            if (!synth.speaking) {
                const utter = new SpeechSynthesisUtterance(text);
                synth.speak(utter);
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
            // optional: voice feedback can be added if backend sends JSON later
        }

        setInterval(sendFrame, 1000);
        initCamera();
        </script>
    </body>
    </html>
    """)

