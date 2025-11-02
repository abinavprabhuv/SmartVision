import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import base64
import time

app = FastAPI()

# Allow browser access (CORS)
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

def get_distance(h, fh):
    """Estimate object distance based on bounding box height"""
    return round(2 * fh / h, 1) if h else 0

def get_position(x, fw):
    """Determine left / center / right"""
    if x < fw / 3:
        return "left"
    elif x < 2 * fw / 3:
        return "center"
    else:
        return "right"

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """Receive one frame, run YOLO, return detections + annotated image"""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = np.array(image)

    fh, fw = frame.shape[:2]
    results = model(frame, verbose=False)[0]

    detections = []
    for det in results.boxes:
        conf = float(det.conf[0].cpu().numpy())
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        cls = model.names[int(det.cls[0].cpu().numpy())]
        bh = y2 - y1
        if bh < MIN_BOX_SIZE:
            continue
        dist = get_distance(bh, fh)
        pos = get_position((x1 + x2) / 2, fw)
        detections.append({"object": cls, "distance": dist, "position": pos})
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"{cls} {dist}m {pos}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    _, buffer = cv2.imencode(".jpg", frame)
    frame_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({"detections": detections, "frame": frame_base64})

@app.get("/")
def index():
    """Frontend HTML page"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
  <title>YOLOv8 Smart Vision</title>
</head>
<body style="text-align:center; background:black; color:white;">
  <h1>YOLOv8 Live Smart Vision</h1>
  <video id="video" autoplay playsinline width="640" height="480"></video>
  <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
  <br>
  <img id="result" width="640" height="480">
  <p id="spoken" style="font-size:20px; margin-top:10px;"></p>

  <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const resultImg = document.getElementById('result');
    const spoken = document.getElementById('spoken');

    // Use browser camera
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then(stream => { video.srcObject = stream; })
      .catch(err => alert("Camera error: " + err));

    let lastSpoken = {};
    const SPEECH_INTERVAL = 5000; // ms

    async function sendFrame() {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg'));
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await fetch("/detect/", { method: "POST", body: formData });
      const data = await response.json();

      // Show annotated frame
      resultImg.src = "data:image/jpeg;base64," + data.frame;

      // Speak detected objects
      const now = Date.now();
      data.detections.forEach(det => {
        const key = det.object;
        if (!lastSpoken[key] || now - lastSpoken[key] > SPEECH_INTERVAL) {
          const msg = `${det.object} ahead at ${det.distance} meters on the ${det.position}`;
          spoken.innerText = msg;
          const utterance = new SpeechSynthesisUtterance(msg);
          utterance.rate = 1.0;
          speechSynthesis.speak(utterance);
          lastSpoken[key] = now;
        }
      });
    }

    setInterval(sendFrame, 1000); // send 1 frame per second
  </script>
</body>
</html>
    """)
