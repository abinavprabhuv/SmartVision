import cv2
import threading
import time
import platform
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
from queue import Queue

app = FastAPI()

print("Loading YOLOv8s model...")
model = YOLO("yolov8s.pt")
print("YOLOv8s model loaded.")

cap = cv2.VideoCapture(0)
speech_queue = Queue()
spoken_objects = {}
SPEECH_INTERVAL = 5
CONF_THRESHOLD = 0.6
MIN_BOX_SIZE = 60

IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import pythoncom, win32com.client

def speak_worker():
    if not IS_WINDOWS:
        return
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    while True:
        msg = speech_queue.get()
        if msg is None:
            break
        speaker.Speak(msg, 1)
        speech_queue.task_done()

if IS_WINDOWS:
    threading.Thread(target=speak_worker, daemon=True).start()

def get_distance(h, fh): return round(2 * fh / h, 1) if h else 0
def get_position(x, fw):
    if x < fw/3: return "left"
    elif x < 2*fw/3: return "center"
    else: return "right"

def generate_frames():
    global spoken_objects
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        fh, fw = frame.shape[:2]
        results = model(frame, verbose=False)[0]
        current = {}
        for det in results.boxes:
            conf = float(det.conf[0].cpu().numpy())
            if conf < CONF_THRESHOLD: continue
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            cls = model.names[int(det.cls[0].cpu().numpy())]
            bh = y2 - y1
            if bh < MIN_BOX_SIZE: continue
            dist = get_distance(bh, fh)
            pos = get_position((x1+x2)/2, fw)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{cls} {dist}m {pos}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            current[cls] = (dist,pos)
        now = time.time()
        for obj,(d,p) in current.items():
            if IS_WINDOWS and now - spoken_objects.get(obj,0) > SPEECH_INTERVAL:
                speech_queue.put(f"{obj} ahead at {d} meters on the {p}")
                spoken_objects[obj] = now
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n")

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")

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
def shutdown():
    cap.release(); cv2.destroyAllWindows(); speech_queue.put(None)

