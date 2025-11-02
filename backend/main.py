from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI()

# Load YOLO model once
print("Loading YOLOv8s model...")
model_path = os.path.join(os.path.dirname(__file__), "yolov8s.pt")
model = YOLO(model_path)
print("YOLOv8s model loaded successfully.")

# Mount frontend folder
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

@app.get("/")
def index():
    """Serve index.html"""
    file_path = os.path.join(frontend_dir, "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

def get_distance(box_height, frame_height):
    return round(2 * frame_height / box_height, 1) if box_height > 0 else 0

def get_position(x_center, frame_width):
    if x_center < frame_width / 3:
        return "left"
    elif x_center < 2 * frame_width / 3:
        return "center"
    else:
        return "right"

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """Detect objects from uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_height, frame_width = frame.shape[:2]

    results = model(frame, verbose=False)[0]
    detections = []

    for det in results.boxes:
        conf = float(det.conf[0].cpu().numpy())
        if conf < 0.6:
            continue
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        cls = model.names[int(det.cls[0].cpu().numpy())]
        box_height = y2 - y1
        if box_height < 60:
            continue

        distance = get_distance(box_height, frame_height)
        pos = get_position((x1 + x2) / 2, frame_width)

        detections.append({
            "object": cls,
            "distance": distance,
            "position": pos
        })

    return JSONResponse({"detections": detections})
