from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load YOLOv8s model once at startup
print("Loading YOLOv8s model...")
model = YOLO("yolov8s.pt")
print("YOLOv8s model loaded successfully.")

# Serve frontend files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def index():
    # serve the index.html file from frontend folder
    return HTMLResponse(open("frontend/index.html").read())


def get_distance(box_height, frame_height):
    """Estimate distance based on bounding box height."""
    if box_height == 0:
        return 0
    return round(2 * frame_height / box_height, 1)


def get_position(x_center, frame_width):
    """Determine object position (left, center, right)."""
    if x_center < frame_width / 3:
        return "left"
    elif x_center < frame_width * 2 / 3:
        return "center"
    else:
        return "right"


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """Receive a camera frame, run YOLO detection, and return results."""
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
        cls_id = int(det.cls[0].cpu().numpy())
        cls_name = model.names[cls_id]

        box_height = y2 - y1
        if box_height < 60:
            continue

        distance = get_distance(box_height, frame_height)
        x_center = (x1 + x2) / 2
        position = get_position(x_center, frame_width)

        detections.append({
            "object": cls_name,
            "distance": distance,
            "position": position
        })

    return JSONResponse({"detections": detections})
