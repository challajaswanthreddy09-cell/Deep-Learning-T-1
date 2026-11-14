
import io
import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from ultralytics import YOLO

# Initialize FastAPI
app = FastAPI(title="Indian Traffic Detection API")
PROJECT_DIR = Path(__file__).resolve().parent
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_FRAMES = 3000
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}

# Allow frontend communication during local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Detection-Summary"],
)

# Load YOLOv8 model
model = YOLO(str(PROJECT_DIR / "yolov8n.pt"))


@app.get("/")
async def home():
    return FileResponse(PROJECT_DIR / "index.html")


@app.get("/health")
async def health():
    return {
        "message": "VisionPort Traffic Detection API is running",
        "status": "Active"
    }


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):

    # Validate uploaded image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    # Read uploaded image
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    # Convert image bytes to OpenCV format
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to process the uploaded image."
        )

    try:
        # Perform YOLO object detection
        results = model.predict(
            source=img,
            conf=0.25,
            verbose=False
        )

        # Annotate detected objects
        annotated_img = results[0].plot()

        # Convert annotated image to JPEG
        success, encoded_img = cv2.imencode(
            ".jpg",
            annotated_img
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Image encoding failed."
            )

        # Return processed image
        return StreamingResponse(
            io.BytesIO(encoded_img.tobytes()),
            media_type="image/jpeg"
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection error: {str(e)}"
        )


def analyze_video_file(input_path: Path, output_path: Path) -> Counter[str]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise HTTPException(
            status_code=400,
            detail="Unable to open the uploaded video."
        )

    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        counts: Counter[str] = Counter()
        frame_count = 0
        success, frame = capture.read()
        if not success:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video contains no readable frames."
            )

        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise HTTPException(
                status_code=500,
                detail="Unable to create the annotated video."
            )

        while success:
            frame_count += 1
            if frame_count > MAX_VIDEO_FRAMES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Videos are limited to {MAX_VIDEO_FRAMES} frames."
                )

            result = model.predict(source=frame, conf=0.25, verbose=False)[0]
            if result.boxes is not None:
                for class_id in result.boxes.cls.tolist():
                    counts[result.names[int(class_id)]] += 1
            writer.write(result.plot())
            success, frame = capture.read()

        return counts
    finally:
        capture.release()
        if writer is not None:
            writer.release()


@app.post("/analyze-video/")
def analyze_video(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if not (file.content_type or "").startswith("video/") and suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a supported video file."
        )
    if suffix not in VIDEO_SUFFIXES:
        suffix = ".mp4"

    contents = file.file.read(MAX_VIDEO_BYTES + 1)
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty."
        )
    if len(contents) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Video uploads are limited to 100 MB."
        )

    input_path = None
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as input_file:
            input_path = Path(input_file.name)
            input_file.write(contents)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
            output_path = Path(output_file.name)

        counts = analyze_video_file(input_path, output_path)
        response = FileResponse(
            output_path,
            media_type="video/mp4",
            filename="visionport-analysis.mp4",
            headers={
                "X-Detection-Summary": json.dumps(
                    dict(counts),
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            },
            background=BackgroundTask(output_path.unlink, missing_ok=True),
        )
        output_path = None
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis error: {str(e)}"
        )
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
