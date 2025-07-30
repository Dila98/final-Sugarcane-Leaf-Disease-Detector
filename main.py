from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
from ultralytics import YOLO

# Initialize FastAPI
app = FastAPI()

# Enable CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving annotated images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLOv8 model
model = YOLO("best.pt")  # Ensure this file is in the root directory and committed

# Ensure static directory exists
os.makedirs("static", exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file to disk
        file_path = f"static/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read image using OpenCV
        image = cv2.imread(file_path)

        # Predict using YOLOv8
        results = model.predict(image)

        # Annotate and save the image
        for r in results:
            annotated_frame = r.plot()
            annotated_filename = f"static/annotated_{file.filename}"
            cv2.imwrite(annotated_filename, annotated_frame)

        # Return annotated image URL
        return JSONResponse(
            content={
                "message": "Prediction successful",
                "annotated_image_url": f"https://final-sugarcane-leaf-disease-detector.onrender.com/static/annotated_{file.filename}"
            }
        )

    except Exception as e:
        return JSONResponse(
            content={"message": f"Prediction failed: {str(e)}"},
            status_code=500
        )
