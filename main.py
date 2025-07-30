from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
from ultralytics import YOLO
from fastapi.staticfiles import StaticFiles


app = FastAPI()

# app.mount("/", StaticFiles(directory="build", html=True), name="frontend")
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="build", html=True), name="frontend")


# CORS for frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("best.pt")  # Fixed: just the filename since it's in the same folder

# Create folders if not exist
os.makedirs("temp", exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file to temp/
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check valid image format
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(content={"disease": "Invalid image format ❌"}, status_code=400)

    # Predict
    results = model.predict(source=file_location, save=False)
    annotated_img = results[0].plot()

    # Save annotated image
    annotated_path = f"static/annotated_{file.filename}"
    cv2.imwrite(annotated_path, annotated_img[:, :, ::-1])  # RGB → BGR

    # Get class names
    boxes = results[0].boxes
    if boxes:
        detected_labels = [results[0].names[int(box.cls)] for box in boxes]
        disease_summary = ", ".join(set(detected_labels))
    else:
        disease_summary = "No disease detected"

    return JSONResponse(
        content={
            "disease": disease_summary,
             "annotated_image_url": f"https://sugarcane-leaf-disease-detector.onrender.com/static/annotated_{file.filename}"
        }
    )

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# import shutil
# import os
# import cv2
# from ultralytics import YOLO

# app = FastAPI()

# # Allow all origins (adjust in production!)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load YOLO model (relative path)
# model = YOLO("best.pt")  # Place the model in a `models/` folder

# # Create folders
# os.makedirs("temp", exist_ok=True)
# os.makedirs("static", exist_ok=True)

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # Save uploaded file
#     file_location = f"temp/{file.filename}"
#     with open(file_location, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Validate image
#     if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

#     # Predict
#     results = model.predict(source=file_location, save=False)
#     annotated_img = results[0].plot()

#     # Save annotated image
#     annotated_path = f"static/annotated_{file.filename}"
#     cv2.imwrite(annotated_path, annotated_img[:, :, ::-1])  # RGB → BGR

#     # Get predictions
#     boxes = results[0].boxes
#     disease_summary = (
#         ", ".join({results[0].names[int(box.cls)] for box in boxes})
#         if boxes else "No disease detected"
#     )

#     return JSONResponse(
#         content={
#             "disease": disease_summary,
#             "annotated_image_url": f"/static/annotated_{file.filename}"  # Relative path!
#         }
#     )

# # Serve static files
# app.mount("/static", StaticFiles(directory="static"), name="static")
