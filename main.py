from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import shutil
from text_extraction import warp_conduction, crop_text_regions, merge_images, perform_ocr
from car_classifier import detect_and_classify
# Ensure upload and results directories exist
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "OCR API is running"}

@app.post("/detect")
async def detect_conduction_sticker(file: UploadFile = File(...)):
    # Save uploaded file
    ext = os.path.splitext(file.filename)[1]  # Extracts file extension (e.g., .jpg)
    file_path = os.path.join(UPLOAD_FOLDER, f"uploaded-image{ext}")  # Adds extension
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process image
    warped = warp_conduction(file_path)
    if warped is None:
        return {"error": "Warping failed"}

    cropped = crop_text_regions(warped)
    if cropped is None:
        return {"error": "Text detection failed"}

    merged = merge_images(cropped)
    if merged is None:
        return {"error": "No merged image generated"}

    ocr_result = perform_ocr(merged)
    
    return ocr_result

@app.post("/classify")
async def classify_vehicle(file: UploadFile = File(...)):  # Renamed function to avoid recursion
    # Save uploaded file
    ext = os.path.splitext(file.filename)[1]  # Extracts file extension (e.g., .jpg)
    file_path = os.path.join(UPLOAD_FOLDER, f"uploaded-image{ext}")  # Adds extension
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call the actual classification function
    make_model = detect_and_classify(file_path)  # Use renamed import to avoid recursion
    return make_model


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
