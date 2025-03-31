from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import os
from pydantic import BaseModel
from PIL import Image
from paddleocr import PaddleOCR

# Load YOLO models
model = YOLO("models/segmentation-v3.pt")  # Conduction sticker detection
text_model = YOLO("models/text-detect-v2.pt")  # Text detection model
RESULTS_FOLDER = "results"

# Ensure the results folder exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def warp_conduction(output_path):
    """ Warps the conduction sticker from the detected mask. """
    print("Loading Model...")

    # Read the image
    image = cv2.imread(output_path)
    if image is None:
        print("Error: Image not loaded. Check the file path.")
        return None

    results = model(source=output_path, conf=0.5)
    for result in results:
        if result.masks is not None:
            for mask in result.masks.xy:
                mask = np.array(mask, dtype=np.int32)

                # Find bounding rectangle corners
                sum_coords = mask[:, 0] + mask[:, 1]
                diff_coords = mask[:, 1] - mask[:, 0]

                rect = np.array([
                    mask[np.argmin(sum_coords)],  # Top-left
                    mask[np.argmin(diff_coords)],  # Top-right
                    mask[np.argmax(sum_coords)],  # Bottom-right
                    mask[np.argmax(diff_coords)]   # Bottom-left
                ], dtype=np.float32)

                # Fixed rectangle for transformation
                width, height = 900, 600
                dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

                # Warp perspective
                M = cv2.getPerspectiveTransform(rect, dst_pts)
                warped = cv2.warpPerspective(image, M, (width, height))  # Use the loaded image

                # Save warped image
                warped_path = os.path.join(RESULTS_FOLDER, "warped.jpg")
                cv2.imwrite(warped_path, warped)
                print(f"Warped image saved at: {warped_path}")

                return warped  # Return the warped image
    return None

def crop_text_regions(image):
    """Detects text regions and crops them. Returns PIL images."""
    if isinstance(image, str):  
        img = cv2.imread(image)
        if img is None:
            print("❌ Error: Could not load image.")
            return None
    else:
        img = image  # Use provided NumPy image

    results = text_model.predict(source=img, conf=0.1, save=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

    cropped_images = {"upper": None, "lower": None, "horizontal": None}

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())  
            class_name = result.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped_img = img[y1:y2, x1:x2]

            if cropped_img.size == 0:
                print(f"⚠️ Skipping empty crop for {class_name}")
                continue

            cropped_pil = Image.fromarray(cropped_img)

            if class_name == "vertical-text":
                height = cropped_img.shape[0]
                mid = height // 2

                if mid == 0:
                    print("⚠️ Skipping very small vertical-text region")
                    continue  

                upper_pil = Image.fromarray(cropped_img[:mid, :])
                lower_pil = Image.fromarray(cropped_img[mid:, :])

                upper_path = f"{RESULTS_FOLDER}/image_upper.jpg"
                lower_path = f"{RESULTS_FOLDER}/image_lower.jpg"

                #upper_pil.save(upper_path)
                #lower_pil.save(lower_path)

                cropped_images["upper"] = upper_pil  
                cropped_images["lower"] = lower_pil  

            elif class_name == "horizontal-text":
                horizontal_path = f"{RESULTS_FOLDER}/image_horizontal.jpg"
                #cropped_pil.save(horizontal_path)

                cropped_images["horizontal"] = cropped_pil  

    print(f"🟢 Cropped Images: {cropped_images}")

    return cropped_images
def merge_images(cropped_images, fixed_height=100):
    """Merges extracted text images into a single image."""
    image_files = [img for img in cropped_images.values() if img is not None]

    if not image_files:
        print("❌ No cropped images available for merging.")
        return None

    resized_images = []
    for img in image_files:
        if not isinstance(img, Image.Image):
            print(f"⚠️ Skipping non-PIL image: {type(img)}")
            continue
        aspect_ratio = img.width / img.height
        new_width = int(aspect_ratio * fixed_height)
        resized_images.append(img.resize((new_width, fixed_height), Image.LANCZOS))

    if not resized_images:
        print("❌ No valid images after resizing.")
        return None

    total_width = sum(img.width for img in resized_images)
    merged_image = Image.new("RGB", (total_width, fixed_height))

    x_offset = 0
    for img in resized_images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width

    merged_image_path = os.path.join(RESULTS_FOLDER, "merged_image.jpg")
    merged_image.save(merged_image_path)

    print(f"✅ Merged image successfully created at: {merged_image_path}")
    
    print(f"📏 Merged Image Size: {merged_image.size}")

    return merged_image

def perform_ocr(image, confidence_threshold=0.2):
    """ Performs OCR on the given image and extracts text. """
    if isinstance(image, Image.Image):  # Convert PIL Image to NumPy
        image = np.array(image)
    
    if not isinstance(image, np.ndarray):
        print("Error: Invalid image format for OCR.")
        return {"error": "Invalid image format"}

    reader = PaddleOCR(lang="en")
    results = reader.ocr(image, cls=False)

    if not results or not results[0]:
        print("No text detected.")
        return {"extracted_text": ""}

    extracted = " ".join([det[1][0] for det in results[0] if det[1][1] >= confidence_threshold])
    return {"extracted_text": extracted}
