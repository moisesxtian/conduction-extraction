from ultralytics import YOLO
import numpy as np
import cv2
import os
from PIL import Image
from paddleocr import TextRecognition
import logging

# Suppress PaddleOCR welcome message and unnecessary warnings
# Set to WARNING or ERROR to suppress INFO messages
logging.basicConfig(
    level=logging.WARNING, # Changed to WARNING to suppress INFO logs like "Use PaddlePaddle with GPU version."
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load YOLO models
model = YOLO("models/segmentation-v3.pt")  # Conduction sticker detection
text_model = YOLO("models/text-detect-v2.pt")  # Text detection model
RESULTS_FOLDER = "results"

# Ensure the results folder exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def warp_conduction(image_path: str):
    """
    Warps the conduction sticker from the detected mask and saves it.
    Returns the warped image as a NumPy array or None if detection/warping fails.
    """
    print("⏳ Loading model and processing image for warping...")

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Image not loaded from {image_path}. Check the file path.")
        return None

    results = model(source=image, conf=0.5, verbose=False)

    warped_image = None
    for result in results:
        if result.masks is not None:
            for mask in result.masks.xy:
                mask = np.array(mask, dtype=np.int32)

                if len(mask) < 4:
                    print("⚠️ Not enough points in mask to form a valid rectangle.")
                    continue

                sum_coords = mask[:, 0] + mask[:, 1]
                diff_coords = mask[:, 1] - mask[:, 0]

                # Robust check for array bounds
                if not (len(mask) > np.argmin(sum_coords) and
                        len(mask) > np.argmin(diff_coords) and
                        len(mask) > np.argmax(sum_coords) and
                        len(mask) > np.argmax(diff_coords)):
                    print("⚠️ Mask coordinates indexing issue. Skipping warp.")
                    continue

                rect = np.array([
                    mask[np.argmin(sum_coords)],
                    mask[np.argmin(diff_coords)],
                    mask[np.argmax(sum_coords)],
                    mask[np.argmax(diff_coords)]
                ], dtype=np.float32)

                width, height = 900, 600
                dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

                M = cv2.getPerspectiveTransform(rect, dst_pts)
                warped = cv2.warpPerspective(image, M, (width, height))

                warped_path = os.path.join(RESULTS_FOLDER, "warped.jpg")
                cv2.imwrite(warped_path, warped)
                print(f"✅ Warped image saved at: {warped_path}")

                warped_image = warped
                break
        if warped_image is not None:
            break

    if warped_image is None:
        print("⚠️ No suitable mask found for warping.")
    return warped_image

def crop_text_regions(image_np: np.ndarray):

    if image_np is None:
        print("❌ Error: Input image for cropping is None.")
        return {"upper": None, "lower": None, "horizontal": None}

    print("✂️ Detecting and cropping text regions...")

    results = text_model.predict(source=image_np, conf=0.1, verbose=False)

    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    cropped_images = {"upper": None, "lower": None, "horizontal": None}

    for result in results:
        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)

                cropped_img_np = img_rgb[y1:y2, x1:x2]

                if cropped_img_np.size == 0 or cropped_img_np.shape[0] == 0 or cropped_img_np.shape[1] == 0:
                    print(f"⚠️ Skipping empty or invalid crop for {class_name} at [{x1},{y1},{x2},{y2}].")
                    continue

                cropped_pil = Image.fromarray(cropped_img_np)

                if class_name == "vertical-text":
                    height = cropped_img_np.shape[0]
                    mid = max(1, height // 2)

                    upper_pil = Image.fromarray(cropped_img_np[:mid, :])
                    lower_pil = Image.fromarray(cropped_img_np[mid:, :])

                    cropped_images["upper"] = upper_pil
                    cropped_images["lower"] = lower_pil

                elif class_name == "horizontal-text":
                    cropped_images["horizontal"] = cropped_pil
        else:
            print("No text detection boxes found in this result.")
    return cropped_images

def merge_images(cropped_images: dict, fixed_height: int = 100):
    """
    Merges extracted text images into a single horizontal image.
    Resizes images to a fixed height while maintaining aspect ratio.
    Returns the merged image as a PIL Image or None if merging fails.
    """
    print("➕ Merging extracted text images...")

    image_files = [img for img in cropped_images.values() if img is not None and isinstance(img, Image.Image)]

    if not image_files:
        print("❌ No valid cropped images available for merging.")
        return None

    resized_images = []
    for img in image_files:
        aspect_ratio = img.width / img.height
        new_width = int(aspect_ratio * fixed_height)
        if new_width == 0:
            print(f"⚠️ Skipping image with calculated zero width after resize: {img.size}")
            continue
        resized_images.append(img.resize((new_width, fixed_height), Image.LANCZOS))

    if not resized_images:
        print("❌ No valid images remain after resizing for merging.")
        return None

    total_width = sum(img.width for img in resized_images)
    merged_image = Image.new("RGB", (total_width, fixed_height))

    x_offset = 0
    for img in resized_images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width

    merged_image_path = os.path.join(RESULTS_FOLDER, "merged_for_ocr.jpg")
    merged_image.save(merged_image_path)

    print(f"✅ Merged image successfully created at: {merged_image_path}")
    print(f"📏 Merged Image Size: {merged_image.size}")

    return merged_image

def perform_ocr(image_for_ocr: Image.Image, confidence_threshold: float = 0.2):

    if image_for_ocr is None:
        return {"error": "Input image for OCR is None"}

    image_np = np.array(image_for_ocr)

    try:

        reader = TextRecognition()

        results = reader.predict(input=image_np)
        for result in results:
            extracted_text = result['rec_text']
            print(f"Detected text: {result['rec_text']} with confidence {result['rec_score']}")

        if not extracted_text:
            print("⚠️ No text detected above the confidence threshold or no text recognized.")
            return {"extracted_text": ""}
        else:
            return {"extracted_text": extracted_text}
    except Exception as e:
        print(f"❌ An error occurred during OCR: {e}")
        return {"error": f"OCR failed: {e}"}

# --- Example Usage ---
if __name__ == "__main__":
    test_image_path = "test_image_for_warp.jpg"
    if not os.path.exists(test_image_path):
        print(f"Generating a dummy image: {test_image_path} for testing. Please replace with a real image for actual use.")
        dummy_image = np.zeros((800, 1200, 3), dtype=np.uint8)
        # Add some text to the dummy image for OCR to try and detect
        cv2.putText(dummy_image, "SAMPLE CONDUCTION", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(dummy_image, "0123456789", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        cv2.imwrite(test_image_path, dummy_image)

    # Step 1: Warp the conduction sticker
    warped_sticker_np = warp_conduction(test_image_path)

    if warped_sticker_np is not None:
        # Step 2: Crop text regions from the warped sticker
        cropped_text_regions = crop_text_regions(warped_sticker_np)

        # Step 3: Merge the cropped text images
        merged_ocr_image = merge_images(cropped_text_regions)

        if merged_ocr_image is not None:
            # Step 4: Perform OCR on the merged image
            final_ocr_result = perform_ocr(merged_ocr_image)
            print("\n--- Final OCR Result ---")
            print(final_ocr_result)
        else:
            print("\n--- OCR Skipped: Failed to merge images. ---")
    else:
        print("\n--- Processing stopped: Failed to warp conduction sticker. ---")