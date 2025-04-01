from ultralytics import YOLO
import cv2

# Load YOLO models
car_detector = YOLO("models/yolov8_car_detection.pt")  # Car detection model
brand_detector = YOLO("models/yolov8_cardetect_v1.pt")  # Brand recognition model

classifiers = {
    "Tesla": YOLO("models/Model Classifier/Tesla_classifier_YOLO.pt"),
    "Toyota": YOLO("models/Model Classifier/Toyota_classifier_YOLO.pt"),
    "Mitsubishi": YOLO("models/Model Classifier/Mitsubishi_classifier_YOLO.pt"),
    "Nissan": YOLO("models/Model Classifier/Nissan_classifier_YOLO.pt"),
    "Audi": YOLO("models/Model Classifier/Audi_classifier_YOLO.pt"),
    # Add more brands if needed
}

BRAND_CONF_THRESH = 0.0  # Minimum confidence required for brand recognition
MODEL_CONF_THRESH = 0.0 # Minimum confidence required for model classification

def detect_and_classify(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image.")
        return {"Make": "Unknown", "Model": "Unknown"}

    # Step 1: Car detection
    car_results = car_detector(image, save=True, exist_ok=True,project="pred", name="car")
    
    largest_area = 0
    best_car = None

    # Loop through all detections and select the one with the largest area
    for result in car_results:
        for i in range(len(result.boxes.cls)):
            class_idx = int(result.boxes.cls[i])
            confidence = float(result.boxes.conf[i])

            # Ensure to check for both "car" and "truck" explicitly
            if result.names[class_idx] in ["car", "truck"]:
                # Get the coordinates of the bounding box
                x1, y1, x2, y2 = result.boxes.xyxy[i]
                # Calculate the area of the bounding box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Update largest_area if this bounding box is larger
                if area > largest_area:
                    largest_area = area
                    best_car = result.boxes.xyxy[i]  # Get the coordinates of the largest bounding box

    if best_car is None:
        print("No car detected.")
        return {"Make": "Unknown", "Model": "Unknown"}

    # Crop the image using the coordinates of the largest bounding box
    x1, y1, x2, y2 = map(int, best_car)
    car_crop = image[y1:y2, x1:x2]
    print(f"Largest bounding box with area {largest_area:.2f} pixels detected.")

    # Step 2: Brand recognition
    brand = "Unknown"
    brand_confidence = 0.0
    
    brand_results = brand_detector(car_crop, save=True, project="pred", exist_ok=True)
    if brand_results and len(brand_results[0].boxes.cls) > 0:
        brand_idx = int(brand_results[0].boxes.cls[0])
        brand = brand_results[0].names[brand_idx]
        brand_confidence = float(brand_results[0].boxes.conf[0])
        
        if brand_confidence < BRAND_CONF_THRESH:
            brand = "Unknown"
    
    print(f"Detected brand: {brand} ({brand_confidence*100:.2f}%)")

    # Step 3: Run model-specific classifier
    top_model_name = "Unknown"
    top_model_conf = 0.0
    
    if brand in classifiers:
        model_result = classifiers[brand](car_crop)
        model_probs = model_result[0].probs.data.tolist()
        
        if model_probs:
            model_idx = model_probs.index(max(model_probs))
            top_model_name = model_result[0].names[model_idx].split("_")[-1]
            top_model_conf = max(model_probs)
            
            if top_model_conf < MODEL_CONF_THRESH:
                top_model_name = "Unknown"
    
    print(f"Detected Model: {top_model_name} ({top_model_conf*100:.2f}%)")
    
    return {
        "Make": brand,
        "Model": top_model_name,
        "Confidence": f"{brand_confidence*100:.2f}%"
    }

if __name__ == "__main__":
    result = detect_and_classify("test2.jpg")  # Replace with your image path
    print(result)
