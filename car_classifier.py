from ultralytics import YOLO
import cv2

# Load YOLO models
car_detector = YOLO("models/yolov8_car_detection.pt")  # Car detection model
brand_detector = YOLO("models/yolov8_brand_recognition.pt")  # Brand recognition model
classifiers = {
    "Tesla": YOLO("models/Model Classifier/Tesla_classifier_YOLO.pt"),
    "Toyota": YOLO("models/Model Classifier/Toyota_classifier_YOLO.pt"),
    # Add more brands if needed
}

BRAND_CONF_THRESH = 0.1  # Minimum confidence required for brand recognition
MODEL_CONF_THRESH = 0.5  # Minimum confidence required for model classification

def detect_and_classify(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image.")
        return {"Make": "Unknown", "Model": "Unknown"}

    # Step 1: Car detection
    car_results = car_detector(image, save=True, exist_ok=True)
    
    highest_conf = 0
    best_car = None

    for result in car_results:
        for i in range(len(result.boxes.cls)):
            class_idx = int(result.boxes.cls[i])
            confidence = float(result.boxes.conf[i])
            
            if result.names[class_idx] == "car" and confidence > highest_conf:
                highest_conf = confidence
                best_car = result.boxes.xyxy[i]

    if best_car is None:
        print("No car detected.")
        return {"Make": "Unknown", "Model": "Unknown"}

    x1, y1, x2, y2 = map(int, best_car)
    car_crop = image[y1:y2, x1:x2]
    print(f"Highest confidence car detected: {highest_conf*100:.2f}%")

    # Step 2: Brand recognition
    brand = "Unknown"
    brand_confidence = 0.0
    
    brand_results = brand_detector(car_crop)
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
        "Model": top_model_name
    }

if __name__ == "__main__":
    result = detect_and_classify("test2.jpg")  # Replace with your image path
    print(result)
