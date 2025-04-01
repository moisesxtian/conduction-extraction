import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from collections import defaultdict
import shutil

# Define paths
base_dir = 'dataset'
train_image_dir = os.path.join(base_dir, 'train/images')
train_label_dir = os.path.join(base_dir, 'train/labels')
aug_train_image_dir = os.path.join(base_dir, 'train/aug_images')
aug_train_label_dir = os.path.join(base_dir, 'train/aug_labels')

# Create directories for augmented data if they don't exist
os.makedirs(aug_train_image_dir, exist_ok=True)
os.makedirs(aug_train_label_dir, exist_ok=True)

# Define augmentation pipeline for car logo detection
def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Function to read YOLO format annotations and normalize bounding boxes
def read_yolo_annotation(file_path, image_width, image_height):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        # Convert to YOLO format (normalized center coordinates)
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        bboxes.append([x_center, y_center, width, height])  # already normalized
        class_labels.append(class_id)
    return bboxes, class_labels

# Function to save YOLO format annotations
def save_yolo_annotation(file_path, bboxes, class_labels, image_width, image_height):
    with open(file_path, 'w') as file:
        for bbox, class_label in zip(bboxes, class_labels):
            x_center = bbox[0]
            y_center = bbox[1]
            width = bbox[2]
            height = bbox[3]
            file.write(f"{class_label} {x_center} {y_center} {width} {height}\n")

# Function to analyze class distribution
def analyze_class_distribution(image_files):
    class_count = defaultdict(int)
    for image_file in tqdm(image_files, desc="Analyzing class distribution"):
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(train_label_dir, label_file)
        if not os.path.exists(label_path):
            continue
        _, class_labels = read_yolo_annotation(label_path, 1, 1)  # Image size doesn't matter here
        for class_id in class_labels:
            class_count[class_id] += 1
    return class_count

# Function to augment images and annotations
def augment_images_and_annotations(image_files, minority_classes):
    augmentations = get_augmentation_pipeline()
    for image_file in tqdm(image_files, desc="Augmenting images"):
        # Read image
        image_path = os.path.join(train_image_dir, image_file)
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Read corresponding annotation
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(train_label_dir, label_file)
        if not os.path.exists(label_path):
            continue
        bboxes, class_labels = read_yolo_annotation(label_path, image_width, image_height)

        # Check if the image contains any minority class (for logos or car models)
        if any(cls in minority_classes for cls in class_labels):
            # Apply augmentations
            transformed = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

            # Save augmented image
            aug_image_path = os.path.join(aug_train_image_dir, image_file)
            cv2.imwrite(aug_image_path, transformed_image)

            # Save augmented annotation
            aug_label_path = os.path.join(aug_train_label_dir, label_file)
            save_yolo_annotation(aug_label_path, transformed_bboxes, transformed_class_labels, image_width, image_height)

            # Now move the augmented data to the original train directories to merge the data
            # Move the augmented image to the train image directory
            shutil.move(aug_image_path, os.path.join(train_image_dir, image_file))

            # Move the augmented label to the train label directory
            shutil.move(aug_label_path, os.path.join(train_label_dir, label_file))

if __name__ == "__main__":
    # Get list of image files in the training set
    image_files = [f for f in os.listdir(train_image_dir) if f.endswith(('.jpg', '.png'))]
    # Analyze class distribution to identify minority classes
    class_count = analyze_class_distribution(image_files)
    total_images = len(image_files)
    # Define a threshold for minority class identification (e.g., classes with less than 10% of the total images)
    threshold = 0.1 * total_images
    minority_classes = {cls_id for cls_id, count in class_count.items() if count < threshold}
    # Augment images containing minority classes
    augment_images_and_annotations(image_files, minority_classes)
