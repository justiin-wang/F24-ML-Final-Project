import os
import requests
import zipfile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference
from models.face.inference import YOLOInference as FaceInference
import cv2
import copy

# Define URLs and common directory for storing models
model_urls = {
    'segmentation': 'https://storage.googleapis.com/fishial-ml-resources/segmentator_fpn_res18_416_1.zip',
    'detection': 'https://storage.googleapis.com/fishial-ml-resources/detector_v10_m3.zip',
}
output_dir = './models'

# Step 1: Download and Extract Models
os.makedirs(output_dir, exist_ok=True)

def download_and_extract_models(model_urls, output_dir):
    for model_name, url in model_urls.items():
        print(f"Downloading {model_name} model...")
        zip_path = os.path.join(output_dir, f"{model_name}.zip")
        
        # Download the model
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract the model
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output_dir, model_name))
        print(f"{model_name} model downloaded and extracted!")

#download_and_extract_models(model_urls, output_dir)

# Define model directories
MODEL_DIRS = {
    'segmentation': os.path.join(output_dir, 'segmentation'),
    'detection': os.path.join(output_dir, 'detection'),
    'face': os.path.join(output_dir, 'face'),
}

# Initialize models
segmentator = Inference(
    model_path=os.path.join(MODEL_DIRS['segmentation'], 'model.ts'),
    image_size=416
)

detector = YOLOInference(
    model_path=os.path.join(MODEL_DIRS['detection'], 'model.ts'),
    imsz=(640, 640),
    conf_threshold=0.9,
    nms_threshold=0.3,
    yolo_ver='v10'
)


# Path to the input image
fish_path = 'image2.png'  # Replace with the path to your image

# Step 1: Load and prepare the image
fish_bgr_np = cv2.imread(fish_path)  # Load image as BGR
visulize_img_bgr = fish_bgr_np.copy()  # Make a copy for visualization

# Convert the image to RGB for inference
visulize_img_rgb = cv2.cvtColor(fish_bgr_np, cv2.COLOR_BGR2RGB)
visulize_img = copy.deepcopy(visulize_img_rgb)  # Deep copy for visualization

# Step 3: Run object detection
boxes = detector.predict(visulize_img_rgb)[0]

# Process each detected fish
for box in boxes:
    # Crop the detected fish region (BGR and RGB)
    cropped_fish_bgr = box.get_mask_BGR()
    cropped_fish_rgb = box.get_mask_RGB()

    # Run segmentation on the cropped fish
    segmented_polygons = segmentator.predict(cropped_fish_bgr)[0]

    # Apply the segmentation mask to the cropped fish
    cropped_fish_mask = segmented_polygons.mask_polygon(cropped_fish_rgb)

    # Draw segmentation polygons on the original image
    segmented_polygons.move_to(box.x1, box.y1)  # Adjust polygon position to the full image
    segmented_polygons.draw_polygon(visulize_img)
    
    plt.imshow(cropped_fish_mask)
    plt.show()

# Step 4: Visualize final result
plt.figure(figsize=(10, 6))
plt.imshow(visulize_img)
plt.title("Fish Detection and Segmentation")
plt.axis("off")
plt.show()