import os
import requests
import zipfile
import zipfile
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


# Define input and output folders
#Change the folders to the corresponding fish
input_folder = './fish'  # Folder containing the input images
output_folder = './catfish_masks'  # Folder to save the segmented images
os.makedirs(output_folder, exist_ok=True)
try:
    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):  # Process image files only
            fish_path = os.path.join(input_folder, filename)
            base, ext = os.path.splitext(filename)

            new_filename = f"{base}_mask{ext}"
            if os.path.isfile(os.path.join(output_folder, new_filename)):
                print("Already in masks, continuing...")
                continue
            print(f"Processing {fish_path}...")
            
            # Load the image
            fish_bgr_np = cv2.imread(fish_path)
            visulize_img_bgr = fish_bgr_np.copy()

            # Convert the image to RGB for inference
            visulize_img_rgb = cv2.cvtColor(fish_bgr_np, cv2.COLOR_BGR2RGB)
            visulize_img_rgb = np.array(visulize_img_rgb)
            visulize_img = copy.deepcopy(visulize_img_rgb)

            # Run object detection
            box = detector.predict(visulize_img_rgb)[0][0]
                
            cropped_fish_bgr = box.get_mask_BGR()
            cropped_fish_rgb = box.get_mask_RGB()
            segmented_polygons = segmentator.predict(cropped_fish_bgr)[0]

            # Apply segmentation
            cropped_fish_mask = segmented_polygons.mask_polygon(cropped_fish_rgb)
            segmented_polygons.move_to(box.x1, box.y1)
            segmented_polygons.draw_polygon(visulize_img)
            
            # Save the segmented image to the output folder
            output_path = os.path.join(output_folder, new_filename)
            segmented_bgr = cv2.cvtColor(cropped_fish_mask, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
            
            cv2.imwrite(output_path, segmented_bgr)

            print(f"Segmented image saved to {output_path}")
except EOFError:
    print("Error has occured at: " + filename)
    
    
print("Processing complete!")