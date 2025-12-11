import cv2
import os
import glob
import re

# --- Configuration ---
image_folder = 'C:/Users/hugog/Documents/CFD_CS/proj4/cfd_cs_proj4/field_profiles/'
output_folder = 'C:/Users/hugog/Documents/CFD_CS/proj4/cfd_cs_proj4/'
video_name = 'field_animation_Q3.mp4'

# image_folder = 'C:/Users/hugog/Documents/CFD_CS/proj4/cfd_cs_proj4/scalar_profiles/'
# output_folder = 'C:/Users/hugog/Documents/CFD_CS/proj4/cfd_cs_proj4/'
# video_name = 'scalar_animation_Q0.5.mp4'


fps = 15  # Frames per second (adjust speed here)

# Full path for the video
video_path = os.path.join(output_folder, video_name)

# --- 1. Load and Sort Files ---
# Get all png files
images = glob.glob(os.path.join(image_folder, "*.png"))

# SORTING: This extracts the number from 'field_100.png' to sort numerically
# Otherwise, field_10.png would come before field_2.png
images.sort(key=lambda f: int(re.sub('\D', '', os.path.basename(f))))

if not images:
    print(f"No images found in {image_folder}")
    exit()

print(f"Found {len(images)} images. Processing...")

# --- 2. Setup Video Writer ---
# Read the first image to get dimensions
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define codec (mp4v is standard for .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# --- 3. Write Frames ---
for image in images:
    video.write(cv2.imread(image))

# --- 4. Cleanup ---
cv2.destroyAllWindows()
video.release()

print(f"Animation saved successfully at: {video_path}")