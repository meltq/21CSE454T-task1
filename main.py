import cv2
import os
import matplotlib.pyplot as plt

# --- 1. SETUP PATHS ---
IMAGE_DIR = 'data/without_mask/'

# Path to the pre-trained Haar Cascade XML file.
# This file comes with the opencv-contrib-python installation.
# The cv2.data.haarcascades gives us the path to where these files are stored.
HAAR_CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# --- 2. LOAD THE CLASSIFIER ---
# Create a cascade classifier object.
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

if face_cascade.empty():
    print("Error: Could not load Haar Cascade classifier.")
    print(f"Searched at path: {HAAR_CASCADE_PATH}")
    exit()

# --- 3. IMAGE PROCESSING LOOP ---
# Get a list of all image files in the directory.
try:
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images_to_process = image_files[:6]
    if not images_to_process:
        print(f"No images found in the directory: {IMAGE_DIR}")
        exit()
except FileNotFoundError:
    print(f"Error: The directory '{IMAGE_DIR}' does not exist.")
    print("Please create the folder and add images from the Kaggle dataset.")
    exit()

print(f"Found {len(images_to_process)} images to process. Starting detection...")

# Prepare a plot to display results in a grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten() # Flatten the 2x3 grid into a 1D array for easy iteration

for i, filename in enumerate(images_to_process):
    # Construct the full path to the image
    img_path = os.path.join(IMAGE_DIR, filename)
    
    # --- 4. LOAD AND PREPARE THE IMAGE ---
    # Read the image using OpenCV. It reads in BGR (Blue, Green, Red) format.
    image_bgr = cv2.imread(img_path)
    
    if image_bgr is None:
        print(f"Warning: Could not read image {filename}. Skipping.")
        continue

    # Convert the BGR image to RGB for correct color display with Matplotlib
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert the image to grayscale for the face detection algorithm
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- 5. DETECT FACES ---
    # Use the classifier's detectMultiScale function.
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"  - Detected {len(faces)} face(s) in {filename}")

    # --- 6. DRAW BOUNDING BOXES ---
    # Loop through the coordinates (x, y, w, h) for each detected face.
    for (x, y, w, h) in faces:
        # Draw a green rectangle on the original *color* image.
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # --- 7. DISPLAY THE RESULT ---
    ax = axes[i]
    ax.imshow(image_rgb)
    ax.set_title(f"Detected {len(faces)} faces\n({filename})")
    ax.axis('off') # Hide the axes

# Hide any unused subplots
for j in range(len(images_to_process), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.suptitle("Face Detection Results", fontsize=16, y=1.02)
plt.show()
    
plt.savefig('face_detection_results.png')

print("\nProcessing complete.")
