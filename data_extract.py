import os
from PIL import Image

# Paths
dataset_path = '/Users/krishilparikh/Downloads/archive-5'  # Change this to your dataset path
output_folder = '/Users/krishilparikh/Desktop/Proj/database'  # Folder to save extracted images
labels_file = os.path.join(dataset_path, 'words_new.txt')  # Path to the labels file

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the labels from the labels.txt file
label_map = {}
with open(labels_file, 'r') as f:
    label_lines = f.readlines()
    
    # Assuming the labels are in the format: <image_id> <label>
    for line in label_lines:
        parts = line.strip().split(' ')
        if len(parts) >= 2:
            image_id = parts[0]  # The first part is the image ID
            label = ' '.join(parts[1:])  # The remaining parts are the label
            label_map[image_id] = label  # Store in a dictionary for quick access

# Get already extracted images to avoid duplication
existing_images = {f for f in os.listdir(output_folder) if f.endswith('.png')}

# Function to extract images from a directory
def extract_images_from_directory(directory):
    for root, dirs, files in os.walk(directory):
        print(f"Searching in directory: {root}")  # Print current directory
        for file in files:
            print(f"Found file: {file}")  # Print each file found
            if file.endswith('.png'):
                image_path = os.path.join(root, file)  # Full path to the image
                image_id = os.path.splitext(file)[0]  # Extract the image ID from the filename
                
                # Check if the image is already extracted
                if f'{image_id}.png' in existing_images:
                    print(f"Image {image_id} already extracted, skipping.")  # Image already exists
                    continue  # Skip this image if already extracted
                
                try:
                    # Load the image using PIL
                    with Image.open(image_path) as img:
                        img = img.convert('L')  # Convert to grayscale
                        # Save the image in the new folder
                        new_image_path = os.path.join(output_folder, f'{image_id}.png')
                        img.save(new_image_path)
                        print(f"Extracted and saved image: {new_image_path}")  # Successfully saved image

                    # Save the label corresponding to the image
                    if image_id in label_map:
                        with open(os.path.join(output_folder, 'labels.txt'), 'a') as label_file:
                            label_file.write(f'{image_id} {label_map[image_id]}\n')
                
                except Exception as e:
                    print(f"Failed to process image: {image_path}. Error: {e}")  # Print error if any

print("Starting image extraction...")
extract_images_from_directory(dataset_path)
print("Images extracted and saved successfully!")
