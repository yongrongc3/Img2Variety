import os
import cv2
import numpy as np

# Define the mixing function
def mixgen(image1, image2, lam=0.5):
    return lam * image1 + (1 - lam) * image2

# List of adjacent stages for fusion
adjacent_stages = [('S1', 'S2'), ('S2', 'S3'), ('S3', 'S4'), ('S4', 'S5'), 
                   ('S5', 'S6'), ('S6', 'S7'), ('S7', 'S8'), ('S8', 'S9')]

# Root directory
root_folder = 'J:/Integration_Maize/Photograph-HTP/Maize_mixall_613/train'

# Traverse each subdirectory
for subdir, _, files in os.walk(root_folder):
    # Filter and group files by category
    grouped_files = {}
    
    for file in files:
        if file.endswith('.png'):
            base_name = file.rsplit('.', 1)[0]  # Remove the extension
            parts = base_name.split(' ')       # Split by spaces

            # Extract category, view, and stage
            category = parts[0]
            view = parts[1].split('_')[0]
            stage = parts[1].split('_')[1]
            
            # Group by category
            key = (category, view, stage)
            if category not in grouped_files:
                grouped_files[category] = []
            grouped_files[category].append((file, view, stage))

    # Apply fusion on grouped files
    for category, files_list in grouped_files.items():
        # Sort files by view and stage
        files_list.sort(key=lambda x: (x[1], x[2]))

        # Convert stage to set for easy lookup
        stage_set = set([f[2] for f in files_list])
        
        # Perform fusion based on rules
        for i, (file1, view1, stage1) in enumerate(files_list):
            for j in range(i + 1, len(files_list)):
                file2, view2, stage2 = files_list[j]
                
                # 1. Same view, adjacent stages
                if view1 == view2 and (stage1, stage2) in adjacent_stages:
                    path1 = os.path.join(subdir, file1)
                    path2 = os.path.join(subdir, file2)
                    image1 = cv2.imread(path1)
                    image2 = cv2.imread(path2)
                    mixed_image = mixgen(image1, image2)
                    
                    # Save mixed image
                    mixed_filename = f"{file1.rsplit('.', 1)[0]}_{file2.rsplit('.', 1)[0]}.png"
                    mixed_filepath = os.path.join(subdir, mixed_filename)
                    cv2.imwrite(mixed_filepath, mixed_image)
                    print(f"Saved: {mixed_filename}")

                # 2. Same stage, different views
                elif stage1 == stage2 and view1 != view2:
                    path1 = os.path.join(subdir, file1)
                    path2 = os.path.join(subdir, file2)
                    image1 = cv2.imread(path1)
                    image2 = cv2.imread(path2)
                    mixed_image = mixgen(image1, image2)

                    # Save mixed image
                    mixed_filename = f"{file1.rsplit('.', 1)[0]}_{file2.rsplit('.', 1)[0]}.png"
                    mixed_filepath = os.path.join(subdir, mixed_filename)
                    cv2.imwrite(mixed_filepath, mixed_image)
                    print(f"Saved: {mixed_filename}")

                # 3. Adjacent stages, different views
                elif view1 != view2 and (stage1, stage2) in adjacent_stages:
                    path1 = os.path.join(subdir, file1)
                    path2 = os.path.join(subdir, file2)
                    image1 = cv2.imread(path1)
                    image2 = cv2.imread(path2)
                    mixed_image = mixgen(image1, image2)

                    # Save mixed image
                    mixed_filename = f"{file1.rsplit('.', 1)[0]}_{file2.rsplit('.', 1)[0]}.png"
                    mixed_filepath = os.path.join(subdir, mixed_filename)
                    cv2.imwrite(mixed_filepath, mixed_image)
                    print(f"Saved: {mixed_filename}")
