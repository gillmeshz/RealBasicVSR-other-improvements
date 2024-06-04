import os
import shutil
import argparse
from tqdm import tqdm

def split_images_into_folders(input_dir, output_base_dir, images_per_folder=100):
    # Ensure output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get list of all images in input directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
    
    # Split images into multiple folders
    for i in tqdm(range(0, len(image_files), images_per_folder)):
        folder_name = f'folder_{i // images_per_folder + 1}'
        output_folder = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        for image_file in image_files[i:i + images_per_folder]:
            shutil.copy(os.path.join(input_dir, image_file), output_folder)
    
    print(f"Images split into folders successfully.")

def parse_args():
    parser = argparse.ArgumentParser(description='Split images into folders.')
    parser.add_argument('input_dir', help='Directory of the input images')
    parser.add_argument('output_base_dir', help='Base directory for the output folders')
    parser.add_argument('--images_per_folder', type=int, default=100, help='Number of images per folder')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    split_images_into_folders(args.input_dir, args.output_base_dir, args.images_per_folder)
