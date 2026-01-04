"""
Copy Filtered COCO Action Images

This script finds and copies the filtered action images from your COCO dataset,
handling different directory structures and naming conventions.

Usage:
    python copy_filtered_images.py
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

COCO_ROOT = "dataset/coco_complete"
FILTERED_CAPTIONS = "dataset/coco_actions/action_captions.json"
OUTPUT_IMAGES = "dataset/coco_actions/images"

# ============================================================================
# Functions
# ============================================================================

def find_all_image_directories():
    """Find all possible image directories in COCO dataset."""
    possible_dirs = []
    
    print("Searching for image directories in COCO dataset...")
    
    # Common COCO directory names
    common_names = [
        'train2014', 'val2014', 'test2014',
        'train2017', 'val2017', 'test2017',
        'images', 'train', 'val', 'test'
    ]
    
    # Search in COCO root
    for dirname in common_names:
        full_path = os.path.join(COCO_ROOT, dirname)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            # Check if directory contains images
            files = os.listdir(full_path)
            img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if img_files:
                possible_dirs.append(full_path)
                print(f"Found: {full_path} ({len(img_files)} images)")
    
    # Also search nested directories
    for root, dirs, files in os.walk(COCO_ROOT):
        # Skip if already found
        if root in possible_dirs:
            continue
        # Check if directory contains images
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(img_files) > 100:  # Likely an image directory
            possible_dirs.append(root)
            print(f"   Found: {root} ({len(img_files)} images)")
    
    return possible_dirs

def build_image_index(image_dirs):
    """Build index of all available images."""
    print("\nBuilding image index...")
    image_index = {}  # filename -> full_path
    
    for img_dir in tqdm(image_dirs, desc="Indexing directories"):
        if not os.path.exists(img_dir):
            continue
            
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Store both exact filename and without prefix
                image_index[filename] = os.path.join(img_dir, filename)
                
                # Also index without COCO prefix if present
                if filename.startswith('COCO_'):
                    simple_name = filename.replace('COCO_train2014_', '').replace('COCO_val2014_', '')
                    image_index[simple_name] = os.path.join(img_dir, filename)
    
    print(f" Indexed {len(image_index)} unique images")
    return image_index

def load_filtered_captions():
    """Load the filtered caption data."""
    print(f"\nLoading filtered captions from: {FILTERED_CAPTIONS}")
    
    if not os.path.exists(FILTERED_CAPTIONS):
        raise FileNotFoundError(
            f"Filtered captions not found: {FILTERED_CAPTIONS}\n"
            f"Please run 'python filter_action_dataset.py' first"
        )
    
    with open(FILTERED_CAPTIONS, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    print(f" Loaded metadata for {len(images)} filtered images")
    
    return images

def copy_images(images_metadata, image_index):
    """Copy filtered images using the image index."""
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    
    copied = 0
    missing = 0
    missing_files = []
    
    print(f"\nCopying {len(images_metadata)} images to {OUTPUT_IMAGES}...")
    
    for img_meta in tqdm(images_metadata, desc="Copying images"):
        filename = img_meta['file_name']
        dest_path = os.path.join(OUTPUT_IMAGES, filename)
        
        # Skip if already copied
        if os.path.exists(dest_path):
            copied += 1
            continue
        
        # Try to find the image
        found = False
        
        # Try exact filename
        if filename in image_index:
            source_path = image_index[filename]
            shutil.copy2(source_path, dest_path)
            copied += 1
            found = True
        else:
            # Try without COCO prefix
            simple_name = filename.replace('COCO_train2014_', '').replace('COCO_val2014_', '')
            if simple_name in image_index:
                source_path = image_index[simple_name]
                shutil.copy2(source_path, dest_path)
                copied += 1
                found = True
            else:
                # Try adding COCO prefix
                prefixed_name = f"COCO_train2014_{filename}"
                if prefixed_name in image_index:
                    source_path = image_index[prefixed_name]
                    shutil.copy2(source_path, dest_path)
                    copied += 1
                    found = True
        
        if not found:
            missing += 1
            if missing <= 10:  # Store first 10 missing files
                missing_files.append(filename)
    
    print(f"\n Successfully copied: {copied} images")
    
    if missing > 0:
        print(f"Missing: {missing} images")
        if missing_files:
            print(f"\nFirst few missing files:")
            for mf in missing_files:
                print(f"  - {mf}")
        print(f"\nThis may be normal if some images are in test set or unavailable.")
    
    return copied, missing

def verify_copied_images():
    """Verify copied images."""
    if not os.path.exists(OUTPUT_IMAGES):
        return 0
    
    image_files = [f for f in os.listdir(OUTPUT_IMAGES) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    return len(image_files)

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("COPY FILTERED COCO ACTION IMAGES")
    print("=" * 80)
    
    # Check if already copied
    existing_count = verify_copied_images()
    if existing_count > 0:
        print(f"\nFound {existing_count} existing images in {OUTPUT_IMAGES}")
        response = input("Do you want to re-copy? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping copy. Exiting.")
            return
    
    # Step 1: Find image directories
    image_dirs = find_all_image_directories()
    
    if not image_dirs:
        print("\n✗ No image directories found!")
        print("\nPlease check your COCO dataset structure:")
        print("  Expected: dataset/coco_complete/train2014/")
        print("  Or: dataset/coco_complete/images/")
        return
    
    # Step 2: Build image index
    image_index = build_image_index(image_dirs)
    
    if not image_index:
        print("\n✗ No images found in directories!")
        return
    
    # Step 3: Load filtered captions
    try:
        images_metadata = load_filtered_captions()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return
    
    # Step 4: Copy images
    copied, missing = copy_images(images_metadata, image_index)
    
    # Step 5: Verify results
    final_count = verify_copied_images()
    
    print("\n" + "=" * 80)
    print("COPY COMPLETE!")
    print("=" * 80)
    print(f"Destination: {OUTPUT_IMAGES}")
    print(f"Total copied: {final_count} images")
    print(f"Success rate: {final_count/len(images_metadata)*100:.1f}%")
    
    if final_count > 0:
        print("\n Images ready for training!")
    else:
        print("\nNo images were copied. Please check your COCO dataset structure.")
        print("   Try running: python verify_dataset.py")

if __name__ == "__main__":
    main()