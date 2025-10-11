import pandas as pd
import os
from utils import download_images # Assuming utils.py is in the same folder
from tqdm import tqdm # A library for progress bars

# --- Configuration ---
DATA_DIR = '../data/'
IMAGE_DIR = '../images/'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

def run_downloader():
    # Ensure the destination directory exists
    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"Images will be saved in: {os.path.abspath(IMAGE_DIR)}")

    # Load dataframes
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Combine them to download all images at once
    all_samples_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Found {len(all_samples_df)} total image links to process.")

    # Convert dataframe to a list of dictionaries for easier processing
    records = all_samples_df[['sample_id', 'image_link']].to_dict('records')

    # Use the provided download function
    # The function expects a list of dicts with keys 'id' and 'url'
    # So we need to rename our keys
    download_list = [{'id': r['sample_id'], 'url': r['image_link']} for r in records]
    
    print("Starting image download...")
    download_images(download_list, IMAGE_DIR)
    print("Image download process complete.")

if __name__ == '__main__':
    run_downloader()