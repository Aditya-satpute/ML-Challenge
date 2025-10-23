"""
Download images for a given split using the hackathon-provided utils.
Usage:
  python -m src.download_images --data_csv ./student_resource/dataset/train.csv --out_dir ./images/train
"""
import argparse, os, pandas as pd
from pathlib import Path
from student_resource.src.utils import download_images  # uses the starter's util

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(df)} images to {args.out_dir} ...")
    download_images(df["image_link"].tolist(), args.out_dir)
    print("Done.")
