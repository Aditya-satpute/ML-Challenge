"""
Optional: generate image embeddings and save as a .npz sparse matrix to be concatenated
with text features during training. Requires internet to download pretrained weights
unless you have them cached. Not used by default.
"""
import argparse, os, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
import torch, torchvision.transforms as T
from torchvision import models

def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = args.data_dir
    out_path = args.out_path

    df = pd.read_csv(os.path.join(data_dir, args.split + ".csv"))
    urls = df["image_link"].fillna("")
    # Expect images already downloaded to args.img_dir with filenames <index>.jpg
    # Use the sample utils in the hackathon starter to download.
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(device)
    model.classifier = torch.nn.Identity()
    model.eval()

    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    feats = []
    with torch.inference_mode():
        for i in tqdm(range(len(df))):
            from pathlib import Path
            fname = Path(urls.iloc[i]).name
            fp = os.path.join(args.img_dir, fname)
            if not os.path.exists(fp):
                feats.append(np.zeros(1280, dtype=np.float32)); continue
            im = Image.open(fp).convert("RGB")
            x = tfm(im).unsqueeze(0).to(device)
            v = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
            feats.append(v)
    X = np.stack(feats, axis=0)
    np.save(out_path, X)
    print("Saved:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--img_dir", type=str, required=True, help="directory containing downloaded images named as <rowindex>.jpg")
    ap.add_argument("--split", type=str, default="train", choices=["train","test"])
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()
    build(args)
