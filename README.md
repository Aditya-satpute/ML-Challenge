# Smart Product Pricing : End‑to‑End Solution (Text‑Only Baseline + Optional Images)

This folder contains a **ready‑to‑run** baseline you can submit to the hackathon.
It trains on `catalog_content` using TF‑IDF + handcrafted numeric features, predicts
`price` for the test set, and writes `outputs/test_out.csv` in the required format.

> **Metric:** SMAPE (0–200%). Lower is better.  
> **Rules:** No external price lookup. Images optional.

## Quickstart
```bash
# 1) Create and activate a fresh virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install packages
pip install -r requirements.txt

# 3) Place the hackathon dataset at: ./student_resource/dataset
#    (This is already present in your zip. Paths below assume that layout.)
ls student_resource/dataset
# -> train.csv  test.csv  sample_test.csv  sample_test_out.csv

# 4) Train + Predict (text‑only)
python -m src.train_textonly --data_dir ./student_resource/dataset --out_dir ./outputs --folds 5 --alpha 3.0 --max_features 200000

# 5) Your submission file:
ls outputs/test_out.csv
```

## What’s inside
- `src/train_textonly.py` – main script: TF‑IDF(1–2‑gram, min_df=5, max_features=200k) + numeric regex features → Ridge.
- `src/utils.py` – SMAPE metric + feature extraction.
- `src/image_embeddings_optional.py` – optional script to precompute image features (if you want to extend the model).
- `requirements.txt`

## Tips to Improve Score
- Tune Ridge `alpha` & TF‑IDF `max_features`.
- Add character n‑grams in TF‑IDF.
- Stack 2‑3 linear models (Ridge, Lasso, ElasticNet) on TF‑IDF and average predictions.
- SVD‑compress TF‑IDF then use tree models (e.g., XGBoost/LightGBM) on dense 256–1024D embeddings.
- Add **image embeddings** (EfficientNet‑B0 headless) and concatenate with text features.

## Reproducibility
- Uses fixed `random_state=42`.
- Saves `outputs/artifacts.joblib` and `outputs/cv_summary.json`.

## Submission
Upload `outputs/test_out.csv` (two columns: `sample_id,price`). Ensure the row count equals `test.csv`.
Also submit a 1‑page methodology doc; see `Documentation.md` we generated separately.
