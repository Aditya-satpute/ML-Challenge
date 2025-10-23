# ML-Challenge

# Smart Product Pricing : 1-Page Technical Note
**Team:** AlgoRisers  
**Members:** Aditya Satpute, Vansh Shrivastava, Kanika Shrivastava, Anshika Moundekar 
**Date:** 11 Oct 2025

## Executive Summary
We predict product prices using a lightweight **text-first** pipeline that combines TF-IDF features (word 1–2g + `char_wb` 3–5g) from `catalog_content` with **regex-derived quantity signals** (pack size, total grams/ml, per-unit measures). A **Ridge** regressor is trained on **log-prices** with light **winsorization** and 5-fold CV to align with the SMAPE metric; predictions are clipped ≥ 0. No external data or lookups are used.

---

## Methodology Used
### Problem Framing
- Supervised regression with **SMAPE** as the target evaluation metric (relative-error sensitive).
- Key drivers surfaced in EDA: **brand/category tokens** and **effective quantity** (unit size × pack count).
- Text often encodes units in mixed formats (`3x200g`, `12 fl oz`, `Pack of 6`) → normalize via regex.

### Model Architecture / Algorithms
- **Feature extractors**
  - **Word TF-IDF:** `ngram_range=(1,2)`, `min_df=3`, `max_features=300k`, `sublinear_tf=True`.
  - **Char TF-IDF (`char_wb`):** `ngram_range=(3,5)`, `min_df=10`, `max_features=300k`, `sublinear_tf=True`.
  - **Numeric regex block:**  
    - `pack_qty`, normalized `g_total` (grams) & `ml_total` (milliliters),  
    - derived: `total_units_g/ml`, `per_unit_g/ml`,  
    - text shape: `n_chars`, `n_words`, `n_digits`,  
    - flags: `is_weight`, `is_liquid`, `has_organic`, `has_family`, `has_sugarfree`.
  - Numeric features scaled with `StandardScaler(with_mean=False)`.
- **Fusion:** sparse `hstack([word_tfidf, char_tfidf, numeric])` (CSR).
- **Regressor:** `Ridge(alpha=3.0, max_iter=10000)` trained on **log1p(price)**.

### Training & Validation
- **Preprocessing:** strip `"Item Name:"`, normalize `× → x`, lowercase.
- **Target:** `y = log1p(price)`; labels **winsorized** at 1% tails to reduce outlier dominance.
- **Validation:** `KFold(n_splits=5, shuffle=True, random_state=42)`; report **OOF SMAPE**.
- **Post-processing:** `expm1` inverse transform; **clip ≥ 0** to meet constraints.
- **Reproducibility:** fixed seeds, deterministic scikit-learn; artifacts saved (`vectorizers`, `scaler`, `model`, `cv_summary.json`).

---

## Feature Engineering Techniques Applied
- **Unit/Pack parsing:** robust regex covering `kg/g/l/ml/lb/oz/fl-oz` and multiplicative forms (`k × qty unit`).
- **Quantity normalization:** convert to common bases (g, ml); compute **per-unit** and **total** measures.
- **Character n-grams:** capture SKUs, hyphenated strings, and unit patterns missed by word tokens.
- **Light text cleanup:** remove boilerplate; keep digits/units intact.

---

## Results (Validation)
- **5-Fold OOF SMAPE:** **51.2705%**  
  (per-fold: 51.8822, 51.1363, 51.3781, 50.5367, 51.4191)
- The largest gains came from **char n-grams** and **quantity normalization**; log-target further stabilized relative error.

---

## Submission Outputs
- Generated `outputs/test_out.csv` with **two columns exactly**: `sample_id,price` (same order/count as `test.csv`, all prices ≥ 0).
- No external data, scraping, or price lookups used.

---

## Notes & Future Improvements
- Add **brand target encoding** (KFold mean price by heuristic brand) and/or **segment models** (weight/liquid/other).
- Lightweight **SVD + tree booster** blended with Ridge, and optional **image embeddings** for incremental gains.

