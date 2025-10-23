# src/train_textonly.py
"""
Memory-friendly training for Smart Product Pricing (8 GB Macs OK):

- Word TF-IDF (1–2g) + Char TF-IDF (char_wb 3–5g)  → cast to float32
- Numeric regex features (quantity, pack, per-unit, flags)
- Optional Brand target encoding (KFold, leak-safe) --use_brand_te 1
- Ridge on log1p(price) with winsorization and CV SMAPE
- Optional: SVD + HistGradientBoosting (nonlinear booster) and blend
- Optional: HashingVectorizer mode for ultra-low RAM
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor

from src.utils import smape, build_numeric_features, extract_brand

RANDOM_STATE = 42

# ---------------------------
# IO
# ---------------------------
def _read_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test

# ---------------------------
# Text features
# ---------------------------
def _build_text_blocks(train, test, args):
    text_tr = train["catalog_content"].fillna("")
    text_te = test["catalog_content"].fillna("")

    if args.use_hashing:
        vec_word = HashingVectorizer(
            lowercase=True, ngram_range=(1, 2),
            n_features=args.word_n_features,
            alternate_sign=False, norm='l2'
        )
        vec_char = HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            n_features=args.char_n_features,
            alternate_sign=False, norm='l2'
        )
        Xw_tr = vec_word.transform(text_tr)
        Xw_te = vec_word.transform(text_te)
        Xc_tr = vec_char.transform(text_tr)
        Xc_te = vec_char.transform(text_te)
    else:
        vec_word = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df_word,
            max_features=args.word_max_features,
            sublinear_tf=True,
        )
        vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=args.min_df_char,
            max_features=args.char_max_features,
            sublinear_tf=True,
        )
        Xw_tr = vec_word.fit_transform(text_tr)
        Xw_te = vec_word.transform(text_te)
        Xc_tr = vec_char.fit_transform(text_tr)
        Xc_te = vec_char.transform(text_te)

    # cast to float32 to save memory
    Xw_tr = Xw_tr.astype(np.float32); Xw_te = Xw_te.astype(np.float32)
    Xc_tr = Xc_tr.astype(np.float32); Xc_te = Xc_te.astype(np.float32)
    return vec_word, vec_char, Xw_tr, Xw_te, Xc_tr, Xc_te

# ---------------------------
# Numeric features
# ---------------------------
def _build_numeric_blocks(train, test):
    num_tr = build_numeric_features(train["catalog_content"])
    num_te = build_numeric_features(test["catalog_content"])
    scaler = StandardScaler(with_mean=False)
    Xn_tr = csr_matrix(scaler.fit_transform(num_tr.fillna(0.0).values)).astype(np.float32)
    Xn_te = csr_matrix(scaler.transform(num_te.fillna(0.0).values)).astype(np.float32)
    return scaler, Xn_tr, Xn_te

# ---------------------------
# Brand target encoding (optional)
# ---------------------------
def _brand_target_encoding(train, test, folds):
    brand_tr = train["catalog_content"].fillna("").apply(extract_brand)
    brand_te = test["catalog_content"].fillna("").apply(extract_brand)

    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof_brand = np.zeros(len(train), dtype=np.float32)

    for tr_idx, va_idx in kf.split(train):
        means = train.iloc[tr_idx].groupby(brand_tr.iloc[tr_idx])["price"].mean()
        gmean = float(train.iloc[tr_idx]["price"].mean())
        oof_brand[va_idx] = brand_tr.iloc[va_idx].map(means).fillna(gmean).astype(np.float32).values

    brand_means_full = train.groupby(brand_tr)["price"].mean()
    brand_te_enc = brand_te.map(brand_means_full).fillna(train["price"].mean()).astype(np.float32).values

    return oof_brand.reshape(-1,1), brand_te_enc.reshape(-1,1)

# ---------------------------
# Winsorize
# ---------------------------
def _winsorize(y, pct):
    if pct <= 0:
        return y
    lo = np.percentile(y, pct)
    hi = np.percentile(y, 100 - pct)
    return np.clip(y, lo, hi)

# ---------------------------
# CV training for Ridge
# ---------------------------
def _cv_ridge(X, y, folds, alpha, log_target=True):
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(X.shape[0], dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE, max_iter=10000)
        if log_target:
            yl = np.log1p(y)
            model.fit(X[tr_idx], yl[tr_idx])
            pred = np.expm1(model.predict(X[va_idx]))
        else:
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
        pred = np.clip(pred, 0.0, None)
        oof[va_idx] = pred.astype(np.float32)
        print(f"[Fold {fold}] SMAPE = {smape(y[va_idx], pred):.4f}%")
    return oof, smape(y, oof)

# ---------------------------
# Optional SVD + HGB booster
# ---------------------------
def _svd_hgb_booster(Xtr, Xte, y, folds, svd_dim):
    svd = TruncatedSVD(n_components=svd_dim, random_state=RANDOM_STATE)
    Ztr = svd.fit_transform(Xtr.astype(np.float32))
    Zte = svd.transform(Xte.astype(np.float32))

    y_log = np.log1p(y)
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof_tree = np.zeros(len(y), dtype=np.float32)
    for tr_idx, va_idx in kf.split(Ztr, y_log):
        hgb = HistGradientBoostingRegressor(
            learning_rate=0.05, max_depth=6, max_bins=255,
            min_samples_leaf=20, random_state=RANDOM_STATE
        )
        hgb.fit(Ztr[tr_idx], y_log[tr_idx])
        p = np.expm1(hgb.predict(Ztr[va_idx]))
        oof_tree[va_idx] = np.clip(p, 0, None).astype(np.float32)

    return svd, Ztr, Zte, oof_tree

# ---------------------------
# Main
# ---------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # --------- Read ---------
    train, test = _read_data(args.data_dir)
    y = train["price"].astype(float).values

    # --------- Features ---------
    vec_word, vec_char, Xw_tr, Xw_te, Xc_tr, Xc_te = _build_text_blocks(train, test, args)
    scaler_num, Xn_tr, Xn_te = _build_numeric_blocks(train, test)

    # Optional brand TE
    oof_brand = te_brand = None
    if args.use_brand_te:
        oof_brand, te_brand = _brand_target_encoding(train, test, args.folds)

    # Stack sparse blocks
    blocks_tr = [Xw_tr, Xc_tr, Xn_tr]
    blocks_te = [Xw_te, Xc_te, Xn_te]
    if args.use_brand_te:
        blocks_tr.append(csr_matrix(oof_brand))
        blocks_te.append(csr_matrix(te_brand))

    Xtr = hstack(blocks_tr, format="csr").astype(np.float32)
    Xte = hstack(blocks_te, format="csr").astype(np.float32)

    # --------- Label prep ---------
    y_train = _winsorize(y, args.winsorize)

    # --------- Ridge CV ---------
    oof_ridge, cv_smape_ridge = _cv_ridge(Xtr, y_train, args.folds, args.alpha, log_target=True)
    print(f"[CV] Ridge SMAPE = {cv_smape_ridge:.4f}%")

    # --------- Final Ridge fit ---------
    final_ridge = Ridge(alpha=args.alpha, random_state=RANDOM_STATE, max_iter=10000)
    final_ridge.fit(Xtr, np.log1p(y_train))
    pred_ridge = np.expm1(final_ridge.predict(Xte))
    pred_ridge = np.clip(pred_ridge, 0, None).astype(np.float32)

    # --------- Optional SVD + HGB booster ---------
    blend_used = False
    oof_tree = None
    if args.use_tree and args.svd_dim > 0:
        svd, Ztr, Zte, oof_tree = _svd_hgb_booster(Xtr, Xte, y_train, args.folds, args.svd_dim)
        cv_smape_tree = smape(y, oof_tree)
        print(f"[CV] HGB(SVD={args.svd_dim}) SMAPE = {cv_smape_tree:.4f}%")
        # fit final tree on all
        hgb = HistGradientBoostingRegressor(
            learning_rate=0.05, max_depth=6, max_bins=255,
            min_samples_leaf=20, random_state=RANDOM_STATE
        )
        hgb.fit(Ztr, np.log1p(y_train))
        pred_tree = np.expm1(hgb.predict(Zte))
        pred_tree = np.clip(pred_tree, 0, None).astype(np.float32)
        # blend
        w = float(args.blend_weight)
        test_pred = (w * pred_ridge + (1.0 - w) * pred_tree).astype(np.float32)
        oof_blend = (w * oof_ridge + (1.0 - w) * oof_tree).astype(np.float32)
        print(f"[CV] BLENDED SMAPE = {smape(y, oof_blend):.4f}%")
        blend_used = True
    else:
        test_pred = pred_ridge

    # --------- Write submission ---------
    sub = pd.DataFrame({"sample_id": test["sample_id"], "price": test_pred})
    sub_path = os.path.join(args.out_dir, "test_out.csv")
    sub.to_csv(sub_path, index=False)
    print("Wrote submission:", sub_path)

    # --------- Save artifacts ---------
    import joblib
    art = {
        "vectorizer_word": vec_word,
        "vectorizer_char": vec_char,
        "scaler_num": scaler_num,
        "model_ridge": final_ridge,
        "config": vars(args),
        "cv_smape_ridge": float(cv_smape_ridge),
    }
    if blend_used:
        art.update({"svd": svd})
    joblib.dump(art, os.path.join(args.out_dir, "artifacts.joblib"))

    summary = {"cv_smape_ridge": float(cv_smape_ridge)}
    if blend_used:
        summary["cv_smape_blend"] = float(smape(y, (args.blend_weight * oof_ridge + (1-args.blend_weight) * oof_tree)))
    with open(os.path.join(args.out_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --------- OOF dump ---------
    if args.write_oof:
        pd.DataFrame({"sample_id": train["sample_id"], "price": y, "oof_pred": oof_ridge}).to_csv(
            os.path.join(args.out_dir, "oof_ridge.csv"), index=False
        )
        if blend_used:
            pd.DataFrame({"sample_id": train["sample_id"], "price": y, "oof_pred": oof_tree}).to_csv(
                os.path.join(args.out_dir, "oof_tree.csv"), index=False
            )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./outputs")

    # CV / model
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--winsorize", type=float, default=1.0)
    ap.add_argument("--write_oof", type=int, default=1)

    # TF-IDF settings
    ap.add_argument("--word_max_features", type=int, default=300_000)
    ap.add_argument("--char_max_features", type=int, default=300_000)
    ap.add_argument("--min_df_word", type=int, default=3)
    ap.add_argument("--min_df_char", type=int, default=10)

    # Hashing mode (ultra low RAM)
    ap.add_argument("--use_hashing", type=int, default=0)
    ap.add_argument("--word_n_features", type=int, default=2**19)
    ap.add_argument("--char_n_features", type=int, default=2**19)

    # Optional nonlinear booster
    ap.add_argument("--use_tree", type=int, default=0)
    ap.add_argument("--svd_dim", type=int, default=0)
    ap.add_argument("--blend_weight", type=float, default=0.6)

    # NEW: toggle Brand Target Encoding
    ap.add_argument("--use_brand_te", type=int, default=0)

    args = ap.parse_args()
    main(args)
