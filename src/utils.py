# src/utils.py
import re
import numpy as np
import pandas as pd

# ---------------------------
# Metric
# ---------------------------
def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error in [0, 200].
    SMAPE = 100 * mean( |y - yhat| / ((|y|+|yhat|)/2) ), with 0 when denom=0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    out = np.zeros_like(y_true, dtype=float)
    out[mask] = np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return 100.0 * np.mean(out)

# ---------------------------
# Numeric feature extraction
# ---------------------------

# unit patterns normalized to grams (g) or milliliters (ml)
_UNIT_PATS = [
    (r'(\d+(?:\.\d+)?)\s*kg\b',                                 1000.0, 'g'),
    (r'(\d+(?:\.\d+)?)\s*g(?:ram|rams)?\b',                        1.0, 'g'),
    (r'(\d+(?:\.\d+)?)\s*(?:lb|pounds?)\b',                      453.592, 'g'),
    (r'(\d+(?:\.\d+)?)\s*(?:oz|ounces?)\b',                       28.3495, 'g'),
    (r'(\d+(?:\.\d+)?)\s*l(?:iter|itre|iters|itres)?\b',        1000.0, 'ml'),
    (r'(\d+(?:\.\d+)?)\s*ml\b',                                    1.0, 'ml'),
    (r'(\d+(?:\.\d+)?)\s*fl\s*oz\b',                              29.5735, 'ml'),
]

_PACK_PATS = [
    r'pack\s*of\s*(\d+)',
    r'(\d+)\s*pack\b',
    r'(\d+)\s*ct\b',
    r'(\d+)\s*count\b',
    r'(\d+)\s*pk\b',
    r'(\d+)[\s-]*pcs\b'
]

def _clean_text(s: str) -> str:
    # remove noisy prefix seen in the dataset
    return (s or "").replace("Item Name:", "").strip()

def _pack_qty(s: str) -> int:
    s = s.lower().replace("×", "x")
    for pat in _PACK_PATS:
        m = re.search(pat, s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    # e.g., "3x200g" or "2*500 ml"
    m = re.search(r'(\d+)\s*[x*]\s*\d+', s)
    return int(m.group(1)) if m else 1  # default 1 (single item)

def _sum_units(s: str):
    s = s.lower()
    g_total = ml_total = 0.0
    for pat, mul, unit in _UNIT_PATS:
        for m in re.finditer(pat, s):
            val = float(m.group(1)) * mul
            if unit == 'g':
                g_total += val
            else:
                ml_total += val
    # explicit k x qty unit (e.g., "3x200g", "2x500 ml")
    m = re.search(r'(\d+)\s*[x*]\s*(\d+(?:\.\d+)?)\s*(g|ml)\b', s)
    if m:
        k = int(m.group(1))
        per = float(m.group(2))
        u = m.group(3)
        if u == 'g':
            g_total += k * per
        else:
            ml_total += k * per
    return g_total, ml_total

def build_numeric_features(series: pd.Series) -> pd.DataFrame:
    """
    Return a dense DataFrame of lightweight numeric/regex features derived from catalog_content.
    """
    s = series.fillna("").map(_clean_text)

    packs = s.apply(_pack_qty)
    grams_mls = s.apply(_sum_units)
    grams = grams_mls.apply(lambda t: t[0])
    mls   = grams_mls.apply(lambda t: t[1])

    df = pd.DataFrame({
        "n_chars": s.str.len(),
        "n_words": s.str.split().apply(len),
        "n_digits": s.str.count(r'\d'),
        "pack_qty": packs,
        "g_total": grams,
        "ml_total": mls,
    })
    # derived totals
    df["total_units_g"] = df["g_total"] * df["pack_qty"]
    df["total_units_ml"] = df["ml_total"] * df["pack_qty"]
    # per-unit (robust to packs); regimes
    df["per_unit_g"] = df["g_total"] / df["pack_qty"].clip(lower=1)
    df["per_unit_ml"] = df["ml_total"] / df["pack_qty"].clip(lower=1)
    df["is_weight"] = (df["g_total"] > 0).astype(int)
    df["is_liquid"] = (df["ml_total"] > 0).astype(int)
    # flags commonly correlated with price
    df["has_organic"] = s.str.contains(r'\borganic\b', case=False, regex=True).astype(int)
    df["has_family"]  = s.str.contains(r'family|bulk', case=False, regex=True).astype(int)
    df["has_sugarfree"] = s.str.contains(r'sugar\s*free|zero\s*sugar', case=False, regex=True).astype(int)
    return df

# (Optional utility for brand-based features / target encoding)
def extract_brand(s: str) -> str:
    """
    Heuristic: take leftmost token sequence before punctuation/stopwords as brand.
    """
    s = _clean_text(s)
    stop = {'the', 'and', 'with', 'for', 'in', 'of', 'by'}
    left = re.split(r'[-,:|/]', s, maxsplit=1)[0]
    tokens = left.split()
    brand = []
    for w in tokens:
        w = re.sub(r'[^A-Za-z0-9]+', '', w)
        if not w or w.lower() in stop:
            break
        brand.append(w)
    return (brand[0] if brand else "unk").lower()
