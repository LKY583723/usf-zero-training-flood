#!/usr/bin/env python3
"""Reference implementation of zero-training configuration F.

Companion code for the letter
    "Zero-Training SAR Flood Mapping from a Single SLC Pair: Gate Design and Its Limits"
    (IEEE Geoscience and Remote Sensing Letters, submitted).

Configuration F is fully deterministic and training-free: it has no learned
parameters, needs no GPU, and consumes a **single** Sentinel-1 SLC pre/post pair.

    S_int = union_q { p : I_post^q(p) < tau_s^q }            (per-scene Otsu, dark water)
    S_chg = union_q { p : |I_post^q(p) - I_pre^q(p)| > tau'_s^q }   (per-scene Otsu, change)
    F     = CC_{>=k}( S_int n S_chg n { p : I_post^VV(p) < theta } )
    theta = -12 dB, k = 500 px   (the only two constants fixed across scenes)

Input: single-precision 8-band patch stacks (bands 0-1 pre-event coherence VH/VV,
2-3 co-event coherence VH/VV, 4-5 pre-event intensity dB VH/VV, 6-7 post-event
intensity dB VH/VV), as distributed by the UrbanSARFloods benchmark. The band
ordering was verified against flood ground truth; a reversed coherence ordering
silently disables the coherence strategy.

The dataset itself is third-party and is NOT redistributed with this code; obtain
it from its original source (Zhao et al., CVPRW 2024).

Example
-------
    python urbanflood_configf.py \
        --patches  /data/testing_case_256/20210727_Weihui \
        --gt       /data/20210727_Weihui/testing_case_orig/20210727_Weihui/20210727_Weihui_GT.tif \
        --out      ./masks/weihui

prints the pixel-aggregated open-flood F1 over the patch set and writes one
GeoTIFF mask per input patch.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import numpy as np
import rasterio

# ---------------------------------------------------------------- band layout
CO_PRE_VH, CO_PRE_VV = 0, 1     # pre-event coherence (VH, VV)
CO_CO_VH, CO_CO_VV = 2, 3       # co-event (flood-time) coherence (VH, VV)
INT_PRE_VH, INT_PRE_VV = 4, 5   # pre-event intensity, dB (VH, VV)
INT_POST_VH, INT_POST_VV = 6, 7  # post-event intensity, dB (VH, VV)

DB_LO, DB_HI = -30.0, 0.0       # dB stretch used before Otsu
POST_THR = -12.0                # theta: fixed dark-water bound, dB
CONN_MIN = 500                  # k: fixed minimum connected-component size, px


def _finite(img: np.ndarray) -> np.ndarray:
    """Replace non-finite samples by the median so Otsu never sees NaN."""
    return np.nan_to_num(img, nan=np.nanmedian(img))


def _otsu_binary(values: np.ndarray, flip: bool = False) -> np.ndarray:
    """Per-scene Otsu threshold on a 1-99 percentile stretch.

    flip=False: value > threshold is positive (change strategies)
    flip=True : value < threshold is positive (dark-water strategy)
    Invalid pixels are never positive.

    Notes: the quantisation fills non-finite samples with the low percentile so the
    result does not depend on how a given NumPy build casts NaN to uint8. This form
    reproduces the masks of the research code exactly (checked patch by patch).
    """
    v = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(v)
    finite = v[valid]
    if len(finite) < 100:
        return np.zeros(v.shape, dtype=bool)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return np.zeros(v.shape, dtype=bool)
    q = np.clip((np.where(valid, v, lo) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    ret, _ = cv2.threshold(q, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pred = (q < ret) if flip else (q > ret)
    return pred & valid


def _amp(sar: np.ndarray, index: int) -> np.ndarray:
    """Intensity band as uint8 over the fixed dB stretch (for the Otsu call)."""
    return np.clip((np.nan_to_num(sar[index], nan=DB_LO) - DB_LO) / (DB_HI - DB_LO) * 255.0,
                   0, 255).astype(np.uint8)


# ------------------------------------------------------------------ strategies
def strategy_int(sar: np.ndarray, pol: str = "vv") -> np.ndarray:
    """S_int: post-event dark water (per polarization, per scene)."""
    post = sar[INT_POST_VV] if pol == "vv" else sar[INT_POST_VH]
    return _otsu_binary(post, flip=True)


def strategy_chg(sar: np.ndarray, pol: str = "vv") -> np.ndarray:
    """S_chg: |post - pre| intensity change (per polarization, per scene)."""
    post = sar[INT_POST_VV] if pol == "vv" else sar[INT_POST_VH]
    pre = sar[INT_PRE_VV] if pol == "vv" else sar[INT_PRE_VH]
    return _otsu_binary(np.abs(_finite(post) - _finite(pre)), flip=False)


def conn_filter(mask: np.ndarray, min_size: int = CONN_MIN) -> np.ndarray:
    """Drop connected components smaller than min_size pixels (speckle)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[lab == i] = True
    return out


def config_f(sar: np.ndarray, post_thr: float = POST_THR,
             conn_min: int = CONN_MIN) -> np.ndarray:
    """Configuration F: union of the two adaptive strategies, then the fixed gate."""
    s_int = strategy_int(sar, "vv") | strategy_int(sar, "vh")
    s_chg = strategy_chg(sar, "vv") | strategy_chg(sar, "vh")
    dark = np.nan_to_num(sar[INT_POST_VV]) < post_thr
    return conn_filter(s_int & s_chg & dark, conn_min)


def intermediate_masks(sar: np.ndarray, post_thr: float = POST_THR,
                       conn_min: int = CONN_MIN) -> dict:
    """The four layers of the letter's ablation: A -> intersection -> +dark -> F."""
    s_int = strategy_int(sar, "vv") | strategy_int(sar, "vh")
    s_chg = strategy_chg(sar, "vv") | strategy_chg(sar, "vh")
    cand = s_int & s_chg
    dark = cand & (np.nan_to_num(sar[INT_POST_VV]) < post_thr)
    return {"A_S_int": s_int, "C_intersection": cand, "post_dark": dark,
            "F": conn_filter(dark, conn_min)}


# ------------------------------------------------------------------ evaluation
def f1_score(pred: np.ndarray, gt: np.ndarray, gt_class: int = 1) -> dict:
    """Pixel-aggregated F1 against one ground-truth class (open = 1, urban = 2)."""
    pos = gt == gt_class
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    denom = 2 * tp + fp + fn
    f1 = (2.0 * tp / denom) if denom else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


# ------------------------------------------------------------------------ CLI
def _patch_rc(path: str):
    m = re.search(r"ID_(\d+)_(\d+)_SAR", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-training configuration-F flood mapping.")
    ap.add_argument("--patches", required=True, help="directory of *_SAR.tif 8-band patches")
    ap.add_argument("--out", default=None, help="directory for GeoTIFF masks (optional)")
    ap.add_argument("--gt", default=None, help="scene ground-truth *_GT.tif (optional)")
    ap.add_argument("--gt-class", type=int, default=1, help="positive class in GT (1=open, 2=urban)")
    ap.add_argument("--post-thr", type=float, default=POST_THR, help="dark-water bound theta, dB")
    ap.add_argument("--conn-min", type=int, default=CONN_MIN, help="minimum component size k, px")
    ap.add_argument("--min-flood-px", type=int, default=100,
                    help="skip patches with fewer GT flood pixels (evaluation protocol)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.patches, "*_SAR.tif")))
    if not files:
        raise SystemExit(f"no *_SAR.tif patches found in {args.patches}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    gtf = None
    if args.gt:
        with rasterio.open(args.gt) as ds:
            gtf = ds.read(1)

    tot = {"tp": 0, "fp": 0, "fn": 0}
    used = 0
    for path in files:
        with rasterio.open(path) as ds:
            sar = ds.read().astype(np.float32)
            profile = ds.profile
        mask = config_f(sar, post_thr=args.post_thr, conn_min=args.conn_min)

        if args.out:
            profile.update(count=1, dtype="uint8", compress="lzw")
            with rasterio.open(os.path.join(args.out, os.path.basename(path)), "w",
                               **profile) as dst:
                dst.write(mask.astype(np.uint8), 1)

        if gtf is not None:
            r, c = _patch_rc(path)
            if r is None:
                continue
            gt = gtf[r * sar.shape[1]:(r + 1) * sar.shape[1],
                      c * sar.shape[2]:(c + 1) * sar.shape[2]]
            # protocol: patches with fewer than min_flood_px flood pixels (any class) are skipped
            if int(np.sum(gt > 0)) < args.min_flood_px:
                continue
            m = f1_score(mask, gt, args.gt_class)
            tot["tp"] += m["tp"]; tot["fp"] += m["fp"]; tot["fn"] += m["fn"]
            used += 1

    print(f"patches processed : {len(files)}")
    if used:
        tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
        print(f"scored patches    : {used}  (>= {args.min_flood_px} GT pixels, class {args.gt_class})")
        print(f"pixel-aggregated F1 = {2 * tp / (2 * tp + fp + fn):.4f}"
              f"   (TP {tp}, FP {fp}, FN {fn})")


if __name__ == "__main__":
    main()
