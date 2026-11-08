# Zero-Training SAR Flood Mapping with an Explicit Verification Gate

Code, configuration and result archives for the manuscript

> **Zero-Training SAR Flood Mapping from a Single SLC Pair: Gate Design and Its Limits**
> *IEEE Geoscience and Remote Sensing Letters* (submitted). <!-- DOI to be added on acceptance -->

Reported configuration **F** is **zero-training and fully deterministic**: no learned
parameters, no GPU, and a **single** pre/post Sentinel-1 SLC pair. Reproducing the main
result therefore needs **no model checkpoint** -- only the test patches. Checkpoints are
required solely to reproduce the supervised CNN baselines.

---

## 1. Quick start (configuration F, no training)

```bash
pip install -r requirements.txt

python code/urbanflood_configf.py \
    --patches  /data/testing_case_256/20210727_Weihui \
    --gt       /data/20210727_Weihui/testing_case_orig/20210727_Weihui/20210727_Weihui_GT.tif \
    --out      ./masks/weihui
```

`code/urbanflood_configf.py` is a single-file, dependency-light implementation of the
published pipeline:

```
S_int = union_q { p : I_post^q(p) < tau_s^q }                  per-scene Otsu, dark water
S_chg = union_q { p : |I_post^q(p) - I_pre^q(p)| > tau'_s^q }   per-scene Otsu, change
F     = CC_{>=500}( S_int n S_chg n { p : I_post^VV(p) < -12 dB } )
```

`theta = -12 dB` and `k = 500 px` are the only constants fixed across scenes.
The script prints the pixel-aggregated open-flood F1 over the patch set, and can also
dump one GeoTIFF mask per patch (`--out`).

The implementation was checked patch by patch against the research code that produced
every number in the letter; the two agree exactly on the test patches (`mask equality`,
33 patches spanning all four test scenes). Running the command above on the full Weihui
test scene reproduces the letter's Table I entry exactly:

```
patches processed : 1000
scored patches    : 179  (>= 100 flood pixels)
pixel-aggregated F1 = 0.7728   (TP 250713, FP 44484, FN 102918)     # letter: 0.773
```

---

## 2. Layout

| Path | Contents |
|---|---|
| `code/urbanflood_configf.py` | Reference implementation of configuration F (this file is the entry point) |
| `code/research/` | The research scripts as run (evaluation, calibration, training, diagnostics, figure rendering) |
| `results/` | Result archives -- every number in the letter traces to a file here |
| `figures/` | Figure sources (matplotlib renderer) and the three vector PDFs of the letter |
| `data/README.md` | How to obtain the third-party UrbanSARFloods dataset |

Key archives:

| File | What it documents |
|---|---|
| `results/flood_ge_3event.md` | Three-event protocol: config F vs the reproduced CNNs |
| `results/flood_ge_configABC_unified.md` | A -> C -> F ablation |
| `results/flood_ge_fp_taxonomy.md` | Layer-by-layer precision/recall and false-positive removal |
| `results/flood_ge_gate_independence.md` | Marginal-survival / Jaccard analysis of the gate |
| `results/flood_ge_sensitivity.md` | POST_THR and CONN_MIN sweeps |
| `results/flood_ge_threshold_calibration.md` | Calibration on the official train/validation split |
| `results/flood_ge_urban_branch_*.md` | Zero-training urban branch: 16-event calibration and held-out evaluation |
| `results/flood_ge_split_baseline.md` | Split-based (block-wise Otsu) baseline vs the fixed gate: block sizes 16-128 px, with and without the gate |

---

## 3. Reproducing the figures

```bash
python figures/_render_figures_housestyle.py    # needs the dataset + checkpoints on PYTHONPATH
```

Figure 1 (letter overview), Figure 2 (decision chain and layer-by-layer decomposition) and
Figure 3 (example masks) are rendered to vector PDF at the exact column widths used in the
letter. The renderer carries a built-in layout self-check (text overflow, box overlap).

---

## 4. Supervised CNN baselines

`code/research/train.py` and `code/research/test.py` reproduce Unet, Unet++ and DeepLabV3+
(ResNet-34 encoders, 8 input channels, 3 classes) with the protocol of the letter:
100 epochs, batch 12, Adam lr 1e-5 / wd 1e-4, weighted cross-entropy (1:64:3200),
z-score normalization, no ImageNet initialization, seeds 42 / 0 / 123. Checkpoints are
not redistributed (~760 MB); they can be regenerated with the training script.

---

## 5. What is *not* included, deliberately

| Path | Why |
|---|---|
| `urban_sar_floods/` | Third-party dataset (Zhao et al., CVPRW 2024). **Not redistributed** -- see `data/README.md`. |
| test patches and full scenes | Derived from the above. |
| time-series stacks used for the BCP comparison | Large, and derived from third-party products. |
| CNN checkpoints | ~760 MB; regenerable from `code/research/train.py`. |

> The dataset is **not** covered by this repository's licence and must not be
> redistributed. Obtain it from its original source and cite the dataset paper.

---

## 6. Licence and citation

Released under the MIT licence (`LICENSE`); the third-party dataset keeps its own terms.
Citation metadata is in `CITATION.cff`. If you use configuration F, please cite the letter
above and the UrbanSARFloods dataset paper.
