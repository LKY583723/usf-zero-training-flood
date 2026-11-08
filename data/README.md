# Data

The **UrbanSARFloods** dataset is third-party and is not redistributed here.
Download it from its original source:

- J. Zhao, Z. Xiong, and X. X. Zhu, "UrbanSARFloods: Sentinel-1 SLC-based benchmark
  dataset for urban and open-area flood mapping," in *Proc. IEEE/CVF Conf. Comput. Vis.
  Pattern Recognit. Workshops (CVPRW)*, 2024, pp. 419-429.

## What the code expects

* `testing_case_256/<scene>/<scene>_ID_<row>_<col>_SAR.tif`
  -- 8-band float32 patch stacks, 256 x 256:
  bands 0-1 pre-event coherence (VH, VV); bands 2-3 co-event coherence (VH, VV);
  bands 4-5 pre-event intensity (dB, VH, VV); bands 6-7 post-event intensity (dB, VH, VV).
* `<scene>/testing_case_orig/<scene>/<scene>_GT.tif`
  -- full-scene ground truth (1 = open flood, 2 = urban flood).

Verified band ordering: bands 0-1 are the *pre*-event coherence (higher coherence than
bands 2-3 over flooded ground). A reversed ordering silently disables the coherence
strategy, so it is worth re-checking on any new data release.

Scenes covered by the letter: `20210727_Weihui`, `20230609_NovaKakhovka`,
`20231201_Jubba_1`, `20231201_Jubba_2` (three test events; Jubba_1 and Jubba_2 are two
scenes of one event).

Evaluation protocol: independent 256 x 256 patches, pixel-aggregated F1 per ground-truth
class, patches with fewer than 100 flood pixels excluded, Jubba aggregated over both
scenes by summing confusion matrices.
