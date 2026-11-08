"""门诊断：相干证据(S_coh, Δγ=pre_coh−co_coh)对 config F 错分像素有无区分力。

对每个事件：现 config F 掩膜(复用 flood_strategies, conn=500)。
分组（开放口径 GT==1 正类）：
  TP_open  pred&g1    FN_open ~pred&g1    FP_open  pred&~g1
  UrbanGT  g==2 像素（无论 pred）
统计各组的 Δγ(VV) 分布：中位/均值、>0.15/0.2/0.3 占比；另给 VH 与 max(VV,VH) 的 >0.2 占比。
若 FN(尤其被 −12 误杀的) 的 Δγ 高而 FP 的 Δγ 低 → 相干进门(G-A)有救；若 FP 也高 → 加相干会引 FP。
"""
import glob, os, re, sys
import numpy as np, rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flood_strategies import all_strategies, conn_filter

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = os.path.join(DATA, 'testing_case_256')
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka', '20231201_Jubba_1', '20231201_Jubba_2']
POST_THR, CONN_MIN = -12.0, 500
CAP = 400_000  # 每组最多存像素数(随机降采样)防内存


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def config_f(sar):
    s = all_strategies(sar)
    f = s['int'] & s['chg']
    f = f & (np.nan_to_num(sar[7]) < POST_THR)
    return conn_filter(f, CONN_MIN)


def push(store, vals):
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return
    if len(store) + v.size <= CAP:
        store.append(v)
    else:
        need = CAP - len(store)
        if need > 0:
            store.append(v[:need])  # 近似：只保前 need（确定性降采样）
        # 超出部分丢弃（各 patch 顺序固定，偏差可接受，仅用于趋势判断）


def stats(store):
    if not store:
        return None
    a = np.concatenate(store)
    return dict(n=a.size, med=float(np.median(a)), mean=float(a.mean()),
                f15=float((a > 0.15).mean()), f20=float((a > 0.2).mean()),
                f30=float((a > 0.3).mean()))


for sc in SCENES:
    files = sorted(glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')))
    gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)
    grp = {k: [] for k in ['TP', 'FN', 'FP', 'U_GT']}
    tp = fp = fn = 0
    for f in files:
        r, c = parse_id(f)
        y0, x0 = r * 256, c * 256
        if y0 + 256 > gt_full.shape[0] or x0 + 256 > gt_full.shape[1]:
            continue
        g = gt_full[y0:y0 + 256, x0:x0 + 256]
        if (g > 0).sum() < 100:
            continue
        with rasterio.open(f) as ds:
            sar = ds.read().astype(np.float32)
        pred = config_f(sar)
        dvv = np.nan_to_num(sar[1]) - np.nan_to_num(sar[3])  # Δγ VV
        dvh = np.nan_to_num(sar[0]) - np.nan_to_num(sar[2])  # Δγ VH
        g1 = g == 1
        tp += (pred & g1).sum(); fp += (pred & ~g1).sum(); fn += (~pred & g1).sum()
        push(grp['TP'], dvv[pred & g1])
        push(grp['FN'], dvv[~pred & g1])
        push(grp['FP'], dvv[pred & ~g1])
        push(grp['U_GT'], dvv[g == 2])
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
    print(f'\n=== {sc} ===  open P={p_:.3f} R={r_:.3f} F1={f1:.3f}  (TP{tp} FP{fp} FN{fn})')
    print(f'{"组":6s}{"n":>10s}{"Δγ中位":>9s}{"Δγ均值":>9s}{">0.15":>8s}{">0.2":>8s}{">0.3":>8s}')
    for k, lab in [('TP', 'TP(命中)'), ('FN', 'FN(漏)'), ('FP', 'FP(假阳)'), ('U_GT', '城区GT')]:
        s = stats(grp[k])
        if s is None:
            print(f'{lab:6s}   —(无像素)'); continue
        print(f'{lab:6s}{s["n"]:10d}{s["med"]:9.3f}{s["mean"]:9.3f}'
              f'{s["f15"]:8.2f}{s["f20"]:8.2f}{s["f30"]:8.2f}')
