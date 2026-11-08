"""批量处理 Jubba_1 的 GRD → 对齐网格（跳过已处理）。"""
import glob
import os
import subprocess
import sys

zips = sorted(glob.glob('D:/DisasterM3/UrbanSARFloods/timeseries/20231201_Jubba_1/*.zip'))
OUT = 'D:/DisasterM3/UrbanSARFloods/timeseries/20231201_Jubba_1_aligned'
os.makedirs(OUT, exist_ok=True)

for z in zips:
    import re
    date = re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(z)).group(0)
    done = os.path.exists(os.path.join(OUT, f'{date}_VV.tif'))
    if done:
        print(f'跳过 {date}（已处理）', flush=True)
        continue
    print(f'处理 {date} ...', flush=True)
    r = subprocess.run([sys.executable, '_grd_to_grid.py', z, '20231201_Jubba_1'],
                       capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if r.stdout:
        print(r.stdout.strip()[-100:], flush=True)
    if r.returncode != 0:
        print(f'  ERR: {r.stderr[-150:]}', flush=True)
print('BATCH DONE', flush=True)
