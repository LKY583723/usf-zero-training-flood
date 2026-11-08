"""多线程下载 Jubba testing_case_256 patches（hf-mirror，绕过大文件 403）。
用法：python _dl_jubba_patches.py 20231201_Jubba_1|20231201_Jubba_2
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download

ev = sys.argv[1]
names = json.load(open(f'D:/DisasterM3/UrbanSARFloods/repro/{ev}_patch_names.json'))
OUT = f'D:/DisasterM3/UrbanSARFloods/testing_case_256/{ev}'
os.makedirs(OUT, exist_ok=True)
lock = threading.Lock()
done = [0]


def dl(fn):
    try:
        p = hf_hub_download(repo_id='S1Floodbenchmark/UrbanSARFloods_v1',
                            filename=f'testing_case_256/{ev}/{fn}',
                            local_dir='D:/DisasterM3/UrbanSARFloods', repo_type='dataset')
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f'  {done[0]}/{len(names)}', flush=True)
        return True
    except Exception as e:
        return False


t0 = time.time()
with ThreadPoolExecutor(max_workers=20) as ex:
    results = list(ex.map(dl, names))
ok = sum(results)
print(f'完成 {ok}/{len(names)}，用时 {round(time.time()-t0,1)}s', flush=True)
