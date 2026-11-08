"""下载测试事件灾前 GRD 时间序列（ASF，URS 认证）。

用法：python _dl_timeseries.py Jubba_1 2023-12-01 2022-12-01 2023-11-30
      python _dl_timeseries.py Weihui 2021-07-27 2020-08-01 2021-07-20
"""
import os
import sys
import asf_search as asf

BOUNDS = {
    '20231201_Jubba_1': 'POLYGON((40.583 -1.257, 43.174 -1.257, 43.174 0.922, 40.583 0.922, 40.583 -1.257))',
    '20210727_Weihui': 'POLYGON((113.03 34.66, 116.19 34.66, 116.19 36.70, 113.03 36.70, 113.03 34.66))',
    '20230609_NovaKakhovka': 'POLYGON((32.0 46.5, 34.5 46.5, 34.5 48.5, 32.0 48.5, 32.0 46.5))',
}

ev = sys.argv[1]
ref_date = sys.argv[2]
start = sys.argv[3]
end = sys.argv[4]
wkt = BOUNDS.get(ev)
if not wkt:
    print(f'未知事件 {ev}')
    sys.exit(1)

print(f'搜索 {ev} 灾前 GRD [{start} ~ {end}] ...', flush=True)
results = asf.search(
    platform=asf.PLATFORM.SENTINEL1,
    processingLevel='GRD_HD',
    start=start, end=end,
    intersectsWith=wkt,
    beamMode='IW',
    maxResults=100,
)
# 按日期分组，选均匀分布
by_date = {}
for r in results:
    d = r.properties['startTime'][:10]
    by_date.setdefault(d, r)
dates = sorted(by_date.keys())
print(f'找到 {len(dates)} 个日期', flush=True)

# 均匀选 12 景
n = min(12, len(dates))
if n < 6:
    print('日期太少，无法做时间序列'); sys.exit(1)
idx = [round(i * (len(dates) - 1) / (n - 1)) for i in range(n)]
idx = sorted(set(idx))
selected = [dates[i] for i in idx]
print(f'选 {len(selected)} 景: {selected}', flush=True)

sel_results = [by_date[d] for d in selected]
out = f'D:/DisasterM3/UrbanSARFloods/timeseries/{ev}'
os.makedirs(out, exist_ok=True)

session = asf.ASFSession()
import netrc
n = netrc.netrc()
auth = n.authenticators('urs.earthdata.nasa.gov')
if auth:
    session.auth_with_creds(auth[0], auth[2])
else:
    session.auth_with_creds(input('URS user: '), input('URS pass: '))

ok = 0
for r in sel_results:
    d = r.properties['startTime'][:10]
    fname = f'{ev}_{d}.zip'
    fpath = os.path.join(out, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 100e6:
        print(f'跳过已下载 {d} ({os.path.getsize(fpath)/1e6:.0f}MB)', flush=True)
        ok += 1
        continue
    success = False
    for attempt in range(3):
        try:
            print(f'下载 {d} (try {attempt+1}/3) ...', flush=True)
            r.download(path=out, session=session, filename=fname)
            ok += 1
            success = True
            break
        except Exception as e:
            print(f'  [fail] {d}: {str(e)[:100]}', flush=True)
    if not success:
        print(f'  [SKIP] {d} 下载失败', flush=True)
print(f'DONE ({ok}/{len(sel_results)} 成功)', flush=True)
