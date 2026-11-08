"""GRD → Jubba_1 网格：解析 annotation GCP，warp 重采样到测试网格。

用法：python _grd_to_grid.py <zip路径> <事件>
输出：timeseries/{event}_aligned/{date}_VV.tif, {date}_VH.tif（dB, EPSG:4326 20m）
"""
import os
import re
import sys
import zipfile

import numpy as np

ev = sys.argv[2]
zpath = sys.argv[1]
zname = os.path.basename(zpath)
date = re.search(r'\d{4}-\d{2}-\d{2}', zname).group(0)

# Jubba_1 测试网格（EPSG:4326, 20m）
GT = 'D:/DisasterM3/UrbanSARFloods/20231201_Jubba_1/testing_case_orig/20231201_Jubba_1/20231201_Jubba_1_GT.tif'
OUT = f'D:/DisasterM3/UrbanSARFloods/timeseries/{ev}_aligned'
os.makedirs(OUT, exist_ok=True)

import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps

with rasterio.open(GT) as ds:
    gtbounds = ds.bounds
    res = ds.res[0]
print(f'目标网格: {gtbounds}, res={res}', flush=True)

from rasterio.warp import reproject, Resampling

with zipfile.ZipFile(zpath) as zf:
    names = zf.namelist()
    for pol in ['vv', 'vh']:
        ann = [n for n in names if f'/annotation/s1a-iw-grd-{pol}-' in n]
        tif = [n for n in names if f'/measurement/s1a-iw-grd-{pol}-' in n and n.endswith('.tiff')]
        if not ann or not tif:
            print(f'  {pol}: 缺文件', flush=True)
            continue
        xml = zf.read(ann[0]).decode('utf-8', errors='ignore')
        # 提取 GCP
        pts = re.findall(r'<geolocationGridPoint>(.*?)</geolocationGridPoint>', xml, re.S)
        gcps = []
        for p in pts:
            line = float(re.search(r'<line>(\S+)</line>', p).group(1))
            pixel = float(re.search(r'<pixel>(\S+)</pixel>', p).group(1))
            lat = float(re.search(r'<latitude>(\S+)</latitude>', p).group(1))
            lon = float(re.search(r'<longitude>(\S+)</longitude>', p).group(1))
            gcps.append(GroundControlPoint(line, pixel, lon, lat))
        # 打开原始 tiff（/vsizip/）并转 dB
        src = f'/vsizip/{zpath}/{tif[0]}'
        with rasterio.open(src) as ds:
            data = 10.0 * np.log10(np.maximum(ds.read(1).astype(np.float32), 1))
            t = from_gcps(gcps)
        # 目标网格
        dst_width = int(round((gtbounds.right - gtbounds.left) / res))
        dst_height = int(round((gtbounds.top - gtbounds.bottom) / res))
        dst_transform = rasterio.transform.from_bounds(
            gtbounds.left, gtbounds.bottom, gtbounds.right, gtbounds.top,
            dst_width, dst_height)
        out = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(source=data, destination=out,
                  src_transform=t, src_crs='EPSG:4326',
                  dst_transform=dst_transform, dst_crs='EPSG:4326',
                  resampling=Resampling.bilinear)
        out_tif = os.path.join(OUT, f'{date}_{pol.upper()}.tif')
        with rasterio.open(out_tif, 'w', driver='GTiff', height=dst_height, width=dst_width,
                           count=1, dtype='float32', crs='EPSG:4326',
                           transform=dst_transform) as dst:
            dst.write(out, 1)
        print(f'  {pol}: {out_tif} ({out.shape}) 完成', flush=True)
print('DONE', flush=True)
