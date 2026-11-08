import os, sys
sys.path.insert(0, r'D:\DisasterM3\UrbanSARFloods\repro')
import _render_figures_housestyle as R

os.makedirs(R.OUT, exist_ok=True)
R.fig_gateflow_wide()
print('done')
