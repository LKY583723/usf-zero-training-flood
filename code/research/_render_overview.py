import os
import _render_figures_housestyle as R

os.makedirs(R.OUT, exist_ok=True)
R.fig_overview_col()
print('done')
