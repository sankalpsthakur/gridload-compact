"""Write the baked holiday table (public calendar knowledge, `holidays` package, MIT) used by training and the service."""
import numpy as np
from tsb import cal_np, ROOT
day0, tab = cal_np.build_hol_table()
np.savez(f"{ROOT}/data/cache/hol_table.npz", day0=day0, table=tab, cols=np.array(cal_np.HOL_COLS))
print(tab.shape, cal_np.HOL_COLS)
