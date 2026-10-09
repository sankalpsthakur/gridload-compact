"""SMARD raw history in the TS-Arena platform clock + helpers.

Platform convention (verified against the archive, see calibrate notes in STATUS.md):
the data-portal SMARD plugin reads UTC epoch ms, builds NAIVE datetimes of the UTC wall clock, then
`tz_localize('Europe/Berlin', nonexistent='shift_forward')` and stores the result as UTC.  So
    platform_ts = U - offset(Berlin wall-clock reading of U)      (offset = +1h CET, +2h CEST)
i.e. platform wall clock is 1 h (winter) / 2 h (summer) earlier than true UTC.
Ambiguous wall clocks (fall back) are resolved to the first (DST) occurrence by the plugin code path
(its second-occurrence check never fires), collapsing the repeated hour.
"""
import os
import numpy as np, pandas as pd
from tsb import ROOT
from tsb.archive import REG2UID

REGIONS = list(REG2UID)  # 10 SMARD net-load series

def load_raw(region):
    df = pd.read_parquet(f"{ROOT}/data/smard/{region}.parquet")
    return df

def to_platform_clock(df):
    """df: ts_ms, value (true UTC epoch).  Returns Series indexed by platform UTC timestamps (15 min grid)."""
    naive = pd.to_datetime(df.ts_ms, unit="ms")  # naive UTC wall clock
    idx = pd.DatetimeIndex(naive)
    loc = idx.tz_localize("Europe/Berlin", ambiguous=np.ones(len(idx), dtype=bool), nonexistent="shift_forward")
    plat = loc.tz_convert("UTC")
    s = pd.Series(df.value.to_numpy(), index=plat)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def quarter_series(region):
    return to_platform_clock(load_raw(region))

def hourly_from_quarter(s):
    """Platform hourly view: mean of the four quarter-hours of each hour, label = start of hour (verified)."""
    return s.resample("1h", label="left", closed="left").agg(lambda x: x.mean() if x.notna().sum() == 4 else np.nan)
