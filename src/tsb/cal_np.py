"""Numpy-only calendar features (shared by training and the service).

Platform clock: P = U - offset(Berlin wall-clock reading of U); local Berlin time = P + 2*offset.
`hol_days` is a table (days since 1970-01-01 -> 7 float features) generated once with the `holidays` package.
"""
import datetime as dt
import numpy as np

HOL_COLS = ["f_de", "f_at", "nat", "eve", "xmas_week", "bridge"]  # eve = Dec24 or Dec31

def _last_sunday_utc(year, month, hour_utc=1):
    d = dt.date(year, month, 31)
    while d.weekday() != 6:
        d -= dt.timedelta(days=1)
    return int(dt.datetime(d.year, d.month, d.day, hour_utc, tzinfo=dt.timezone.utc).timestamp())

def dst_windows(y0=2016, y1=2036):
    return np.array([[_last_sunday_utc(y, 3), _last_sunday_utc(y, 10)] for y in range(y0, y1 + 1)], dtype=np.int64)

_DST = dst_windows()

_BOUNDS = _DST.reshape(-1)          # sorted: [spring0, autumn0, spring1, autumn1, ...]

def _is_summer(U):
    """U: int64 epoch seconds of TRUE UTC instants -> bool (CEST)."""
    return (np.searchsorted(_BOUNDS, np.asarray(U, dtype=np.int64), side="right") % 2) == 1

def local_shift_seconds(P):
    """P: int64 platform epoch seconds -> seconds to add to get true Berlin local time (2*offset)."""
    P = np.asarray(P, dtype=np.int64)
    summer = _is_summer(P + 7200)             # hypothesis off=2h
    off = np.where(summer, 2, 1)
    return (2 * off * 3600).astype(np.int64)

def build_hol_table(start="2016-01-01", end="2036-12-31"):
    import holidays, pandas as pd
    days = pd.date_range(start, end, freq="D")
    yrs = range(days[0].year, days[-1].year + 1)
    states = ["BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV", "NI", "NW", "RP", "SL", "SN", "ST", "SH", "TH"]
    de_nat = holidays.country_holidays("DE", years=yrs)
    de_state = [holidays.country_holidays("DE", subdiv=s, years=yrs) for s in states]
    at = holidays.country_holidays("AT", years=yrs)
    rows = []
    for d in days:
        dd = d.date()
        f_de = float(np.mean([dd in h for h in de_state]))
        nat = 1.0 if dd in de_nat else 0.0
        f_at = 1.0 if dd in at else 0.0
        eve = 1.0 if (d.month, d.day) in ((12, 24), (12, 31)) else 0.0
        xmas = 1.0 if ((d.month == 12 and 27 <= d.day <= 30) or (d.month == 1 and 2 <= d.day <= 5)) else 0.0
        prev = (d - pd.Timedelta(days=1)).date(); nxt = (d + pd.Timedelta(days=1)).date()
        bridge = 1.0 if ((d.dayofweek == 4 and prev in de_nat) or (d.dayofweek == 0 and nxt in de_nat)) else 0.0
        rows.append((f_de, f_at, nat, eve, xmas, bridge))
    day0 = int((days[0] - pd.Timestamp("1970-01-01")).days)
    return day0, np.array(rows, dtype=np.float32)

class Calendar:
    def __init__(self, day0, table):
        self.day0, self.table = int(day0), np.asarray(table, dtype=np.float32)

    def features(self, P):
        """P: int64 platform epoch seconds -> dict(dow int8, hour float32, hol float32[n,6], doy float32)."""
        P = np.asarray(P, dtype=np.int64)
        L = P + local_shift_seconds(P)
        days = L // 86400
        dow = ((days + 3) % 7).astype(np.int8)                         # 1970-01-01 was a Thursday -> Mon=0
        hour = ((L % 86400) / 3600.0).astype(np.float32)
        idx = np.clip(days - self.day0, 0, len(self.table) - 1)
        hol = self.table[idx]
        # day of year from days via datetime64
        d64 = days.astype("datetime64[D]")
        doy = ((d64 - d64.astype("datetime64[Y]").astype("datetime64[D]")).astype(np.int64) + 1).astype(np.float32)
        return dict(dow=dow, hour=hour, hol=hol, doy=doy)
