"""Calendar features on the platform clock.

Platform clock for SMARD series = true UTC - offset(Berlin), offset = 1h (CET) / 2h (CEST), so
local Berlin time = platform ts + 2*offset (1h->+2h, 2h->+4h).  Local date drives day-type features.
Holiday tables are deterministic public-calendar knowledge (python `holidays` package, MIT), baked into
a small array at build time; the service itself needs only numpy.
"""
import numpy as np, pandas as pd

def local_hours_offset(platform_idx):
    """Return 2*offset (hours) to add to platform ts to get true local Berlin time, via the DST window of U=P+off."""
    P = pd.DatetimeIndex(platform_idx)
    # Berlin offset at an instant = (local naive - utc naive)
    def berlin_off(U):
        loc = U.tz_convert("Europe/Berlin")
        return np.asarray((loc.tz_localize(None) - U.tz_localize(None)) / pd.Timedelta(hours=1))
    U2 = P + pd.Timedelta(hours=2)   # hypothesis: summer time (off=2h)
    o2 = berlin_off(U2)
    U1 = P + pd.Timedelta(hours=1)
    o1 = berlin_off(U1)
    off = np.where(o2 == 2, 2, np.where(o1 == 1, 1, 2))
    return (2 * off).astype(np.int8)

def holiday_table(start="2017-01-01", end="2029-12-31"):
    import holidays
    days = pd.date_range(start, end, freq="D")
    yrs = range(days[0].year, days[-1].year + 1)
    states = ["BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV", "NI", "NW", "RP", "SL", "SN", "ST", "SH", "TH"]
    de_nat = holidays.country_holidays("DE", years=yrs)
    de_state = [holidays.country_holidays("DE", subdiv=s, years=yrs) for s in states]
    at = holidays.country_holidays("AT", years=yrs)
    rows = []
    for d in days:
        dd = d.date()
        f_de = np.mean([dd in h for h in de_state])
        f_at = 1.0 if dd in at else 0.0
        nat = 1.0 if dd in de_nat else 0.0
        dec24 = 1.0 if (d.month, d.day) == (12, 24) else 0.0
        dec31 = 1.0 if (d.month, d.day) == (12, 31) else 0.0
        xmas_week = 1.0 if ((d.month == 12 and 27 <= d.day <= 30) or (d.month == 1 and 2 <= d.day <= 5)) else 0.0
        prev = (d - pd.Timedelta(days=1)).date(); nxt = (d + pd.Timedelta(days=1)).date()
        bridge = 1.0 if ((d.dayofweek == 4 and prev in de_nat) or (d.dayofweek == 0 and nxt in de_nat)) else 0.0
        rows.append((f_de, f_at, nat, dec24, dec31, xmas_week, bridge))
    return pd.DataFrame(rows, index=days, columns=["f_de", "f_at", "nat", "dec24", "dec31", "xmas_week", "bridge"]).astype(np.float32)

HOL_COLS = ["f_de", "f_at", "nat", "dec24", "dec31", "xmas_week", "bridge"]

def calendar_arrays(platform_idx, hol=None):
    """Per-timestamp arrays aligned with platform_idx (UTC DatetimeIndex):
    returns dict(local_dow[int8], local_hour[float], hol[float32, n x 7], doy[float])."""
    P = pd.DatetimeIndex(platform_idx)
    shift = local_hours_offset(P).astype(np.int64)
    L = P + pd.to_timedelta(shift, unit="h")
    if hol is None:
        hol = holiday_table()
    d0 = L.tz_localize(None).normalize()
    h = hol.reindex(d0).to_numpy().astype(np.float32)
    return dict(local_dow=np.asarray(L.dayofweek, dtype=np.int8), local_hour=np.asarray(L.hour + L.minute / 60.0, dtype=np.float32),
                hol=h, doy=np.asarray(L.dayofyear, dtype=np.float32), shift=shift)
