"""Readers for the public TS-Arena-Archive (HF DAG-UPB/TS-Arena-Archive, CC BY 4.0) partitions we fetched."""
import glob, os, re
import numpy as np, pandas as pd
from tsb import ROOT
A = f"{ROOT}/data/archive"
SCHED = {2: "smard_load_challenge_24h_15min", 5: "smard_load_challenge_72h_1h"}
UID2REG = {
    "electricity_net_consumption_BZ_AT_quarterhour_410": "AT",
    "electricity_net_consumption_BZ_DE-LU_quarterhour_410": "DE-LU",
    "electricity_net_consumption_Country_DE_quarterhour_410": "DE",
    "electricity_net_consumption_Country_LU_quarterhour_410": "LU",
    "electricity_net_consumption_TSO_50Hertz_quarterhour_410": "50Hertz",
    "electricity_net_consumption_TSO_APG_quarterhour_410": "APG",
    "electricity_net_consumption_TSO_Amprion_quarterhour_410": "Amprion",
    "electricity_net_consumption_TSO_Creos_quarterhour_410": "Creos",
    "electricity_net_consumption_TSO_TenneT_quarterhour_410": "TenneT",
    "electricity_net_consumption_TSO_TransnetBW_quarterhour_410": "TransnetBW",
}
REG2UID = {v: k for k, v in UID2REG.items()}

def round_dirs(defn):
    """registration_start (ISO str) -> dir suffix, from the context_data partitions."""
    base = f"{A}/context_data/schedule_id={SCHED[defn]}"
    out = {}
    for d in sorted(glob.glob(f"{base}/registration_month=*/registration_start=*")):
        m = re.search(r"registration_start=(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})", d)
        out[f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}Z"] = d.split("context_data/")[1]
    return out

def _read_dir(top, sub):
    fl = sorted(glob.glob(f"{A}/{top}/{sub}/*.parquet"))
    if not fl:
        return None
    return pd.read_parquet(fl[0])  # duplicates files are identical copies

def context(defn, sub):
    df = _read_dir("context_data", sub)
    return {u: g.set_index("ts").value.sort_index() for u, g in df.groupby("unique_id")}, int(df.round_id.iloc[0])

def truth(defn, sub):
    df = _read_dir("ground_truth", sub)
    if df is None:
        return None
    return {u: g.set_index("ts").value.sort_index() for u, g in df.groupby("unique_id")}

def forecasts(defn, sub):
    """-> {model_name: {unique_id: DataFrame(ts index, value, q_0.1..q_0.9)}}"""
    out = {}
    for md in sorted(glob.glob(f"{A}/forecasts/{sub}/model_name=*")):
        name = md.split("model_name=")[1]
        fl = sorted(glob.glob(f"{md}/*.parquet"))
        if not fl:
            continue
        df = pd.read_parquet(fl[0])
        df["ts"] = pd.to_datetime(df.ts, utc=True)
        out[name] = {u: g.set_index("ts").sort_index().drop(columns="unique_id") for u, g in df.groupby("unique_id")}
    return out
