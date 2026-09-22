"""
Shared loading and helper functions for the PTB-XL label-provenance analysis.

Every numbered script imports from here, so the definitions of
"reviewed", "NORM", the unreviewed site/device, and the within-patient
pairing are written once.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

PHYSIONET_DIR = "ptb-xl/1.0.3"
SAMPLING_HZ = 100  # low-resolution PTB-XL records

# The records without documented human validation sit almost entirely
# on this site/device combination (see 01_describe_provenance.py).
UNREVIEWED_SITE = 0
UNREVIEWED_DEVICE = "CS100    3"  # PTB-XL pads device names with spaces

QUALITY_COLS = [
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
    "extra_beats",
    "pacemaker",
]
DIAG_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ---------------------------------------------------------------- loading
def load_ptbxl(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load ptbxl_database.csv and add the derived columns used everywhere."""
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)

    df["scp_dict"] = df["scp_codes"].apply(ast.literal_eval)

    # NORM = the NORM statement is present at any likelihood.
    df["is_norm"] = df["scp_dict"].apply(lambda d: int("NORM" in d))

    # "reviewed" = documented human validation in the metadata.
    df["reviewed"] = df["validated_by_human"].astype(bool)

    df["recording_date"] = pd.to_datetime(df["recording_date"], errors="coerce")
    df["year"] = df["recording_date"].dt.year

    # PTB-XL encodes ages above 89 as 300; treat those as missing for means.
    df["age_clean"] = df["age"].where(df["age"] <= 89)

    for col in QUALITY_COLS:
        df[f"{col}_flag"] = df[col].notna().astype(int)

    diagnostic = scp[scp["diagnostic"] == 1]["diagnostic_class"].dropna()
    df["diag_classes"] = df["scp_dict"].apply(
        lambda d: sorted({diagnostic[k] for k in d if k in diagnostic.index})
    )

    df["in_unreviewed_block"] = (
        (df["site"] == UNREVIEWED_SITE) & (df["device"] == UNREVIEWED_DEVICE)
    )
    return df


# ---------------------------------------------------------------- pairing
def closest_within_patient_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every patient with at least one reviewed and one unreviewed record,
    return the single reviewed/unreviewed pair closest in time.

    Columns from the unreviewed record carry the suffix _unrev,
    columns from the reviewed record carry _rev.
    gap_seconds > 0 means the unreviewed record was taken later.
    """
    per_patient = df.groupby("patient_id")["reviewed"].agg(["size", "mean"])
    spanning = per_patient[
        (per_patient["size"] > 1)
        & (per_patient["mean"] > 0)
        & (per_patient["mean"] < 1)
    ].index

    sub = df[df["patient_id"].isin(spanning)]
    pairs = sub[~sub["reviewed"]].merge(
        sub[sub["reviewed"]], on="patient_id", suffixes=("_unrev", "_rev")
    )
    pairs["gap_seconds"] = (
        pairs["recording_date_unrev"] - pairs["recording_date_rev"]
    ).dt.total_seconds()
    pairs["abs_gap_seconds"] = pairs["gap_seconds"].abs()
    pairs["gap_days"] = pairs["abs_gap_seconds"] / 86400
    pairs["same_device"] = pairs["device_unrev"] == pairs["device_rev"]

    return (
        pairs.sort_values("abs_gap_seconds")
        .drop_duplicates("patient_id", keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------- waveforms
def load_lead(filename_lr: str, lead: str = "II") -> tuple[np.ndarray, np.ndarray]:
    """
    Stream one low-resolution PTB-XL record from PhysioNet (no full download)
    and return (time in seconds, signal in mV) for a single lead.
    """
    import wfdb

    folder, record = filename_lr.rsplit("/", 1)
    signal, meta = wfdb.rdsamp(record, pn_dir=f"{PHYSIONET_DIR}/{folder}")
    x = signal[:, meta["sig_name"].index(lead)]
    t = np.arange(len(x)) / SAMPLING_HZ
    return t, x


# ---------------------------------------------------------------- output
def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"
