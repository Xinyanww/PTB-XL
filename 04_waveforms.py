"""
Step 4 — Look at the raw waveforms of the discordant pairs.

Question: when the two pathways disagree, do the two recordings actually
look alike? If not, the disagreement may reflect the signal, not the reader.

Waveforms are streamed from PhysioNet record by record (no 1.7 GB download).

Outputs
  figures/pairs/patient_<id>.png   one figure per discordant pair
  figures/closest_discordant_pair.png
"""

import matplotlib.pyplot as plt

from common import (FIG_DIR, QUALITY_COLS, closest_within_patient_pairs,
                    ensure_dirs, load_lead, load_ptbxl, section)

ensure_dirs()
(FIG_DIR / "pairs").mkdir(exist_ok=True)

df = load_ptbxl()
pairs = closest_within_patient_pairs(df)
disc = pairs[pairs["is_norm_rev"] != pairs["is_norm_unrev"]].sort_values("abs_gap_seconds")


def label(scp: dict) -> str:
    """SCP statement codes as stored in PTB-XL, e.g. 'NORM, SR'."""
    return ", ".join(scp)


def gap_text(sec: float) -> str:
    sec = abs(sec)
    return f"{sec:.0f} s" if sec < 3600 else f"{sec / 86400:.1f} days"


def plot_pair(row, path, title=None):
    t_r, x_r = load_lead(row["filename_lr_rev"])
    t_u, x_u = load_lead(row["filename_lr_unrev"])

    # chronological order: earlier record on top
    panels = [
        (row["recording_date_rev"], t_r, x_r, "documented human validation", row["scp_dict_rev"]),
        (row["recording_date_unrev"], t_u, x_u, "no documented human validation", row["scp_dict_unrev"]),
    ]
    panels.sort(key=lambda p: p[0])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    for ax, (ts, t, x, kind, scp) in zip(axes, panels):
        ax.plot(t, x, lw=0.8)
        ax.set_title(f"{ts:%Y-%m-%d %H:%M:%S} — {kind}: {label(scp)}", loc="left", fontsize=10)
        ax.set_ylabel("mV")
        ax.grid(alpha=0.25)
    axes[1].set_xlabel("seconds")
    fig.suptitle(title or f"Patient {int(row['patient_id'])} — two 10-s ECGs "
                          f"{gap_text(row['gap_seconds'])} apart (lead II)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


section("1. All discordant pairs")
for _, row in disc.iterrows():
    note = "" if row["same_device"] else "   <- different devices; interpret with care"
    print(f"patient {int(row['patient_id']):>6}  gap {gap_text(row['gap_seconds']):>10}  "
          f"reviewed NORM={row['is_norm_rev']}  unreviewed NORM={row['is_norm_unrev']}{note}")
    plot_pair(row, FIG_DIR / "pairs" / f"patient_{int(row['patient_id'])}.png")

section("2. The closest discordant pair")
key = disc.iloc[0]
print(f"patient {int(key['patient_id'])}, gap {gap_text(key['gap_seconds'])}, "
      f"ECG ids {key['ecg_id_rev']} (reviewed) and {key['ecg_id_unrev']} (unreviewed)")

print("\nrecording-quality annotations for the two records:")
for suffix, name in [("_rev", "reviewed"), ("_unrev", "unreviewed")]:
    flags = {c: key[c + suffix] for c in QUALITY_COLS if key[c + suffix] == key[c + suffix]}
    print(f"  {name:>10}: {flags if flags else 'none flagged'}")

plot_pair(key, FIG_DIR / "closest_discordant_pair.png",
          title=f"Same patient, two 10-second ECGs recorded "
                f"{gap_text(key['gap_seconds'])} apart (lead II)")

print("""
Reading:
  For the closest pair the two tracings are visibly different, so the label
  disagreement there is not a disagreement about the same signal. Either the
  rhythm changed within about a minute, or the first recording was poor and
  was immediately repeated (check the quality flags above). In both cases the
  label belongs to the ten seconds recorded, not to the patient, which is
  the assumption every design in Steps 2-3 relied on.
""")
