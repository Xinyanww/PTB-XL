"""
Step 3 — Within-patient comparison.

Question: if the same patient has one reviewed and one unreviewed record,
do the NORM labels disagree, and in which direction?

This removes between-patient differences, but introduces two new problems
checked here: (a) the two records are not the same recording, and
(b) the reviewed records were selected for review, not sampled.

Outputs
  results/03_closest_pairs.csv
  results/03_discordant_pairs.csv
"""

import pandas as pd

from common import (RESULTS_DIR, UNREVIEWED_DEVICE, UNREVIEWED_SITE,
                    closest_within_patient_pairs, ensure_dirs, load_ptbxl,
                    pct, section)

ensure_dirs()
df = load_ptbxl()
pairs = closest_within_patient_pairs(df)

section("1. Patients with records under both pathways")
print(f"patients: {len(pairs):,}")

section("2. Are the paired records the same recording?")
print("gap between closest reviewed/unreviewed records (days):")
print(pairs["gap_days"].describe().round(2).to_string())
print(f"\nidentical timestamps : {(pairs['abs_gap_seconds'] == 0).sum()}")
print(f"same device          : {pct(pairs['same_device'].mean())}")
print(f"unreviewed one later : {pct((pairs['gap_seconds'] > 0).mean())}")
print("""
Reading:
  No pair shares a timestamp, so these are distinct recordings, typically
  days apart on the same device. Ageing between records is not a concern
  (median gap is days, order is balanced), but a real change in the
  patient's state between recordings is.
""")

section("3. NORM agreement — the full 2x2")
tab = pd.crosstab(pairs["is_norm_rev"], pairs["is_norm_unrev"],
                  rownames=["reviewed NORM"], colnames=["unreviewed NORM"])
print(tab.to_string())

informative = pairs[(pairs["is_norm_rev"] == 1) | (pairs["is_norm_unrev"] == 1)]
disc = pairs[pairs["is_norm_rev"] != pairs["is_norm_unrev"]]
rev_norm_only = ((pairs["is_norm_rev"] == 1) & (pairs["is_norm_unrev"] == 0)).sum()
unrev_norm_only = ((pairs["is_norm_unrev"] == 1) & (pairs["is_norm_rev"] == 0)).sum()

print(f"\nraw agreement                        : {pct(1 - len(disc) / len(pairs))}")
print(f"patients with >=1 NORM (informative) : {len(informative)}")
print(f"discordant                           : {len(disc)} "
      f"({pct(len(disc) / len(informative))} of informative)")
print(f"reviewed NORM, unreviewed not        : {rev_norm_only}")
print(f"unreviewed NORM, reviewed not        : {unrev_norm_only}")
print(f"discordant pairs on different devices: {(~disc['same_device']).sum()}")
print("""
Reading:
  Raw agreement looks high only because most pairs are abnormal on both
  sides. Among patients where at least one record is NORM, disagreement is
  substantial and lopsided toward the reviewed record being NORM.
  The sample is small; this is descriptive, not an estimate.
""")

section("4. Who was selected for review? (same site, same device)")
blk = df[(df["site"] == UNREVIEWED_SITE) & (df["device"] == UNREVIEWED_DEVICE)]
sel = (blk.groupby("reviewed")
          .agg(n=("ecg_id", "size"),
               norm_rate=("is_norm", "mean"),
               mean_age=("age_clean", "mean"),
               share_female=("sex", "mean"),
               second_opinion=("second_opinion", "mean"),
               median_year=("year", "median"))
          .round(3))
print(sel.to_string())
print(f"\nsecond-opinion rate, whole dataset: {pct(df['second_opinion'].mean())}")
print("""
Reading:
  Review was spread across the whole period (not a one-off batch), so it
  followed some selection rule. Reviewed records are older and less often
  NORM. Note second_opinion can only be 1 for reviewed records, so the 0 in
  the unreviewed row is mechanical; the informative comparison is against
  the dataset-wide rate.

  This selection pushes the reviewed record toward looking LESS normal,
  opposite to the direction observed in section 3, so it cannot by itself
  explain that result. What it cannot rule out: if records were sent to
  review because the machine read looked wrong, the observed direction is
  that rule, not a human-vs-machine difference.
""")

pairs.to_csv(RESULTS_DIR / "03_closest_pairs.csv", index=False)
disc.to_csv(RESULTS_DIR / "03_discordant_pairs.csv", index=False)
