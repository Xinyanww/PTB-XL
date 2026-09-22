"""
Step 2 — Why a between-record comparison cannot identify a provenance effect.

Question: are unreviewed records labeled differently, and can that
difference be attributed to the annotation process?

Short answer: they are labeled very differently, and it cannot.

Outputs
  results/02_block_vs_rest.csv
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import (DIAG_CLASSES, QUALITY_COLS, RESULTS_DIR, UNREVIEWED_DEVICE,
                    ensure_dirs, load_ptbxl, pct, section)

ensure_dirs()
df = load_ptbxl()
block = df["in_unreviewed_block"]

section("1. The unreviewed block vs everything else")
rows = {}
for name, g in [("unreviewed block", df[block]), ("rest", df[~block])]:
    r = {
        "n": len(g),
        "share reviewed": g["reviewed"].mean(),
        "years": f"{int(g['year'].min())}-{int(g['year'].max())}",
        "mean age": g["age_clean"].mean(),
        "share female (sex==1)": g["sex"].mean(),
    }
    for c in DIAG_CLASSES:
        r[f"{c} prevalence"] = g["diag_classes"].apply(lambda d, c=c: c in d).mean()
    for c in QUALITY_COLS:
        r[f"{c} flagged"] = g[f"{c}_flag"].mean()
    rows[name] = r
table = pd.DataFrame(rows)
print(table.to_string())
table.to_csv(RESULTS_DIR / "02_block_vs_rest.csv")

pm = table.loc["pacemaker flagged"]
print(f"\npacemaker ratio (block / rest): {pm.iloc[0] / pm.iloc[1]:.1f}x")

print("""
Reading:
  The block differs on every axis at once: site, device, era, age, sex mix,
  diagnostic mix, and pacemaker rate. The quality flags also move in
  inconsistent directions (baseline drift lower, static noise higher), which
  suggests they were annotated differently across the two groups and are not
  valid controls.
""")

section("2. What a naive regression would say (shown to illustrate the problem)")
model_df = df[["is_norm", "in_unreviewed_block", "age_clean", "sex", "year"]
              + [f"{c}_flag" for c in QUALITY_COLS]].dropna().copy()
model_df["block"] = model_df["in_unreviewed_block"].astype(int)
controls = " + ".join(["age_clean", "sex", "year"] + [f"{c}_flag" for c in QUALITY_COLS])

m0 = smf.logit("is_norm ~ block", data=model_df).fit(disp=0)
m1 = smf.logit(f"is_norm ~ block + {controls}", data=model_df).fit(disp=0)
for label, m in [("unadjusted", m0), ("adjusted", m1)]:
    print(f"{label:>10}: OR for NORM = {np.exp(m.params['block']):.2f}")

print("""
Reading:
  Adjustment does not shrink the gap; if anything it widens. That is not
  evidence of a provenance effect. 'block' is collinear with site, device,
  era, and case mix, and the adjustment set cannot separate them. The
  coefficient has no causal interpretation and is not reported as a finding.
""")

section("3. Within site 0: could the two devices be compared instead?")
s0 = df[df["site"] == 0]
s0_tab = (s0.groupby("device")
            .agg(n=("ecg_id", "size"),
                 share_reviewed=("reviewed", "mean"),
                 norm_rate=("is_norm", "mean"),
                 mean_age=("age_clean", "mean"),
                 first_year=("year", "min"),
                 last_year=("year", "max"))
            .query("n >= 100").round(3))
print(s0_tab.to_string())
print(f"""
Reading:
  Holding the site fixed does not help. The two devices serve different
  populations (very different mean age and NORM rate), so '{UNREVIEWED_DEVICE.split()[0]}
  vs the other device' is still a comparison of patients, not of pipelines.
  This motivates the within-patient comparison in Step 3.
""")
