"""
Step 5 (exploratory) — Probe the 'Edit:' field in free-text reports.

Observation from Step 4: some reviewed reports appear to keep the machine's
original German text and append the human revision after 'Edit:'.
If that holds generally, the machine reading and the human reading of the
SAME recording are both recoverable, which none of Steps 2-4 could achieve.

This script only checks whether that reading is plausible. It does not
analyse the edits.

Outputs
  results/05_edit_examples.csv
"""

import re

import pandas as pd

from common import RESULTS_DIR, ensure_dirs, load_ptbxl, pct, section

ensure_dirs()
df = load_ptbxl()
report = df["report"].fillna("")

df["has_edit"] = report.str.contains("Edit:", regex=False)
df["has_unconfirmed_tag"] = report.str.contains("unbest", case=False)  # 'unbestätigter bericht'

section("1. Where does 'Edit:' appear?")
print(pd.crosstab(df["has_edit"], df["reviewed"], margins=True).to_string())

section("2. Does the machine's 'unconfirmed report' tag survive review?")
print(pd.crosstab(df["has_unconfirmed_tag"], df["reviewed"], margins=True).to_string())


def edit_part(text: str) -> str:
    return text.split("Edit:", 1)[1] if "Edit:" in text else ""


def parenthetical_codes(text: str) -> set[str]:
    """Codes inside the last (...) of the edit text, e.g. '(NORM 100, LAFB 100)'."""
    groups = re.findall(r"\(([^)]*)\)", text)
    return set(re.findall(r"\b([A-Z][A-Z0-9]{1,6})\b", groups[-1])) if groups else set()


section("3. Do the codes in '( ... )' match the stored scp_codes?")
edits = df[df["has_edit"]].copy()
edits["edit_text"] = edits["report"].apply(edit_part).str.strip()
edits["paren_codes"] = edits["edit_text"].apply(parenthetical_codes)
edits["scp_keys"] = edits["scp_dict"].apply(lambda d: set(d))
with_paren = edits[edits["paren_codes"].str.len() > 0]

print(f"records with 'Edit:'                    : {len(edits):,}")
print(f"  ... with a parenthetical code list    : {len(with_paren):,}")
if len(with_paren):
    subset = with_paren.apply(lambda r: r["paren_codes"] <= r["scp_keys"], axis=1)
    exact = with_paren.apply(
        lambda r: r["paren_codes"] == {k for k in r["scp_keys"] if k != "SR"}, axis=1)
    print(f"  parenthetical codes all in scp_codes  : {pct(subset.mean())}")
    print(f"  exact match (ignoring rhythm code SR) : {pct(exact.mean())}")

section("4. Examples to read by hand")
cols = ["ecg_id", "reviewed", "report", "edit_text", "scp_codes"]
sample = edits.sample(min(15, len(edits)), random_state=0)[cols]
for _, r in sample.iterrows():
    print(f"\n[{r['ecg_id']}] reviewed={r['reviewed']}")
    print(f"  report : {r['report'][:160]}")
    print(f"  edit   : {r['edit_text'][:100]}")
    print(f"  scp    : {r['scp_codes']}")
edits[cols].to_csv(RESULTS_DIR / "05_edit_examples.csv", index=False)

print("""
Reading:
  If 'Edit:' appears only on reviewed records, the 'unconfirmed' machine tag
  survives review, and the parenthetical codes match scp_codes, then the
  text before 'Edit:' is plausibly the machine reading and the stored label
  is the human one. That would allow a same-recording comparison.
  If any of these fail, this field is not usable as described.
""")
