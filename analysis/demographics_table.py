"""
Generates a participant demographics table for the paper.
Reads data/human.csv (Hebrew-language survey responses).
Outputs a formatted table to stdout and saves it as results/demographics_table.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/human.csv")
OUT_PATH = Path("results/demographics_table.csv")

df = pd.read_csv(DATA_PATH)

# Rename relevant columns to English
BIRTH_YEAR_COL = df.columns[2]   # שנת לידה
SEX_COL = df.columns[3]          # מין
EDU_COL = df.columns[4]          # רמת השכלה
OCC_COL = df.columns[5]          # מקצוע

# ── Age ─────────────────────────────────────────────────────────────────────
SURVEY_YEAR = 2025
df["age"] = SURVEY_YEAR - df[BIRTH_YEAR_COL]

# ── Sex translations ─────────────────────────────────────────────────────────
sex_map = {"נקבה": "Female", "זכר": "Male"}
df["sex_en"] = df[SEX_COL].map(sex_map).fillna("Other/Not reported")

# ── Education translations ───────────────────────────────────────────────────
edu_map = {
    "סטודנט":           "Student (undergraduate)",
    "תואר בראשון":      "Bachelor's degree",
    "תואר שני":         "Master's degree",
    "12 שנות לימוד":    "High school (12 years)",
    "דוקטורט":          "Doctorate",
}
df["edu_en"] = df[EDU_COL].map(edu_map).fillna(df[EDU_COL])

# ── Occupation: normalise architecture variants ───────────────────────────────
arch_variants = {"אדריכלות", "אדריכלית", "ארכיטקטורה", "אדריכל"}
df["occ_en"] = df[OCC_COL].apply(
    lambda v: "Architecture" if str(v).strip() in arch_variants else v
)

occ_other_map = {
    "הנדסת חשמל":           "Electrical engineering",
    "מנהל":                  "Management",
    "מורה":                  "Teaching",
    "מחקר":                  "Research",
    "עובדת סוציאלית":        "Social work",
    "מלונאות ומנהל עסקים":  "Hospitality & business management",
    "ניהול":                 "Management",
    "הנדסה אזרחית":          "Civil engineering",
}
df["occ_en"] = df["occ_en"].replace(occ_other_map)

# ══════════════════════════════════════════════════════════════════════════════
# Build table rows
# ══════════════════════════════════════════════════════════════════════════════

N = len(df)
rows = []

def pct(n): return f"{n / N * 100:.1f}%"
def add(label, value, n_or_range=""):
    rows.append({"Characteristic": label, "Value": value, "n / Range": n_or_range})

# Total
add("Total participants", "", str(N))

# Age
add("Age (years)", f"M = {df['age'].mean():.1f}, SD = {df['age'].std():.1f}",
    f"{int(df['age'].min())}–{int(df['age'].max())}")

# Sex
add("Sex", "", "")
for label, grp in [("Female", "Female"), ("Male", "Male")]:
    n = (df["sex_en"] == grp).sum()
    add(f"  {label}", f"{n} ({pct(n)})", "")

# Education
add("Education level", "", "")
edu_order = [
    "Student (undergraduate)",
    "Bachelor's degree",
    "Master's degree",
    "High school (12 years)",
    "Doctorate",
]
for lvl in edu_order:
    n = (df["edu_en"] == lvl).sum()
    if n:
        add(f"  {lvl}", f"{n} ({pct(n)})", "")

# Occupation
add("Occupation", "", "")
for occ, n in df["occ_en"].value_counts().items():
    add(f"  {occ}", f"{n} ({pct(n)})", "")

# ── Output ────────────────────────────────────────────────────────────────────
table = pd.DataFrame(rows)

col_widths = {
    "Characteristic": max(table["Characteristic"].str.len().max(), 20),
    "Value":          max(table["Value"].str.len().max(), 30),
    "n / Range":      12,
}

sep = "  "
header = sep.join(c.ljust(col_widths[c]) for c in table.columns)
divider = sep.join("-" * col_widths[c] for c in table.columns)

print("\nTable 1. Participant Demographics\n")
print(header)
print(divider)
for _, r in table.iterrows():
    print(sep.join(str(r[c]).ljust(col_widths[c]) for c in table.columns))

OUT_PATH.parent.mkdir(exist_ok=True)
table.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
