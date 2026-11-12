"""Rename the Hebrew column headers of release/data/human.csv to English keys.

Metadata columns: participant_id, consent, birth_year, sex, education, field.
PANAS columns:    scene<N>_<item>   (N = 1..10, item = English PANAS item); values are Likert 1-5.
Hebrew categories in sex/education/field are left unchanged.

Usage: python tools/anglicise_human.py release/data/human.csv
"""
import re
import sys

import pandas as pd

# Hebrew keyword in the form's item text -> English PANAS item (mapping used for all results)
ITEM_MAP = {
    "להתעניין": "interested", "לחוץ": "nervous", "נרגש": "excited",
    "מוטרד": "upset", "חזק": "strong", "אשם": "guilty", "בהלה": "scared",
    "עוין": "hostile", "התלהבות": "enthusiastic", "גאה": "proud",
    "רוגז": "irritable", "עירני": "alert", "מבוייש": "ashamed",
    "שראה": "inspired", "עצבני": "jittery", "נחוש": "determined",
    "קשוב": "attentive", "מתח": "distressed", "פעיל": "active",
    "פחד": "afraid",
}
LIKERT = {"כלל לא": 1, "מעט": 2, "במידה בינונית": 3, "במידה רבה": 4, "רבה מאוד": 5}
META = ["participant_id", "consent", "birth_year", "sex", "education", "field"]


def main(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    new = list(META)
    for col in df.columns[6:]:
        m = re.search(r"\[עד כמה החלל [^[]+\]", col)
        kw = next(k for k in ITEM_MAP if k in m.group(0))
        s = re.search(r"\.(\d+)$", col)
        scene = (int(s.group(1)) if s else 0) + 1
        new.append(f"scene{scene}_{ITEM_MAP[kw]}")
    assert len(set(new)) == len(new) == 206, "duplicate or missing columns"
    df.columns = new
    for c in new[6:]:
        df[c] = df[c].map(lambda v: LIKERT[v.strip()] if v.strip() else "")
    df.to_csv(path, index=False, encoding="utf-8")
    print("renamed", len(new), "columns")


if __name__ == "__main__":
    main(sys.argv[1])
