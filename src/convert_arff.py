import pandas as pd

ARFF_PATH = "data/raw/KDDTest+.arff"   # 🔴 CHANGE THIS
OUTPUT_CSV = "data/raw/data.csv"

data_started = False
rows = []

with open(ARFF_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower().startswith("@data"):
            data_started = True
            continue
        if data_started:
            rows.append(line.split(","))

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("✅ ARFF data section extracted and saved as CSV")
