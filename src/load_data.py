import pandas as pd
import yaml
import os

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

raw_path = params["data"]["raw"]
processed_path = params["data"]["processed"]

# Load CSV (NOT ARFF)
df = pd.read_csv(raw_path)

# Ensure processed directory exists
os.makedirs(os.path.dirname(processed_path), exist_ok=True)

# Save processed CSV
df.to_csv(processed_path, index=False)

print("CSV loaded and saved to", processed_path)

