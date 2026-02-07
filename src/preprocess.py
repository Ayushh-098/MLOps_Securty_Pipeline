import pandas as pd
import yaml
import os

with open("params.yaml") as f:
    params = yaml.safe_load(f)

input_path = params["data"]["processed"]
output_path = params["data"]["clean"]

os.makedirs(os.path.dirname(output_path), exist_ok=True)

df = pd.read_csv(input_path)

# Example preprocessing (adjustable later)
df = df.dropna()

df.to_csv(output_path, index=False)

print("Preprocessing completed. Clean data saved to", output_path)
