import sys
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler

print("Step 1: Loading Covertype dataset...", flush=True)
sys.stdout.flush()

data = fetch_covtype()
print(f"  Shape: {data.data.shape}", flush=True)

print("Step 2: Normalizing...", flush=True)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(data.data)

num_points, num_dims = df_scaled.shape
output_file = "covertype_for_kmeans.txt"

print(f"Step 3: Writing {num_points} points x {num_dims} dims...", flush=True)
print("  This may take 1-2 minutes, please wait...", flush=True)

with open(output_file, "w") as f:
    f.write(f"{num_points} {num_dims}\n")
    for i, row in enumerate(df_scaled):
        f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
        if i % 50000 == 0:
            print(f"  Progress: {i}/{num_points} rows written...", flush=True)

print(f"\nDone! File saved: {output_file}", flush=True)

# Verify
with open(output_file, "r") as f:
    first_line = f.readline().strip()
print(f"Verification - First line: {first_line}", flush=True)