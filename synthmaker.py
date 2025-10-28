import pandas as pd
import random

# Load cleaned dataset
df = pd.read_csv("gooddata.csv")

# Keep only valid names (no missing values)
df = df.dropna(subset=["Full Name", "label"])

# Split names into first and last parts by label
def split_names(group_df):
    first_parts = []
    last_parts = []
    for name in group_df["Full Name"]:
        parts = str(name).strip().split()
        if len(parts) == 1:
            first_parts.append(parts[0])
            last_parts.append(parts[0])
        else:
            first_parts.append(parts[0])
            last_parts.append(parts[-1])
    return first_parts, last_parts

# Separate by label
df_ones = df[df["label"] == 1]
df_zeros = df[df["label"] == 0]

first_ones, last_ones = split_names(df_ones)
first_zeros, last_zeros = split_names(df_zeros)

# Generate synthetic names within label group
def generate_names(first_parts, last_parts, num):
    synthetic_names = []
    for _ in range(num):
        first = random.choice(first_parts)
        last = random.choice(last_parts)
        synthetic_names.append(f"{first} {last}")
    return synthetic_names

# Generate separately for 1s and 0s
names_ones = generate_names(first_ones, last_ones, len(df_ones))
names_zeros = generate_names(first_zeros, last_zeros, len(df_zeros))

# Combine synthetic datasets
synthetic_df = pd.DataFrame({
    "Full Name": names_ones + names_zeros,
    "label": [1]*len(names_ones) + [0]*len(names_zeros)
})

# Drop duplicates and original names
synthetic_df = synthetic_df[~synthetic_df["Full Name"].isin(df["Full Name"])].drop_duplicates()

# Save
synthetic_df.to_csv("synthetic_dataset.csv", index=False)

print(f"✅ Synthetic dataset created with {len(synthetic_df)} unique names")
print("💾 Saved as synthetic_dataset.csv")
