import pandas as pd

# load dataset
df = pd.read_csv("somali_classifier_full_dataset.csv")

# find duplicates based on "Full Name"
dupes = df[df.duplicated(subset=["Full Name"], keep=False)]

if dupes.empty:
    print("✅ No duplicate names found!")
else:
    print(f"⚠️ Found {dupes['Full Name'].nunique()} duplicated names "
          f"({len(dupes)} total duplicate rows).")

    # group by label (Somali=1, Non-Somali=0)
    label_counts = dupes["label"].value_counts()

    somali_count = label_counts.get(1, 0)
    nonsomali_count = label_counts.get(0, 0)

    print("\n📊 Duplicate Breakdown:")
    print(f"Somali (label=1): {somali_count}")
    print(f"Non-Somali (label=0): {nonsomali_count}")
    print(f"Total duplicates: {somali_count + nonsomali_count}")

    # calculate percentages
    total = somali_count + nonsomali_count
    if total > 0:
        print(f"\n⚖️ Ratio:")
        print(f"Somali: {(somali_count / total) * 100:.2f}%")
        print(f"Non-Somali: {(nonsomali_count / total) * 100:.2f}%")

    # list the duplicate names (for inspection)
    print("\n🧾 Duplicate Names:\n")
    print(dupes["Full Name"].value_counts().head(20))  # show top 20 for preview

    # save duplicates & cleaned dataset
    dupes.to_csv("duplicates_found.csv", index=False)
    df.drop_duplicates(subset=["Full Name"], keep="first").to_csv("cleaned_dataset.csv", index=False)

    print("\n💾 Files saved:")
    print("- duplicates_found.csv (all duplicates)")
    print("- cleaned_dataset.csv (unique names only)")
