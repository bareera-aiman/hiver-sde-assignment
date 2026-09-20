import pandas as pd

BASE_PATH = "results/spotify_intent_sample.csv"
OUTPUT_PATH = "results/spotify_gold_annotations.csv"

ANNOTATION_FILES = [
    "results/spotify_intent_sample_labeled.csv",  # Cases 1–50
    "results/annotation/friend_2_batch.csv",     # Cases 51–100
    "results/annotation/friend_3_batch.csv",     # Cases 101–150
    "results/annotation/you_batch.csv",           # Cases 151–200
]

print("Loading clean base dataset...")
base = pd.read_csv(
    BASE_PATH,
    encoding="utf-8",
    encoding_errors="replace",
)

print("Base rows:", len(base))

# The clean sample already contains empty annotation columns.
# Remove them so the completed annotations can be merged in cleanly.
base = base.drop(
    columns=["case_type", "intent_label", "notes"],
    errors="ignore",
)


annotation_frames = []

for path in ANNOTATION_FILES:
    print(f"Loading annotations: {path}")

    df = pd.read_csv(
        path,
        encoding="utf-8",
        encoding_errors="replace",
    )
    # Keep only actually annotated rows.
    df = df[df["intent_label"].notna()].copy()

    if "notes" not in df.columns:
        df["notes"] = ""

    annotation_frames.append(
        df[["case_id", "case_type", "intent_label", "notes"]]
    )


annotations = pd.concat(
    annotation_frames,
    ignore_index=True,
)


# Check for duplicate case IDs before merging.
duplicates = annotations[
    annotations["case_id"].duplicated(keep=False)
]

if len(duplicates) > 0:
    print("\nERROR: Duplicate case IDs found:")
    print(duplicates.sort_values("case_id").to_string(index=False))
    raise ValueError("Duplicate case IDs detected.")


# Keep only the annotation columns and merge by case_id.
result = base.merge(
    annotations,
    on="case_id",
    how="left",
    validate="one_to_one",
)


print("\n--- FINAL GOLD DATASET ---")
print("Rows:", len(result))
print("Unique case IDs:", result["case_id"].nunique())
print("Duplicate case IDs:", result["case_id"].duplicated().sum())
print("Case type labels:", result["case_type"].notna().sum())
print("Intent labels:", result["intent_label"].notna().sum())


missing = result[result["intent_label"].isna()]

if len(missing) > 0:
    print("\nMissing intent labels:")
    print(missing["case_id"].tolist())
    raise ValueError("Some cases are still missing intent labels.")


result.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)

print("\nSaved:", OUTPUT_PATH)
print("Merge complete.")