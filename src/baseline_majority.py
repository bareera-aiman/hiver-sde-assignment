import pandas as pd

INPUT_PATH = "results/spotify_gold_annotations.csv"
OUTPUT_PATH = "results/baseline_majority_predictions.csv"

# Load the human-labeled gold dataset.
df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8",
    encoding_errors="replace",
)

# Find the most common intent.
majority_intent = df["intent_label"].value_counts().idxmax()

# Predict the majority intent for every case.
df["predicted_intent"] = majority_intent

# Calculate accuracy.
accuracy = (
    df["predicted_intent"] == df["intent_label"]
).mean()

print("--- MAJORITY BASELINE ---")
print("Number of cases:", len(df))
print("Majority intent:", majority_intent)
print("Correct predictions:", int(accuracy * len(df)))
print(f"Accuracy: {accuracy:.4f} ({accuracy:.1%})")

# Save predictions so the evaluation is reproducible.
df[
    [
        "case_id",
        "intent_label",
        "predicted_intent",
    ]
].to_csv(
    OUTPUT_PATH,
    index=False,
)

print("Saved:", OUTPUT_PATH)