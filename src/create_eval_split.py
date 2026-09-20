import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

INPUT = "results/spotify_gold_annotations.csv"
DEV_OUTPUT = "results/spotify_dev.csv"
TEST_OUTPUT = "results/spotify_test.csv"

df = pd.read_csv(INPUT)

X = df["customer_message"]
y = df["intent_label"]
groups = df["customer_id"]

# Use grouped + stratified splitting:
# - grouped: the same customer cannot appear in both dev and test
# - stratified: tries to preserve the intent distribution
splitter = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

# Find the fold closest to our desired 50-example test set.
best_fold = None
best_score = float("inf")

for fold, (_, test_idx) in enumerate(splitter.split(X, y, groups)):
    test_size = len(test_idx)

    # Prefer a test set close to 50 examples.
    score = abs(test_size - 50)

    if score < best_score:
        best_score = score
        best_fold = (fold, test_idx)

fold, test_idx = best_fold

test = df.iloc[test_idx].copy()
dev = df.drop(test.index).copy()

dev.to_csv(DEV_OUTPUT, index=False)
test.to_csv(TEST_OUTPUT, index=False)

print("Evaluation split created")
print("------------------------")
print(f"Development set: {len(dev)}")
print(f"Test set:        {len(test)}")
print(f"Selected fold:   {fold}")

print("\nCustomer overlap:")
print(
    "Overlapping customers:",
    len(set(dev["customer_id"]) & set(test["customer_id"]))
)

print("\nDevelopment intent distribution:")
print(dev["intent_label"].value_counts())

print("\nTest intent distribution:")
print(test["intent_label"].value_counts())

print("\nFiles:")
print(DEV_OUTPUT)
print(TEST_OUTPUT)