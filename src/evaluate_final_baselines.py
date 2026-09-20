import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DEV_PATH = "results/spotify_dev.csv"
TEST_PATH = "results/spotify_test.csv"
OUTPUT_PATH = "results/final_baseline_predictions.csv"


dev = pd.read_csv(
    DEV_PATH,
    encoding="utf-8",
    encoding_errors="replace",
)

test = pd.read_csv(
    TEST_PATH,
    encoding="utf-8",
    encoding_errors="replace",
)


X_dev = dev["customer_message"]
y_dev = dev["intent_label"]

X_test = test["customer_message"]
y_test = test["intent_label"]


# --------------------------------------------------
# 1. Majority baseline
# --------------------------------------------------

majority_intent = y_dev.value_counts().idxmax()

majority_predictions = [majority_intent] * len(test)

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions,
)


# --------------------------------------------------
# 2. TF-IDF + Logistic Regression
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
        ),
    ),
])


model.fit(X_dev, y_dev)

tfidf_predictions = model.predict(X_test)

tfidf_accuracy = accuracy_score(
    y_test,
    tfidf_predictions,
)


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

output = test[
    [
        "case_id",
        "customer_id",
        "customer_message",
        "intent_label",
    ]
].copy()

output["majority_prediction"] = majority_predictions
output["tfidf_prediction"] = tfidf_predictions

output.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n--- FINAL HELD-OUT TEST BASELINES ---")

print(f"Development cases: {len(dev)}")
print(f"Test cases:        {len(test)}")

print("\nMajority baseline")
print(f"Majority intent:       {majority_intent}")
print(
    f"Correct predictions:  "
    f"{int(majority_accuracy * len(test))}/{len(test)}"
)
print(f"Accuracy:              {majority_accuracy:.2%}")

print("\nTF-IDF + Logistic Regression")
print(
    f"Correct predictions:  "
    f"{int(tfidf_accuracy * len(test))}/{len(test)}"
)
print(f"Accuracy:              {tfidf_accuracy:.2%}")

print("\n--- TF-IDF CLASSIFICATION REPORT ---")
print(
    classification_report(
        y_test,
        tfidf_predictions,
        zero_division=0,
    )
)

print(f"\nSaved: {OUTPUT_PATH}")
