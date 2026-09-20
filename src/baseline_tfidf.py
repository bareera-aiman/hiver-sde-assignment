import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Our 200 hand-labeled examples are the gold dataset.
INPUT_PATH = "results/spotify_gold_annotations.csv"

df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8",
    encoding_errors="replace",
)

X = df["customer_message"]
y = df["intent_label"]


# TF-IDF converts text into numerical features.
# (1, 2) means we use both individual words and two-word phrases.
# Example: "payment failed" becomes a useful feature.
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


# 5-fold stratified CV keeps the intent distribution roughly similar
# across the five folds.
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)


# Each prediction is made by a model that did NOT train on that example.
# This gives us out-of-fold predictions for all 200 cases.
predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
)


accuracy = accuracy_score(y, predictions)


print("--- TF-IDF + LOGISTIC REGRESSION ---")
print("Total examples:", len(X))
print("Cross-validation folds:", 3)
print(f"Out-of-fold accuracy: {accuracy:.4f} ({accuracy:.1%})")

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y,
        predictions,
        zero_division=0,
    )
)