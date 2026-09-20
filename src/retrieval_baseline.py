import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Historical Spotify interactions.
# Each row contains a customer message and the reply Spotify gave.
INPUT_PATH = "results/spotify_cases.csv"

df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8",
    encoding_errors="replace",
)

# We only need the customer message and historical Spotify reply
# for this first retrieval experiment.
df = df[
    ["case_id", "customer_message", "historical_spotify_reply"]
].dropna()


# TF-IDF converts customer messages into numerical vectors.
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
)

message_vectors = vectorizer.fit_transform(
    df["customer_message"]
)


def find_similar_cases(customer_message, top_k=3):
    """
    Find historical customer messages that are most similar
    to the new customer message.

    Returns the top matching historical cases and their
    original Spotify replies.
    """

    # Convert the new message using the SAME vectorizer
    # that was fitted on the historical messages.
    query_vector = vectorizer.transform([customer_message])

    # Cosine similarity measures how similar the query is
    # to every historical customer message.
    similarities = cosine_similarity(
        query_vector,
        message_vectors,
    )[0]

    # Get the indices of the highest-scoring matches.
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()

    # Keep the similarity score so we can inspect
    # whether the retrieved examples are actually relevant.
    results["similarity"] = similarities[top_indices]

    return results


# Test the retriever on one real customer message.
test_message = (
    "My Spotify is not playing any songs. "
    "It keeps stopping when I try to listen."
)

results = find_similar_cases(test_message)

print("--- RETRIEVAL TEST ---")
print("\nCustomer message:")
print(test_message)

print("\n--- SIMILAR HISTORICAL CASES ---")

for _, row in results.iterrows():
    print(f"\nCase ID: {row['case_id']}")
    print(f"Similarity: {row['similarity']:.3f}")
    print(f"Customer: {row['customer_message']}")
    print(f"Spotify reply: {row['historical_spotify_reply']}")