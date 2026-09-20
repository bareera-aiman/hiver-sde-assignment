import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_PATH = "results/spotify_cases.csv"


df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8",
    encoding_errors="replace",
)

df = df[
    [
        "case_id",
        "customer_id",
        "customer_message",
        "context",
        "historical_spotify_reply",
    ]
].dropna(
    subset=[
        "customer_message",
        "historical_spotify_reply",
    ]
)


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
)

message_vectors = vectorizer.fit_transform(
    df["customer_message"]
)


def find_similar_cases(
    customer_message,
    top_k=3,
    exclude_case_id=None,
    exclude_customer_id=None,
):
    candidates = df.copy()

    if exclude_case_id is not None:
        candidates = candidates[
            candidates["case_id"] != exclude_case_id
        ]

    if exclude_customer_id is not None:
        candidates = candidates[
            candidates["customer_id"] != exclude_customer_id
        ]

    candidate_indices = candidates.index

    query_vector = vectorizer.transform(
        [customer_message]
    )

    similarities = cosine_similarity(
        query_vector,
        message_vectors[candidate_indices],
    )[0]

    top_positions = similarities.argsort()[
        -top_k:
    ][::-1]

    selected_indices = candidate_indices[
        top_positions
    ]

    results = df.loc[
        selected_indices
    ].copy()

    results["similarity"] = similarities[
        top_positions
    ]

    return results.reset_index(drop=True)