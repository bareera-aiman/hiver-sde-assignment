import pandas as pd

DATA_PATH = "data/twcs.csv"
OUTPUT_PATH = "results/spotify_cases.csv"

BRAND = "SpotifyCares"
MAX_CONTEXT_MESSAGES = 6


# -----------------------------------
# Load data
# -----------------------------------
df = pd.read_csv(DATA_PATH)

print("Loaded rows:", len(df))


# -----------------------------------
# Normalize tweet IDs
# -----------------------------------
df["tweet_id"] = pd.to_numeric(
    df["tweet_id"],
    errors="coerce"
).astype("Int64")

df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
).astype("Int64")


# -----------------------------------
# Build tweet lookup
# -----------------------------------
tweets = (
    df.set_index("tweet_id")[
        [
            "author_id",
            "inbound",
            "created_at",
            "text",
            "in_response_to_tweet_id",
        ]
    ]
    .to_dict("index")
)


# -----------------------------------
# Find Spotify support tweets
# -----------------------------------
spotify_support = df[
    (df["inbound"] == False) &
    (df["author_id"] == BRAND)
].copy()

print(
    "Spotify support tweets:",
    len(spotify_support)
)


# -----------------------------------
# Keep Spotify replies to customers
# -----------------------------------
# We want:
#
# CUSTOMER
#    ↓
# SPOTIFY
#
# not:
#
# SPOTIFY
#    ↓
# CUSTOMER
#
# Therefore the parent tweet must be inbound=True.

support_replies = spotify_support[
    spotify_support["in_response_to_tweet_id"].notna()
].copy()


customer_ids = set(
    df[
        df["inbound"] == True
    ]["tweet_id"].dropna()
)


support_replies = support_replies[
    support_replies["in_response_to_tweet_id"].isin(customer_ids)
].copy()


print(
    "Spotify replies directly to customers:",
    len(support_replies)
)


# -----------------------------------
# Reconstruct context BEFORE customer
# -----------------------------------
def get_previous_context(tweet_id, max_messages=6):
    """
    Walk backwards from a customer tweet.

    The returned context contains the customer message
    and messages immediately preceding it.
    """

    context = []
    current_id = tweet_id

    for _ in range(max_messages):

        if pd.isna(current_id):
            break

        tweet = tweets.get(current_id)

        if tweet is None:
            break

        role = (
            "CUSTOMER"
            if tweet["inbound"]
            else "SPOTIFY"
        )

        context.append(
            f"[{role}] {tweet['text']}"
        )

        current_id = tweet["in_response_to_tweet_id"]

    context.reverse()

    return "\n".join(context)


# -----------------------------------
# Build case records
# -----------------------------------
cases = []

for case_number, (_, support_row) in enumerate(
    support_replies.iterrows(),
    start=1
):

    customer_id = support_row[
        "in_response_to_tweet_id"
    ]

    customer = tweets.get(customer_id)

    if customer is None:
        continue

    cases.append(
        {
            "case_id": case_number,
            "customer_tweet_id": customer_id,
            "support_tweet_id": support_row["tweet_id"],
            "customer_id": customer["author_id"],
            "customer_message": customer["text"],
            "context": get_previous_context(
                customer_id,
                MAX_CONTEXT_MESSAGES,
            ),
            "historical_spotify_reply": support_row["text"],
            "customer_created_at": customer["created_at"],
            "support_created_at": support_row["created_at"],
        }
    )


# -----------------------------------
# Create dataframe
# -----------------------------------
result = pd.DataFrame(cases)


# -----------------------------------
# Save
# -----------------------------------
result.to_csv(
    OUTPUT_PATH,
    index=False,
)


# -----------------------------------
# Summary
# -----------------------------------
print("\n--- CASE DATASET ---")
print("Cases created:", len(result))

print("\nColumns:")
for column in result.columns:
    print("-", column)

print("\nSaved to:", OUTPUT_PATH)


# -----------------------------------
# Preview
# -----------------------------------
print("\n--- FIRST 5 CASES ---")

for _, row in result.head(5).iterrows():

    print("\n" + "=" * 90)
    print(f"CASE {row['case_id']}")

    print("\nCONTEXT:")
    print(row["context"])

    print("\nHISTORICAL SPOTIFY REPLY:")
    print(row["historical_spotify_reply"])


print("\nBuild complete.")