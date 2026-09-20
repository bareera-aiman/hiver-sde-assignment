import os
import pandas as pd


INPUT_PATH = "results/agent_test_outputs.csv"
OUTPUT_PATH = "results/reply_baseline_outputs.csv"


TRIVIAL_REPLY = (
    "Thanks for contacting Spotify. "
    "Please provide more details about the issue so we can help."
)


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Missing required file: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8",
        encoding_errors="replace",
    )

    # We need the complete test set.
    if len(df) != 50:
        raise ValueError(
            f"Expected 50 completed test cases, "
            f"but found {len(df)}."
        )

    required_columns = [
        "case_id",
        "customer_message",
        "conversation_context",
        "reply",
        "retrieved_case_1",
        "retrieved_similarity_1",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    baseline_df = df[
        [
            "case_id",
            "customer_message",
            "conversation_context",
            "reply",
            "retrieved_case_1",
            "retrieved_similarity_1",
        ]
    ].copy()

    # Baseline 1: trivial generic reply.
    baseline_df["trivial_reply"] = TRIVIAL_REPLY

    # Baseline 2: use the top retrieved historical Spotify reply.
    # The actual historical reply is not currently stored in
    # agent_test_outputs.csv, so retrieve it from spotify_cases.csv.
    cases_df = pd.read_csv(
        "results/spotify_cases.csv",
        encoding="utf-8",
        encoding_errors="replace",
        usecols=[
            "case_id",
            "historical_spotify_reply",
        ],
    )

    cases_df["case_id"] = cases_df["case_id"].astype(str)
    baseline_df["retrieved_case_1"] = (
        baseline_df["retrieved_case_1"].astype(str)
    )

    baseline_df = baseline_df.merge(
        cases_df,
        left_on="retrieved_case_1",
        right_on="case_id",
        how="left",
        suffixes=("", "_historical"),
    )

    baseline_df = baseline_df.rename(
        columns={
            "historical_spotify_reply":
                "retrieval_only_reply"
        }
    )

    baseline_df = baseline_df.drop(
        columns=["case_id_historical"],
        errors="ignore",
    )

    baseline_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n--- REPLY BASELINES CREATED ---")
    print(f"Cases: {len(baseline_df)}")
    print("Baseline 1: Trivial generic reply")
    print("Baseline 2: Top-1 retrieval reply")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()