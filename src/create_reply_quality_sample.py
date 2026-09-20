import os
import pandas as pd


INPUT_PATH = "results/agent_test_outputs.csv"
OUTPUT_PATH = "results/reply_quality_human_sample.csv"

SAMPLE_SIZE = 40
RANDOM_SEED = 42


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

    # We require the complete 50-case test output.
    if len(df) != 50:
        raise ValueError(
            f"Expected 50 completed test cases, "
            f"but found {len(df)}.\n"
            "Do not create the final reply-quality sample "
            "until the agent evaluation is complete."
        )

    required_columns = [
        "case_id",
        "customer_message",
        "conversation_context",
        "reply",
        "retrieved_case_1",
        "retrieved_similarity_1",
        "retrieved_case_2",
        "retrieved_similarity_2",
        "retrieved_case_3",
        "retrieved_similarity_3",
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

    sample = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED,
    ).copy()

    sample = sample.sort_values(
        by="case_id"
    ).reset_index(drop=True)

    # Columns presented to the human evaluator.
    human_sample = sample[
        [
            "case_id",
            "customer_message",
            "conversation_context",
            "reply",
            "retrieved_case_1",
            "retrieved_similarity_1",
            "retrieved_case_2",
            "retrieved_similarity_2",
            "retrieved_case_3",
            "retrieved_similarity_3",
        ]
    ].copy()

    # Blank columns for manual evaluation.
    human_sample["human_score"] = ""
    human_sample["human_notes"] = ""

    human_sample.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n--- REPLY QUALITY SAMPLE ---")
    print(f"Source cases: {len(df)}")
    print(f"Human-evaluation cases: {len(human_sample)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()