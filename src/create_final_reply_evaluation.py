import os
import pandas as pd


AGENT_PATH = "results/agent_test_outputs.csv"
BASELINE_PATH = "results/reply_baseline_outputs.csv"
OUTPUT_PATH = "results/reply_quality_evaluation.csv"

SAMPLE_SIZE = 40
RANDOM_SEED = 42


def main():
    if not os.path.exists(AGENT_PATH):
        raise FileNotFoundError(
            f"Missing: {AGENT_PATH}"
        )

    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(
            f"Missing: {BASELINE_PATH}"
        )

    agent_df = pd.read_csv(
        AGENT_PATH,
        encoding="utf-8",
        encoding_errors="replace",
    )

    baseline_df = pd.read_csv(
        BASELINE_PATH,
        encoding="utf-8",
        encoding_errors="replace",
    )

    # Require the complete final test set.
    if len(agent_df) != 50:
        raise ValueError(
            f"Expected 50 agent results, "
            f"found {len(agent_df)}."
        )

    if len(baseline_df) != 50:
        raise ValueError(
            f"Expected 50 baseline results, "
            f"found {len(baseline_df)}."
        )

    merged = agent_df.merge(
        baseline_df[
            [
                "case_id",
                "trivial_reply",
                "retrieval_only_reply",
            ]
        ],
        on="case_id",
        how="inner",
    )

    if len(merged) != 50:
        raise ValueError(
            "Agent and baseline results do not contain "
            "the same 50 test cases."
        )

    # Reproducible 40-case sample.
    sample = merged.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED,
    )

    sample = sample.sort_values(
        by="case_id"
    ).reset_index(drop=True)

    evaluation_df = sample[
        [
            "case_id",
            "customer_message",
            "conversation_context",
            "reply",
            "trivial_reply",
            "retrieval_only_reply",
            "retrieved_case_1",
            "retrieved_case_2",
            "retrieved_case_3",
        ]
    ].copy()

    # Human evaluation columns.
    evaluation_df["human_score"] = ""
    evaluation_df["human_notes"] = ""

    evaluation_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n--- FINAL REPLY EVALUATION PACKAGE ---")
    print(f"Test cases available: {len(merged)}")
    print(f"Evaluation cases: {len(evaluation_df)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()