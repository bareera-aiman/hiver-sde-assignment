import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score


HUMAN_PATH = "results/reply_quality_evaluation_human.xlsx"
LLM_PATH = "results/reply_quality_llm_judgments.csv"
OUTPUT_PATH = "results/reply_quality_agreement.csv"

CRITERIA = [
    "correctness",
    "relevance",
    "grounding",
    "actionability",
    "communication",
    "overall",
]

HUMAN_COLUMNS = {
    "correctness": "correctness_score",
    "relevance": "relevance_score",
    "grounding": "grounding_score",
    "actionability": "actionability_score",
    "communication": "communication_score",
    "overall": "human_score",
}


def main():
    if not os.path.exists(HUMAN_PATH):
        raise FileNotFoundError(
            f"Missing: {HUMAN_PATH}"
        )

    if not os.path.exists(LLM_PATH):
        raise FileNotFoundError(
            f"Missing: {LLM_PATH}"
        )

    human_df = pd.read_excel(
        HUMAN_PATH,
        sheet_name="Human Evaluation",
    )

    llm_df = pd.read_csv(
        LLM_PATH,
        encoding="utf-8",
        encoding_errors="replace",
    )

    if len(human_df) != 40:
        raise ValueError(
            f"Expected 40 human-rated cases, found {len(human_df)}."
        )

    if len(llm_df) != 40:
        raise ValueError(
            f"Expected 40 LLM judgments, found {len(llm_df)}."
        )

    human_df["case_id"] = human_df["case_id"].astype(str)
    llm_df["case_id"] = llm_df["case_id"].astype(str)

    merged = human_df[
        ["case_id"] + list(HUMAN_COLUMNS.values())
    ].merge(
        llm_df[
            ["case_id"] + CRITERIA
        ],
        on="case_id",
        how="inner",
        suffixes=("_human", "_llm"),
    )

    if len(merged) != 40:
        raise ValueError(
            f"Expected 40 matched cases, found {len(merged)}."
        )

    rows = []

    for criterion in CRITERIA:
        human_col = HUMAN_COLUMNS[criterion]
        llm_col = criterion

        human_scores = pd.to_numeric(
            merged[human_col], errors="coerce"
        )
        llm_scores = pd.to_numeric(
            merged[llm_col], errors="coerce"
        )

        if human_scores.isna().any():
            raise ValueError(
                f"Missing human scores for {criterion}."
            )

        if llm_scores.isna().any():
            raise ValueError(
                f"Missing LLM scores for {criterion}."
            )

        exact = (
            human_scores == llm_scores
        ).mean()

        within_one = (
            (human_scores - llm_scores).abs() <= 1
        ).mean()

        kappa = cohen_kappa_score(
            human_scores,
            llm_scores,
            weights="quadratic",
        )

        rows.append(
            {
                "criterion": criterion,
                "cases": len(merged),
                "exact_agreement": exact,
                "within_1_agreement": within_one,
                "weighted_cohen_kappa": kappa,
                "human_mean": human_scores.mean(),
                "llm_mean": llm_scores.mean(),
            }
        )

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n--- REPLY QUALITY AGREEMENT ---")
    print(results.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()