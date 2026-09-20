import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


INPUT_PATH = "results/agent_test_outputs.csv"


def main():
    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8",
        encoding_errors="replace",
    )

    print("\n--- CLASSIFICATION REPORT ---")

    print(
        classification_report(
            df["gold_intent"],
            df["predicted_intent"],
            zero_division=0,
        )
    )

    print("\n--- CONFUSION MATRIX ---")

    labels = sorted(
        set(df["gold_intent"]) |
        set(df["predicted_intent"])
    )

    matrix = confusion_matrix(
        df["gold_intent"],
        df["predicted_intent"],
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    print(matrix_df)

    print("\n--- INCORRECT PREDICTIONS ---")

    mistakes = df[
        df["gold_intent"] != df["predicted_intent"]
    ]

    for _, row in mistakes.iterrows():
        print("\n" + "=" * 70)
        print("Case:", row["case_id"])
        print("Customer:", row["customer_message"])
        print("Gold:", row["gold_intent"])
        print("Predicted:", row["predicted_intent"])
        print("Reply:", row["reply"])
        print(
            "Top retrieval similarity:",
            round(row["retrieved_similarity_1"], 3),
        )


if __name__ == "__main__":
    main()