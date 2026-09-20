import pandas as pd

from escalation_policy import decide_escalation


INPUT_PATH = "results/agent_test_outputs.csv"
OUTPUT_PATH = "results/escalation_test_outputs.csv"


def main():
    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8",
        encoding_errors="replace",
    )

    results = []

    for _, row in df.iterrows():
        action, reason = decide_escalation(
            intent=row["predicted_intent"],
            customer_message=row["customer_message"],
            conversation_context=row["conversation_context"],
        )

        results.append(
            {
                "case_id": row["case_id"],
                "gold_intent": row["gold_intent"],
                "predicted_intent": row["predicted_intent"],
                "customer_message": row["customer_message"],
                "retrieval_similarity": row[
                    "retrieved_similarity_1"
                ],
                "action": action,
                "reason": reason,
            }
        )

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n--- ESCALATION SUMMARY ---")
    print(output_df["action"].value_counts())

    print("\n--- ESCALATED CASES ---")

    escalated = output_df[
        output_df["action"] == "ESCALATE"
    ]

    for _, row in escalated.iterrows():
        print("\n" + "=" * 70)
        print("Case:", row["case_id"])
        print("Message:", row["customer_message"])
        print("Predicted intent:", row["predicted_intent"])
        print(
            "Retrieval similarity:",
            round(row["retrieval_similarity"], 3),
        )
        print("Reason:", row["reason"])

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()