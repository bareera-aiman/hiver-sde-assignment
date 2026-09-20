import os
import pandas as pd

from agent import generate_agent_response


TEST_PATH = "results/spotify_test.csv"
OUTPUT_PATH = "results/agent_test_outputs.csv"

# Safety limit:
# Never make more than 30 new API calls in one run.
MAX_NEW_CASES_PER_RUN = int(
    os.environ.get("MAX_NEW_CASES_PER_RUN", "30")
)


def main():
    test_df = pd.read_csv(
        TEST_PATH,
        encoding="utf-8",
        encoding_errors="replace",
    )

    # Load previous checkpoint if available.
    if os.path.exists(OUTPUT_PATH):
        output_df = pd.read_csv(
            OUTPUT_PATH,
            encoding="utf-8",
            encoding_errors="replace",
        )

        completed_ids = set(
            output_df["case_id"].astype(str)
        )

        print(
            f"Found {len(output_df)} completed cases. "
            "Resuming evaluation."
        )
    else:
        output_df = pd.DataFrame()
        completed_ids = set()

        print(
            "No previous results found. "
            "Starting evaluation."
        )

    total = len(test_df)
    new_cases = 0

    for i, row in test_df.iterrows():

        case_id = str(row["case_id"])

        # Skip cases already completed.
        if case_id in completed_ids:
            continue

        # Safety stop before exceeding the per-run limit.
        if new_cases >= MAX_NEW_CASES_PER_RUN:
            print(
                f"\nSafety limit reached: "
                f"{MAX_NEW_CASES_PER_RUN} new API calls."
            )
            break

        print(f"Processing {i + 1}/{total}...")

        try:
            agent_result = generate_agent_response(
                customer_message=row["customer_message"],
                conversation_context=row.get(
                    "context",
                    "",
                ),
                top_k=3,
                exclude_case_id=row["case_id"],
                exclude_customer_id=row["customer_id"],
            )

            retrieved_cases = agent_result["retrieved_cases"]

            result = {
                "case_id": row["case_id"],
                "customer_id": row["customer_id"],
                "customer_message": row["customer_message"],
                "conversation_context": row.get(
                    "context",
                    "",
                ),
                "gold_intent": row["intent_label"],
                "predicted_intent": agent_result["intent"],
                "reply": agent_result["reply"],
                "action": agent_result["action"],
                "reason": agent_result["reason"],
                "retrieved_case_1": retrieved_cases.iloc[0][
                    "case_id"
                ],
                "retrieved_similarity_1": retrieved_cases.iloc[0][
                    "similarity"
                ],
                "retrieved_case_2": retrieved_cases.iloc[1][
                    "case_id"
                ],
                "retrieved_similarity_2": retrieved_cases.iloc[1][
                    "similarity"
                ],
                "retrieved_case_3": retrieved_cases.iloc[2][
                    "case_id"
                ],
                "retrieved_similarity_3": retrieved_cases.iloc[2][
                    "similarity"
                ],
            }

            result_df = pd.DataFrame([result])

            if output_df.empty:
                output_df = result_df
            else:
                output_df = pd.concat(
                    [output_df, result_df],
                    ignore_index=True,
                )

            # SAVE IMMEDIATELY.
            output_df.to_csv(
                OUTPUT_PATH,
                index=False,
                encoding="utf-8",
            )

            completed_ids.add(case_id)
            new_cases += 1

            print(
                f"Saved case {case_id}. "
                f"Completed: {len(output_df)}/{total}"
            )

        except Exception as error:
            print(
                "\nEvaluation stopped because of an error."
            )
            print(f"Case: {case_id}")
            print(f"Error: {error}")
            print(
                "\nAll completed cases have already "
                "been saved."
            )
            break

    if output_df.empty:
        print("\nNo cases completed.")
        return

    accuracy = (
        output_df["gold_intent"]
        == output_df["predicted_intent"]
    ).mean()

    print("\n--- EVALUATION STATUS ---")
    print(f"Cases completed: {len(output_df)}/{total}")
    print(f"New API calls this run: {new_cases}")
    print(f"Current intent accuracy: {accuracy:.2%}")
    print(f"Saved to: {OUTPUT_PATH}")

    if len(output_df) == total:
        print("\nFULL EVALUATION COMPLETE.")
    else:
        print(
            "\nEvaluation is incomplete. "
            "Run the same command later to resume."
        )


if __name__ == "__main__":
    main()