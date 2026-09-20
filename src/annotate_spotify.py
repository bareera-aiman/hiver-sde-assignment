import os
import pandas as pd

INPUT_PATH = "results/spotify_intent_sample.csv"
OUTPUT_PATH = "results/spotify_intent_sample_labeled.csv"

VALID_CASE_TYPES = {
    "1": "SUPPORT_REQUEST",
    "2": "FOLLOW_UP",
    "3": "ACKNOWLEDGEMENT",
    "4": "FEEDBACK",
    "5": "OTHER",
}

VALID_INTENTS = {
    "1": "PLAYBACK_OR_APP_ISSUE",
    "2": "SUBSCRIPTION_OR_PAYMENT",
    "3": "ACCOUNT",
    "4": "FEATURE_OR_UI",
    "5": "CONTENT_OR_CATALOG",
    "6": "PROMOTION_OR_EVENT",
    "7": "PRIVACY_OR_DATA",
    "8": "SUPPORT_CHANNEL_OR_LANGUAGE",
    "9": "OTHER",
}


def choose_label(title, options):
    print(f"\n{title}")

    for key, value in options.items():
        print(f"{key}. {value}")

    while True:
        choice = input("Enter number: ").strip()

        if choice in options:
            return options[choice]

        print("Invalid choice. Try again.")


# -----------------------------------
# Load existing labeled file if present
# -----------------------------------
if os.path.exists(OUTPUT_PATH):
    df = pd.read_csv(OUTPUT_PATH)
    print("Resuming existing annotation file.")
else:
    df = pd.read_csv(INPUT_PATH)
    print("Starting from original annotation sample.")

# Ensure annotation columns are explicitly text columns.
for column in ["case_type", "intent_label", "notes"]:
    if column not in df.columns:
        df[column] = ""

    df[column] = df[column].fillna("").astype("string")


print("Cases loaded:", len(df))


# -----------------------------------
# Find first unlabeled case
# -----------------------------------
case_type_series = df["case_type"].fillna("").astype(str).str.strip()

unlabeled_indices = df.index[case_type_series == ""]

if len(unlabeled_indices) == 0:
    print("All cases are already labeled.")
    raise SystemExit

start_index = unlabeled_indices[0]

print("Starting from case:",
      df.loc[start_index, "case_id"])


# -----------------------------------
# Annotation loop
# -----------------------------------
for i in range(start_index, len(df)):

    row = df.loc[i]

    # Skip already labeled rows
    if str(row["case_type"]).strip() != "" and \
       str(row["case_type"]).lower() != "nan":
        continue

    print("\n" + "=" * 100)
    print(f"CASE {row['case_id']}")
    print("=" * 100)

    print("\nCONTEXT:")
    print(row["context"])

    print("\nLATEST CUSTOMER MESSAGE:")
    print(row["customer_message"])

    case_type = choose_label(
        "CASE TYPE",
        VALID_CASE_TYPES,
    )

    intent = choose_label(
        "INTENT",
        VALID_INTENTS,
    )

    notes = input(
        "\nNotes (optional, press Enter to skip): "
    ).strip()

    df.loc[i, "case_type"] = case_type
    df.loc[i, "intent_label"] = intent
    df.loc[i, "notes"] = notes

    # Save after every case.
    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved.")


print("\nAnnotation complete.")
print("Saved to:", OUTPUT_PATH)