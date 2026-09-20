import json
import os
import time

import pandas as pd
from groq import Groq


INPUT_PATH = "results/reply_quality_evaluation.csv"
OUTPUT_PATH = "results/reply_quality_llm_judgments.csv"

MODEL = "openai/gpt-oss-20b"

CRITERIA = [
    "correctness",
    "relevance",
    "grounding",
    "actionability",
    "communication",
    "overall",
]

client = Groq(api_key=os.environ["GROQ_API_KEY"])


JUDGE_PROMPT = """
You are evaluating the quality of an AI-generated customer-support reply.

Evaluate ONLY the provided AI reply against:
1. The customer's current message
2. The conversation context

Do NOT treat the historical Spotify reply as ground truth.
Do NOT assume that a reply is correct merely because it sounds professional.
Do NOT invent missing context.

Score each criterion from 1 to 5:

CORRECTNESS
5 = Correct and appropriate
4 = Mostly correct, with minor issues
3 = Acceptable/mixed
2 = Significant problems
1 = Incorrect or misleading

RELEVANCE
5 = Directly addresses the customer's issue
4 = Relevant with minor unnecessary content
3 = Partly relevant
2 = Mostly irrelevant
1 = Does not address the issue

GROUNDING
5 = Fully supported by the message/context
4 = Mostly grounded
3 = Some questionable assumptions
2 = Several unsupported claims
1 = Invents important facts or scenarios

ACTIONABILITY
5 = Gives a clear and useful next step
4 = Useful guidance with minor limitations
3 = Some useful guidance
2 = Limited practical help
1 = Does not help the customer move forward

COMMUNICATION
5 = Clear, concise and professional
4 = Good communication with minor issues
3 = Acceptable
2 = Poorly communicated
1 = Very unclear or inappropriate

OVERALL
Give your overall assessment from 1 to 5.

Important:
- Judge the AI reply itself, not whether it matches a historical support reply.
- Penalize hallucinated or unsupported details.
- A polite reply can still receive a low score.
- A short reply can receive a high score if it is appropriate for the context.
- Use the conversation context when interpreting short follow-ups or acknowledgements.

Return ONLY valid JSON with exactly these fields:
{
  "correctness": integer,
  "relevance": integer,
  "grounding": integer,
  "actionability": integer,
  "communication": integer,
  "overall": integer,
  "reason": "brief explanation"
}
"""


def judge_reply(row):
    prompt = f"""
{JUDGE_PROMPT}

CUSTOMER MESSAGE:
{row["customer_message"]}

CONVERSATION CONTEXT:
{row["conversation_context"]}

AI-GENERATED REPLY:
{row["reply"]}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        reasoning_effort="low",
        include_reasoning=False,
        max_completion_tokens=300,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "reply_quality_judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "correctness": {"type": "integer"},
                        "relevance": {"type": "integer"},
                        "grounding": {"type": "integer"},
                        "actionability": {"type": "integer"},
                        "communication": {"type": "integer"},
                        "overall": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "correctness",
                        "relevance",
                        "grounding",
                        "actionability",
                        "communication",
                        "overall",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response.")

    result = json.loads(content)

    for criterion in CRITERIA:
        score = result[criterion]

        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(
                f"Invalid {criterion} score: {score}"
            )

    return result


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

    if len(df) != 40:
        raise ValueError(
            f"Expected exactly 40 evaluation cases, found {len(df)}."
        )

    required_columns = [
        "case_id",
        "customer_message",
        "conversation_context",
        "reply",
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

    # Resume safely if the judge is interrupted.
    if os.path.exists(OUTPUT_PATH):
        results_df = pd.read_csv(
            OUTPUT_PATH,
            encoding="utf-8",
            encoding_errors="replace",
        )
    else:
        results_df = pd.DataFrame(
            columns=[
                "case_id",
                *CRITERIA,
                "reason",
            ]
        )

    completed_ids = set(
        results_df["case_id"].astype(str)
    ) if not results_df.empty else set()

    pending = df[
        ~df["case_id"].astype(str).isin(completed_ids)
    ].copy()

    print("\n--- REPLY QUALITY LLM JUDGE ---")
    print(f"Total cases: {len(df)}")
    print(f"Already judged: {len(completed_ids)}")
    print(f"Remaining: {len(pending)}")

    for _, row in pending.iterrows():
        case_id = str(row["case_id"])

        print(f"\nJudging case {case_id}...")

        try:
            result = judge_reply(row)

            result_row = {
                "case_id": case_id,
                **result,
            }

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame([result_row]),
                ],
                ignore_index=True,
            )

            results_df.to_csv(
                OUTPUT_PATH,
                index=False,
                encoding="utf-8",
            )

            print(
                f"Saved case {case_id}. "
                f"Overall: {result['overall']}"
            )

            # Small delay to avoid unnecessarily aggressive requests.
            time.sleep(1)

        except Exception as exc:
            print(
                f"\nERROR on case {case_id}: {exc}"
            )
            print(
                "Progress already saved. "
                "Fix the issue and rerun to resume."
            )
            raise

    print("\n--- LLM JUDGE COMPLETE ---")
    print(f"Cases judged: {len(results_df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()