from groq_client import generate_agent_output
from retriever import find_similar_cases
from escalation_policy import decide_escalation


INTENT_DEFINITIONS = """
PLAYBACK_OR_APP_ISSUE:
Problems playing music/audio, playback failures, crashes, buffering,
songs stopping, or the Spotify app not working during playback.

SUBSCRIPTION_OR_PAYMENT:
Premium subscription, billing, payment, pricing, trials, plans,
renewals, or subscription-related issues.

ACCOUNT:
Login, account access, account management, profile, or account-related
problems.

FEATURE_OR_UI:
Questions or problems about how Spotify works, features, settings,
interface, recommendations, or feature availability.

CONTENT_OR_CATALOG:
Questions/problems about songs, albums, artists, playlists, catalog
availability, missing content, or music availability.

PROMOTION_OR_EVENT:
Promotions, campaigns, discounts, competitions, or Spotify events.

PRIVACY_OR_DATA:
Privacy, personal data, data handling, or data-related requests.

SUPPORT_CHANNEL_OR_LANGUAGE:
Questions about contacting Spotify support, support availability,
support channels, or language support.

OTHER:
The message does not clearly fit another intent.
"""


def generate_agent_response(
    customer_message,
    conversation_context="",
    top_k=3,
    exclude_case_id=None,
    exclude_customer_id=None,
):
    retrieved_cases = find_similar_cases(
        customer_message=customer_message,
        top_k=top_k,
        exclude_case_id=exclude_case_id,
        exclude_customer_id=exclude_customer_id,
    )

    historical_examples = []

    for _, row in retrieved_cases.iterrows():
        historical_examples.append(
            f"""
Historical conversation context:
{row["context"]}

Historical customer message:
{row["customer_message"]}

Historical Spotify reply:
{row["historical_spotify_reply"]}
"""
        )

    historical_context = "\n---\n".join(historical_examples)

    prompt = f"""
You are a customer-support agent for Spotify.

Your task is to:

1. Classify the customer's current message into exactly ONE intent.
2. Draft a helpful customer-support reply grounded in how Spotify
   historically handled similar situations.

INTENT DEFINITIONS:
{INTENT_DEFINITIONS}

CURRENT CUSTOMER MESSAGE:
{customer_message}

CURRENT CONVERSATION CONTEXT:
{conversation_context}

HISTORICAL SPOTIFY SUPPORT EXAMPLES:
{historical_context}

GROUNDING RULES:

- Use the current conversation context when deciding the intent.
- Use historical examples as evidence for how Spotify handled similar
  problems.
- Base the reply primarily on the historical evidence.
- Do not mention the historical examples.
- Do not copy a historical reply word-for-word.
- Do not invent account-specific information.
- Do not claim that you performed an action.
- Do not invent refunds, credits, policies, or account changes.
- If historical examples show that Spotify asks for additional
  information, it is acceptable to ask for that information.
- Do not assume that a retrieved example is exactly the same as the
  customer's current problem.
- If the customer's message is a short acknowledgement or follow-up,
  use the current conversation context to understand it.
- If the intent cannot be determined confidently from the current
  message and context, use OTHER.
- Keep the reply concise, relevant, and professional.

Return only:
- intent
- reply
"""

    llm_result = generate_agent_output(prompt)

    action, reason = decide_escalation(
        intent=llm_result["intent"],
        customer_message=customer_message,
        conversation_context=conversation_context,
    )

    return {
        "intent": llm_result["intent"],
        "reply": llm_result["reply"],
        "action": action,
        "reason": reason,
        "retrieved_cases": retrieved_cases,
    }


if __name__ == "__main__":
    result = generate_agent_response(
        customer_message=(
            "My Spotify keeps stopping when I try to play music."
        ),
        conversation_context="",
        top_k=3,
    )

    print("\n--- AGENT RESULT ---")
    print("Intent:", result["intent"])
    print("Reply:", result["reply"])
    print("Action:", result["action"])
    print("Reason:", result["reason"])

    print("\n--- RETRIEVED CASES ---")

    for _, row in result["retrieved_cases"].iterrows():
        print(
            f"\nCase: {row['case_id']}"
            f"\nSimilarity: {row['similarity']:.3f}"
            f"\nCustomer: {row['customer_message']}"
            f"\nSpotify reply: {row['historical_spotify_reply']}"
        )