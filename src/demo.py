from agent import generate_agent_response


def main():
    print("=" * 60)
    print("Spotify Customer Support AI")
    print("=" * 60)

    customer_message = input("\nCustomer message:\n> ").strip()

    if not customer_message:
        print("Customer message cannot be empty.")
        return

    conversation_context = input(
        "\nConversation context (optional):\n> "
    ).strip()

    print("\nProcessing...\n")

    result = generate_agent_response(
        customer_message=customer_message,
        conversation_context=conversation_context,
        top_k=3,
    )

    print("=" * 60)
    print("AGENT RESULT")
    print("=" * 60)

    print(f"\nIntent: {result['intent']}")

    print("\nReply:")
    print(result["reply"])

    print(f"\nAction: {result['action']}")

    print(f"Reason: {result['reason']}")

    print("\n" + "=" * 60)
    print("RETRIEVED HISTORICAL CASES")
    print("=" * 60)

    for _, row in result["retrieved_cases"].iterrows():
        print("\n" + "-" * 60)
        print(f"Case: {row['case_id']}")
        print(f"Similarity: {row['similarity']:.3f}")
        print(f"Customer: {row['customer_message']}")
        print(f"Spotify reply: {row['historical_spotify_reply']}")


if __name__ == "__main__":
    main()
