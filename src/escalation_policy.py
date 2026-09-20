def decide_escalation(
    intent,
    customer_message,
    conversation_context="",
):
    """
    Deterministic escalation policy.

    Returns:
        action: AUTO_HANDLE or ESCALATE
        reason: explanation for the decision
    """

    message = str(customer_message).lower().strip()
    context = str(conversation_context).lower().strip()

    # 1. Privacy and personal-data issues go to a human.
    if intent == "PRIVACY_OR_DATA":
        return (
            "ESCALATE",
            "Privacy or personal-data issues may require human review."
        )

    # 2. Account-specific actions that the agent cannot perform safely.
    account_action_terms = [
        "change my email",
        "change my password",
        "delete my account",
        "close my account",
        "refund",
        "chargeback",
        "cancel my subscription",
        "restore my account",
        "recover my account",
    ]

    if intent == "ACCOUNT" and any(
        term in message for term in account_action_terms
    ):
        return (
            "ESCALATE",
            "The request may require account-specific action."
        )

    # 3. An unclear intent should be reviewed by a human.
    if intent == "OTHER":
        return (
            "ESCALATE",
            "The customer's intent could not be classified confidently."
        )

    # 4. Otherwise, allow automated handling.
    return (
        "AUTO_HANDLE",
        "The intent is sufficiently clear for automated handling."
    )