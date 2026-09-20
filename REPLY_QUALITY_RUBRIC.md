# Reply Quality Evaluation Rubric

## Purpose

Evaluate whether the AI-generated support reply is useful, safe, relevant,
and grounded in the available conversation and historical support evidence.

A historical reply is NOT treated as the single correct answer.
Multiple valid replies may exist.

---

## Overall Score: 1–5

### 5 — Excellent
The reply is correct, directly relevant, well grounded in the available
evidence, appropriately actionable, concise, and professional.

### 4 — Good
The reply is generally correct and useful, with only a minor omission,
awkward wording, or small weakness.

### 3 — Acceptable / Mixed
The reply is partially useful but has a noticeable weakness, such as
missing an important detail, limited actionability, or weak grounding.

### 2 — Poor
The reply has a substantial problem: it is partially incorrect,
poorly grounded, not sufficiently relevant, or unlikely to help the
customer.

### 1 — Unacceptable
The reply is clearly incorrect, irrelevant, unsafe, or contains serious
unsupported claims or hallucinations.

---

## Evaluation Criteria

### 1. Correctness
Does the reply correctly address the customer's issue using the
current message and available conversation context?

### 2. Relevance
Does the reply directly respond to what the customer is asking or
experiencing?

### 3. Grounding
Is the reply supported by the current conversation and/or retrieved
historical support evidence?

The evaluator should penalize:
- invented policies
- invented refunds or credits
- invented account actions
- unsupported troubleshooting steps
- unsupported claims about Spotify features
- fabricated facts

### 4. Actionability
Does the reply provide an appropriate next step?

For cases where more information is needed, asking a useful clarifying
question is considered actionable.

A clarification is preferable to inventing an answer.

### 5. Communication Quality
Is the reply concise, clear, professional, and appropriate for customer
support?

---

## Important Evaluation Rule

Do NOT compare the generated reply against one historical Spotify reply
as if that reply were the ground-truth answer.

The historical examples are evidence of how similar cases were handled,
not a single required response.

---

## LLM-as-Judge Input

The judge should receive:

1. Current customer message
2. Current conversation context
3. Generated AI reply
4. Retrieved historical customer messages
5. Retrieved historical Spotify replies

The judge should NOT receive the human intent label.

This keeps reply-quality evaluation separate from intent-classification
evaluation.

---

## Human-vs-LLM Judge Agreement

A subset of approximately 40 replies will be independently rated by a
human and by the LLM judge.

Agreement will be reported using:

- Exact agreement
- Agreement within one score point
- Weighted Cohen's kappa

The purpose is to check whether the automated judge behaves reasonably
compared with human assessment.

---

## Limitations

The reply-quality score is not an absolute measure of whether a reply
is "correct."

Human judgement is subjective, especially for short follow-ups,
ambiguous messages, and cases where multiple support responses could
reasonably work.

The evaluation should therefore be interpreted together with:

- intent accuracy
- macro F1
- failure analysis
- human-vs-LLM judge agreement