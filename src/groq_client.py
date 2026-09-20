import json
import os

from groq import Groq


MODEL = "openai/gpt-oss-20b"

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)


def generate_agent_output(prompt):
    """
    Generate a structured intent + reply response from Groq.
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
                "name": "support_agent_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string"
                        },
                        "reply": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "intent",
                        "reply"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response.")

    return json.loads(content)