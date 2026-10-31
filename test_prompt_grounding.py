"""Check that the shared expert prompt supports abstention instead of forbidding it."""
from src.prompts.loader import get_prompt

prompt = get_prompt("expert_system", expert_name="notes_agent", mode="rag")

FORBIDDEN = [
    "I don't have this in my notes",       # used to be banned, which blocked abstention
    "Answer confidently like a tutor who knows the subject deeply",
    "at least one worked example",          # forced invented examples
    "this is your most important instruction",
]
REQUIRED = [
    "GROUNDING",
    "set is_knowledge_present to false",
    "Do not fill the gap from general knowledge",
    "false assumption",
    "confidence_score",
]

for phrase in FORBIDDEN:
    assert phrase not in prompt, f"prompt still contains: {phrase!r}"
    print(f"  PASS: removed {phrase!r}")
for phrase in REQUIRED:
    assert phrase in prompt, f"prompt is missing: {phrase!r}"
    print(f"  PASS: contains {phrase!r}")

# The template must still format (unescaped braces would raise here).
assert "$z^{-1}$" in prompt
print("\nAll prompt grounding tests passed.")
