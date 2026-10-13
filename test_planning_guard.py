from unittest.mock import MagicMock, patch

from src.experts.base import run_expert


class FakeTool:
    """Simulate the knowledge-base tool interface."""

    name = "search_notes_agent_knowledge_base"

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.invoke_count = 0

    def invoke(self, args):
        self.invoke_count += 1

        if self.error:
            raise self.error

        return self.result


def run_test(name, tool_result=None, tool_error=None, expect_success=False):
    fake_tool = FakeTool(result=tool_result, error=tool_error)

    # The model requests a knowledge-base search on its first call.
    tool_call_response = MagicMock()
    tool_call_response.content = "Searching the knowledge base."
    tool_call_response.tool_calls = [
        {
            "name": fake_tool.name,
            "args": {"query": "Explain Fourier Transform"},
            "id": "test-call-1",
        }
    ]

    fake_llm = MagicMock()

    # First call: request knowledge retrieval.
    # Second call: stop requesting tools so synthesis can begin.
    final_react_response = MagicMock()
    final_react_response.content = "I have enough information to answer."
    final_react_response.tool_calls = []

    fake_llm.invoke.side_effect = [
        tool_call_response,
        final_react_response,
    ]

    fake_final_response = MagicMock()
    fake_final_response.content = "Mocked answer"
    fake_final_response.agent_name = "notes_agent"
    fake_final_response.sources = []
    fake_final_response.confidence_score = 0.8
    fake_final_response.is_knowledge_present = True

    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = fake_final_response

    with (
        patch(
            "src.experts.base.get_expert_tools",
            return_value=[fake_tool],
        ),
        patch(
            "src.experts.base.ChatOpenAI",
        ) as mock_chat,
    ):
        mock_chat.return_value.bind_tools.return_value = fake_llm
        mock_chat.return_value.with_structured_output.return_value = (
            fake_structured_llm
        )

        response, trace = run_expert(
            "notes_agent",
            "Explain Fourier Transform",
            mode="planning",
        )

    synthesis_calls = fake_structured_llm.invoke.call_count

    print(f"\nTest: {name}")
    print(f"  Knowledge present: {response.is_knowledge_present}")
    print(f"  Confidence: {response.confidence_score}")
    print(f"  Tool calls: {fake_tool.invoke_count}")
    print(f"  Synthesis calls: {synthesis_calls}")
    print(f"  Trace steps: {len(trace)}")

    if expect_success:
        assert response.is_knowledge_present is True
        assert synthesis_calls == 1
    else:
        assert response.is_knowledge_present is False
        assert response.confidence_score == 0.0
        assert synthesis_calls == 0

    assert fake_tool.invoke_count == 1

    print(f"  PASS: {name}")


# 1. Retrieval raises an exception.
run_test(
    "Retrieval exception",
    tool_error=RuntimeError("SIMULATED_RETRIEVAL_FAILURE"),
)

# 2. Knowledge base is unavailable.
run_test(
    "Index unavailable",
    tool_result="[No Index Found for notes_agent. Please add PDFs.]",
)

# 3. Retrieval returns no documents.
run_test(
    "Empty retrieval",
    tool_result=[],
)

# 4. Retrieval returns usable content.
run_test(
    "Successful retrieval",
    tool_result=(
        "The Fourier transform converts a time-domain signal "
        "into its frequency-domain representation."
    ),
    expect_success=True,
)

print("\nAll planning guard tests passed.")