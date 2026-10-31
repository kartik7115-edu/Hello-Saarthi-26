from unittest.mock import MagicMock, patch

from src.experts.base import run_expert


class FakeTool:
    """Simulate the knowledge-base tool interface."""

    name = "search_notes_agent_knowledge_base"

    def __init__(self, result=None, error=None, results=None):
        self.result = result
        self.error = error
        self.results = results
        self.invoke_count = 0

    def invoke(self, args):
        self.invoke_count += 1

        if self.error:
            raise self.error

        if self.results is not None:
            index = self.invoke_count - 1
            if index < len(self.results):
                return self.results[index]

        return self.result

def run_retry_test(name, tool_results, expected_success):
    fake_tool = FakeTool(results=tool_results)

    first_search = MagicMock()
    first_search.content = "Searching the knowledge base."
    first_search.tool_calls = [
        {
            "name": fake_tool.name,
            "args": {"query": "Explain Fourier Transform"},
            "id": "search-1",
        }
    ]

    second_search = MagicMock()
    second_search.content = "Retrying the knowledge search."
    second_search.tool_calls = [
        {
            "name": fake_tool.name,
            "args": {"query": "Fourier Transform frequency domain"},
            "id": "search-2",
        }
    ]

    stop_response = MagicMock()
    stop_response.content = "No further tool calls."
    stop_response.tool_calls = []

    fake_llm = MagicMock()

    # Each search request is followed by another model response.
    fake_llm.invoke.side_effect = [
        first_search,
        second_search,
        stop_response,
    ]

    fake_final_response = MagicMock()
    fake_final_response.content = "Mocked grounded answer"
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
    print(f"  Tool calls: {fake_tool.invoke_count}")
    print(f"  Synthesis calls: {synthesis_calls}")
    print(f"  Trace steps: {len(trace)}")

    assert fake_tool.invoke_count == len(tool_results)

    if expected_success:
        assert response.is_knowledge_present is True
        assert synthesis_calls == 1
    else:
        assert response.is_knowledge_present is False
        assert response.confidence_score == 0.0
        assert synthesis_calls == 0

    print(f"  PASS: {name}")

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
# 5. First search is empty; retry returns usable evidence.
run_retry_test(
    "Empty search followed by successful retry",
    tool_results=[
        [],
        (
            "The Fourier transform converts a time-domain signal "
            "into its frequency-domain representation."
        ),
    ],
    expected_success=True,
)

# 6. Both searches are empty; the response must be blocked.
run_retry_test(
    "Repeated empty searches",
    tool_results=[
        [],
        [],
    ],
    expected_success=False,
)

# 7. Real evidence that happens to contain "error:" must not be rejected.
run_test(
    "Evidence containing 'error:'",
    tool_result=(
        "Quantization error: the difference between the sampled value "
        "and its quantized representation."
    ),
    expect_success=True,
)

# 8. A failure message returned (not raised) by the tool must still block.
run_test(
    "Failure message returned as a result",
    tool_result="Error: The tool failed. No verified result was returned.",
)


def run_retry_hint_test():
    """After an empty search, the model is told to retry and the prompt allows it."""
    from langchain_core.messages import ToolMessage

    from src.experts.base import EMPTY_SEARCH_RETRY_HINT

    evidence = (
        "The Fourier transform converts a time-domain signal "
        "into its frequency-domain representation."
    )
    fake_tool = FakeTool(results=[[], evidence])

    def search(call_id):
        msg = MagicMock()
        msg.content = "Searching the knowledge base."
        msg.tool_calls = [{
            "name": fake_tool.name,
            "args": {"query": "Explain Fourier Transform"},
            "id": call_id,
        }]
        return msg

    stop_response = MagicMock()
    stop_response.content = "No further tool calls."
    stop_response.tool_calls = []

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [search("search-1"), search("search-2"), stop_response]

    fake_final_response = MagicMock()
    fake_final_response.is_knowledge_present = True
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = fake_final_response

    with (
        patch("src.experts.base.get_expert_tools", return_value=[fake_tool]),
        patch("src.experts.base.ChatOpenAI") as mock_chat,
    ):
        mock_chat.return_value.bind_tools.return_value = fake_llm
        mock_chat.return_value.with_structured_output.return_value = fake_structured_llm
        run_expert("notes_agent", "Explain Fourier Transform", mode="planning")

    # run_expert passes the same (growing) message list on every call.
    messages = fake_llm.invoke.call_args_list[-1][0][0]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    system_prompt = messages[0].content

    print("\nTest: Retry hint and retry-friendly prompt")
    print(f"  First tool message: {tool_messages[0].content[:60]}...")

    assert "different keywords" in system_prompt
    assert "only when the first search returned usable" not in system_prompt
    assert tool_messages[0].content == EMPTY_SEARCH_RETRY_HINT
    assert tool_messages[1].content.startswith("The Fourier transform")
    print("  PASS: Retry hint and retry-friendly prompt")


# 9. The empty search is followed by an explicit retry instruction.
run_retry_hint_test()

print("\nAll planning guard tests passed.")
