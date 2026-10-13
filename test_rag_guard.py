from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.experts.base import run_expert


QUERY = "Explain Fourier Transform"
EXPERT = "notes_agent"


class FakeRetriever:
    def __init__(self, docs=None, error=None):
        self.docs = docs
        self.error = error
        self.invoke_count = 0

    def invoke(self, query):
        self.invoke_count += 1

        if self.error:
            raise self.error

        return self.docs


def run_test(name, retriever, expect_success=False):
    fake_response = MagicMock()
    fake_response.content = "Mocked answer"
    fake_response.agent_name = EXPERT
    fake_response.sources = []
    fake_response.confidence_score = 0.8
    fake_response.is_knowledge_present = True

    with (
        patch("src.experts.base.get_retriever", return_value=retriever),
        patch("src.experts.base.ChatOpenAI") as mock_chat,
    ):
        structured_llm = (
            mock_chat.return_value
            .with_structured_output.return_value
        )
        structured_llm.invoke.return_value = fake_response

        response, trace = run_expert(
            EXPERT,
            QUERY,
            mode="rag",
        )

    llm_calls = structured_llm.invoke.call_count

    print(f"\nTest: {name}")
    print(f"  Knowledge present: {response.is_knowledge_present}")
    print(f"  Confidence: {response.confidence_score}")
    print(f"  LLM calls: {llm_calls}")

    if expect_success:
        assert response.is_knowledge_present is True
        assert llm_calls == 1

        # Verify that retrieved evidence reaches the LLM.
        call_args = structured_llm.invoke.call_args
        messages = call_args.args[0]
        prompt_text = "\n".join(
            str(message.content) for message in messages
        )

        assert "Fourier transform converts" in prompt_text
        print("  Retrieved evidence included in prompt: YES")

    else:
        assert response.is_knowledge_present is False
        assert response.confidence_score == 0.0
        assert llm_calls == 0

    assert trace == []

    print(f"  PASS: {name}")


# 1. Retriever or index unavailable.
run_test(
    "Retriever unavailable",
    retriever=None,
)

# 2. Retrieval raises an exception.
run_test(
    "Retrieval exception",
    retriever=FakeRetriever(
        error=RuntimeError("SIMULATED_RETRIEVAL_FAILURE")
    ),
)

# 3. Retrieval returns no documents.
run_test(
    "Empty retrieval",
    retriever=FakeRetriever(docs=[]),
)

# 4. Documents exist, but their text is unusable.
run_test(
    "Unusable document text",
    retriever=FakeRetriever(
        docs=[
            Document(
                page_content="   ",
                metadata={"source": "empty.pdf"},
            )
        ]
    ),
)

# 5. Successful retrieval.
run_test(
    "Successful retrieval",
    retriever=FakeRetriever(
        docs=[
            Document(
                page_content=(
                    "Fourier transform converts a time-domain "
                    "signal into its frequency-domain representation."
                ),
                metadata={"source": "test_notes.pdf"},
            )
        ]
    ),
    expect_success=True,
)

print("\nAll RAG guard tests passed.")