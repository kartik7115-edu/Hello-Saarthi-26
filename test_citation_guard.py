"""Tests that reported sources and confidence match what was actually retrieved.

Covers:
  * retrieved passages reach the model labelled with their real file names
  * citations to files that were never retrieved are dropped
  * invented snippets are replaced with real excerpts, page numbers reported as 0
  * an answer that says the knowledge is missing gets confidence 0 and no sources
  * planning mode passes all retrieved passages to the model, not just the first 1500 chars

All LLM calls are mocked; no API key is needed.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.experts.base import (
    MAX_TOOL_RESULT_CHARS,
    UNKNOWN_PAGE,
    format_documents,
    get_expert_tools,
    parse_labelled_evidence,
    run_expert,
)
from src.schemas.models import Citation, ExpertName, ExpertResponse

EXPERT = "notes_agent"
QUERY = "What is the region of convergence of the Z-transform?"

ROC_TEXT = (
    "The region of convergence (ROC) is the set of z for which the "
    "Z-transform sum converges. For a causal sequence the ROC lies "
    "outside the outermost pole."
)
POLE_TEXT = "The ROC cannot contain any poles of X(z)."


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


def make_response(sources, knowledge_present=True, confidence=0.9):
    return ExpertResponse(
        agent_name=ExpertName(EXPERT),
        content="Answer text",
        sources=sources,
        confidence_score=confidence,
        is_knowledge_present=knowledge_present,
    )


def run_rag(docs, model_response):
    """Run RAG mode with a fake retriever and a fixed model response."""
    with (
        patch("src.experts.base.get_retriever", return_value=FakeRetriever(docs)),
        patch("src.experts.base.ChatOpenAI") as mock_chat,
    ):
        structured_llm = mock_chat.return_value.with_structured_output.return_value
        structured_llm.invoke.return_value = model_response
        response, _ = run_expert(EXPERT, QUERY, mode="rag")
    prompt = "\n".join(str(m.content) for m in structured_llm.invoke.call_args.args[0])
    return response, prompt


def check(name, condition, detail=""):
    assert condition, f"{name} {detail}"
    print(f"  PASS: {name}")


DOCS = [
    Document(page_content=ROC_TEXT, metadata={"source": "knowledge_base/notes_agent/ZTransform_notes.pdf"}),
    Document(page_content=POLE_TEXT, metadata={"source": "Poles_and_Zeros.txt"}),
]


print("\nformat_documents / parse_labelled_evidence")
text = format_documents(DOCS + [
    Document(page_content="   ", metadata={"source": "blank.pdf"}),
    Document(page_content="No metadata here.", metadata={}),
])
check("labels use the file name, not the path", "[Source: ZTransform_notes.pdf]" in text)
check("blank documents are skipped", "blank.pdf" not in text)
check("missing source is labelled unknown", "[Source: unknown]" in text)
evidence = parse_labelled_evidence(text)
check("evidence parsed per source", evidence["Poles_and_Zeros.txt"] == POLE_TEXT, repr(evidence))


print("\nRAG: labels reach the model")
_, prompt = run_rag(DOCS, make_response([]))
check("source label in prompt", "[Source: ZTransform_notes.pdf]" in prompt)
check("citation instructions in prompt", "page_number" in prompt and "to 0" in prompt)


print("\nRAG: invented citations")
response, _ = run_rag(DOCS, make_response([
    Citation(source_file="ZTransform_notes.pdf", page_number=12,
             snippet="For a causal sequence the ROC lies outside the outermost pole."),
    Citation(source_file="Oppenheim_Chapter_10.pdf", page_number=640,
             snippet="The ROC is a ring in the z-plane."),
]))
files = [c.source_file for c in response.sources]
check("unretrieved file dropped", "Oppenheim_Chapter_10.pdf" not in files, str(files))
check("retrieved file kept", files == ["ZTransform_notes.pdf"], str(files))
check("page number reported as unknown", response.sources[0].page_number == UNKNOWN_PAGE)
check("verbatim snippet kept",
      response.sources[0].snippet == "For a causal sequence the ROC lies outside the outermost pole.")


print("\nRAG: invented snippet, path-style and duplicate citations")
response, _ = run_rag(DOCS, make_response([
    Citation(source_file="Poles_and_Zeros.txt", page_number=3, snippet="Poles are always inside the ROC."),
    Citation(source_file="notes\\POLES_AND_ZEROS.TXT", page_number=4, snippet="Poles are always inside the ROC."),
]))
check("case and path differences still match, duplicates removed", len(response.sources) == 1,
      str(response.sources))
check("invented snippet replaced with real text", response.sources[0].snippet == POLE_TEXT,
      response.sources[0].snippet)
check("canonical file name used", response.sources[0].source_file == "Poles_and_Zeros.txt")


print("\nRAG: answer says the knowledge is missing")
response, _ = run_rag(DOCS, make_response(
    [Citation(source_file="ZTransform_notes.pdf", page_number=1, snippet=ROC_TEXT)],
    knowledge_present=False, confidence=0.85,
))
check("confidence forced to 0", response.confidence_score == 0.0)
check("sources cleared", response.sources == [])


print("\nRAG: answer with knowledge keeps its confidence")
response, _ = run_rag(DOCS, make_response([], confidence=0.7))
check("confidence unchanged", response.confidence_score == 0.7)


print("\nPlanning: real search tool output is labelled and not truncated")
long_docs = [
    Document(page_content=f"Chunk {i}: " + ("x " * 480), metadata={"source": f"lecture_{i}.txt"})
    for i in range(5)
]
with patch("src.experts.base.get_retriever", return_value=FakeRetriever(long_docs)):
    tools = get_expert_tools(EXPERT)
search_tool = next(t for t in tools if "knowledge_base" in t.name)
check("tool keeps its name", search_tool.name == "search_notes_agent_knowledge_base")
tool_output = search_tool.invoke({"query": QUERY})
check("tool output is labelled", tool_output.startswith("[Source: lecture_0.txt]"))
check("5 chunks fit in the limit", len(tool_output) <= MAX_TOOL_RESULT_CHARS, str(len(tool_output)))

search_call = MagicMock()
search_call.content = "Searching."
search_call.tool_calls = [{"name": search_tool.name, "args": {"query": QUERY}, "id": "s1"}]
stop = MagicMock()
stop.content = "Done."
stop.tool_calls = []
fake_llm = MagicMock()
fake_llm.invoke.side_effect = [search_call, stop]

with (
    patch("src.experts.base.get_expert_tools", return_value=[search_tool]),
    patch("src.experts.base.ChatOpenAI") as mock_chat,
):
    mock_chat.return_value.bind_tools.return_value = fake_llm
    structured_llm = mock_chat.return_value.with_structured_output.return_value
    structured_llm.invoke.return_value = make_response([
        Citation(source_file="lecture_4.txt", page_number=9, snippet="made up"),
        Citation(source_file="lecture_99.txt", page_number=1, snippet="made up"),
    ])
    response, _ = run_expert(EXPERT, QUERY, mode="planning")

synthesis_prompt = "\n".join(str(m.content) for m in structured_llm.invoke.call_args.args[0])
check("last chunk reaches synthesis (no 1500-char cut)", "[Source: lecture_4.txt]" in synthesis_prompt)
check("citation instructions in synthesis prompt", "page_number" in synthesis_prompt)
files = [c.source_file for c in response.sources]
check("planning keeps retrieved citation, drops invented one", files == ["lecture_4.txt"], str(files))
check("planning snippet replaced with real text", response.sources[0].snippet.startswith("Chunk 4:"))

print("\nAll citation guard tests passed.")
