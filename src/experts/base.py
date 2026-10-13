from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, create_retriever_tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from typing import List, Dict, Any, Tuple
import logging

from src.prompts.loader import get_prompt
from src.tools.utilities import calculator
from src.tools.kb_ops import get_retriever
from src.schemas.models import ExpertResponse, ExpertName, Citation

logger = logging.getLogger(__name__)

MAX_REACT_STEPS = 5

def get_expert_tools(expert_name: str):
    """
    Returns the tools available to the expert.
    """
    tools = [calculator]
    retriever = get_retriever(expert_name)
    if retriever:
        retriever_tool = create_retriever_tool(
            retriever,
            f"search_{expert_name}_knowledge_base",
            f"Searches the knowledge base for {expert_name}. Use this when you need info from the stored documents."
        )
        tools.append(retriever_tool)
    else:
        @tool
        def search_knowledge_base(query: str) -> str:
            """Searches the expert's knowledge base."""
            return f"[No Index Found for {expert_name}. Please add PDFs to knowledge_base/{expert_name} and restart.]"
        tools.append(search_knowledge_base)

    return tools

def _build_history_messages(messages: List[Dict[str, Any]]) -> List:
    """Converts session message dicts to LangChain message objects (excludes last message)."""
    lc_messages = []
    for msg in messages[:-1]:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(SystemMessage(content=f"Assistant: {msg['content']}"))
    return lc_messages

def run_expert(expert_name: str, query: str, mode: str = "fast", messages: List[Dict[str, Any]] = []) -> Tuple[ExpertResponse, List[Dict]]:
    """
    Runs the expert agent and returns (ExpertResponse, react_trace).

    In planning (ReAct) mode, the agent loops: Think -> Act -> Observe
    up to MAX_REACT_STEPS times, collecting a visible trace of each step.

    react_trace format: [{"step": int, "thought": str, "tools_called": [...], "observations": [...]}]
    In fast mode, react_trace is an empty list.
    """
    logger.info(f"--- Running Expert: {expert_name} (Mode: {mode}) ---")

    tools = get_expert_tools(expert_name)
    prompt_str = get_prompt("expert_system", expert_name=expert_name, mode=mode)

    if expert_name == "notes_agent":
        prompt_str += "\n\nCRITICAL: You must answer ONLY from the provided context (citations). " \
                      "If the answer is NOT in the context, you MUST set 'is_knowledge_present' to False " \
                      "and return a generic 'I do not have this information in my notes' message."

    llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_tokens=3000)
    structured_llm = llm.with_structured_output(ExpertResponse, method="json_schema", strict=True)
    history = _build_history_messages(messages)

    if mode == "planning":
        react_prompt = (
            f"{prompt_str}\n\n"
            "You are operating in ReAct (Reasoning + Acting) mode.\n"
            "You MUST search the knowledge base before answering.\n"
            "Do not rely on your training knowledge as a substitute "
            "for the retrieved evidence.\n"
            "If retrieval fails or returns no usable evidence, do not "
            "attempt to answer from unsupported knowledge.\n\n"
            "Workflow:\n"
            "1. Search the knowledge base for the user's question.\n"
            "2. Review the retrieved evidence.\n"
            "3. Search again only when the first search returned usable "
            "evidence but more information is needed.\n"
            "4. Answer using evidence from the knowledge base.\n"
        )

        llm_with_tools = llm.bind_tools(tools)
        react_trace = []
        all_tool_outputs = []

        # Track knowledge retrieval separately from other tools.
        knowledge_search_attempted = False
        knowledge_search_succeeded = False
        knowledge_search_failed = False

        lc_messages = (
            [SystemMessage(content=react_prompt)]
            + history
            + [HumanMessage(content=query)]
        )

        def planning_failure(reason: str):
            logger.warning(
                "Planning response blocked for %s: %s",
                expert_name,
                reason,
            )
            response = ExpertResponse(
                agent_name=ExpertName(expert_name),
                content=(
                    "I couldn't verify an answer against the available "
                    "study materials because knowledge retrieval failed "
                    "or returned no usable evidence. Please try again later."
                ),
                sources=[],
                confidence_score=0.0,
                is_knowledge_present=False,
            )
            return response, react_trace

        def is_knowledge_tool(tool_name: str) -> bool:
            name = tool_name.lower()
            return (
                "knowledge_base" in name
                or name.startswith("search_")
            )

        def has_usable_evidence(result) -> bool:
            if result is None:
                return False

            if isinstance(result, (list, tuple)):
                if not result:
                    return False

                # LangChain retriever tools may return Document objects.
                if all(hasattr(item, "page_content") for item in result):
                    return any(
                        isinstance(item.page_content, str)
                        and item.page_content.strip()
                        for item in result
                    )

                result_text = "\n".join(str(item) for item in result)
            else:
                result_text = str(result)

            result_text = result_text.strip()
            if not result_text:
                return False

            result_lower = result_text.lower()

            failure_markers = (
                "[no index found",
                "tool not found",
                "error:",
                "retrieval failed",
                "knowledge base is unavailable",
                "no relevant information found",
                "no documents found",
                "no documents were found",
                "no usable evidence",
            )

            if result_lower in ("[]", "()", "none", "null"):
                return False

            if any(marker in result_lower for marker in failure_markers):
                return False

            return True

        for step in range(1, MAX_REACT_STEPS + 1):
            logger.info(
                "--- ReAct Step %s/%s for %s ---",
                step,
                MAX_REACT_STEPS,
                expert_name,
            )

            response = llm_with_tools.invoke(lc_messages)
            thought = response.content or ""

            step_record = {
                "step": step,
                "thought": thought,
                "tools_called": [],
                "observations": [],
            }

            if not response.tool_calls:
                step_record["thought"] = (
                    thought or "I have enough information to answer."
                )
                react_trace.append(step_record)
                break

            lc_messages.append(response)

            for tool_call in response.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                t_id = tool_call.get("id", t_name)

                knowledge_tool = is_knowledge_tool(t_name)

                if knowledge_tool:
                    knowledge_search_attempted = True

                matching_tool = next(
                    (t for t in tools if t.name == t_name),
                    None,
                )

                if matching_tool is None:
                    result = "Tool Not Found"
                    if knowledge_tool:
                        knowledge_search_failed = True
                else:
                    try:
                        result = (
                            matching_tool.invoke(t_args)
                            if hasattr(matching_tool, "invoke")
                            else matching_tool(t_args)
                        )

                        if knowledge_tool:
                            if has_usable_evidence(result):
                                knowledge_search_succeeded = True
                            else:
                                knowledge_search_failed = True

                    except Exception:
                        logger.exception(
                            "Tool %s failed for expert %s",
                            t_name,
                            expert_name,
                        )
                        result = (
                            "Error: The tool failed. "
                            "No verified result was returned."
                        )

                        if knowledge_tool:
                            knowledge_search_failed = True

                result_str = str(result)[:1500]

                step_record["tools_called"].append(
                    {"name": t_name, "args": t_args}
                )
                step_record["observations"].append(result_str[:300])

                all_tool_outputs.append(
                    f"[Step {step}] Tool `{t_name}`: {result_str}"
                )

                lc_messages.append(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=t_id,
                    )
                )

                # Do not waste additional LLM calls after a failed search.
                if knowledge_tool and not has_usable_evidence(result):
                    react_trace.append(step_record)
                    return planning_failure(
                        "Knowledge search failed or returned no usable evidence"
                    )

            react_trace.append(step_record)

        # Never synthesize a final answer without successful KB retrieval.
        if not knowledge_search_attempted:
            return planning_failure(
                "The knowledge search was never attempted"
            )

        if knowledge_search_failed or not knowledge_search_succeeded:
            return planning_failure(
                "Knowledge retrieval did not produce reliable evidence"
            )

        synthesis_context = "\n\n".join(all_tool_outputs)

        final_prompt = (
            f"The user asked: {query}\n\n"
            f"Tool outputs:\n{synthesis_context}\n\n"
            "Produce a comprehensive, well-structured answer using "
            "only facts supported by the successful knowledge-search "
            "results. Do not invent facts or citations. Cite available "
            "source filenames when present. If the retrieved evidence "
            "does not answer the question, set is_knowledge_present=False."
        )

        final_messages = (
            [
                SystemMessage(
                    content=(
                        f"{prompt_str}\n"
                        f"You must respond as {expert_name}.\n"
                        "Ground every factual claim in the retrieved "
                        "knowledge-base evidence."
                    )
                )
            ]
            + history
            + [HumanMessage(content=final_prompt)]
        )

        expert_response = structured_llm.invoke(final_messages)
        return expert_response, react_trace

    elif mode == "rag":
        # Retrieve evidence before allowing the LLM to answer.
        retriever = get_retriever(expert_name)

        def retrieval_failure(reason: str):
            logger.warning(
                "RAG response blocked for %s: %s",
                expert_name,
                reason,
            )
            response = ExpertResponse(
                agent_name=ExpertName(expert_name),
                content=(
                    "I couldn't verify an answer from the available "
                    "study materials because retrieval failed or returned "
                    "no usable evidence. Please try again later."
                ),
                sources=[],
                confidence_score=0.0,
                is_knowledge_present=False,
            )
            return response, []

        if retriever is None:
            return retrieval_failure("Retriever or index unavailable")

        try:
            docs = retriever.invoke(query)
        except Exception:
            logger.exception(
                "RAG retrieval failed for %s",
                expert_name,
            )
            return retrieval_failure("Retrieval raised an exception")

        if not docs:
            return retrieval_failure("No documents returned")

        # Keep only documents with usable text.
        usable_docs = [
            doc for doc in docs
            if isinstance(getattr(doc, "page_content", None), str)
            and doc.page_content.strip()
        ]

        if not usable_docs:
            return retrieval_failure("No usable document text returned")

        kb_context = "\n\n".join(
            doc.page_content for doc in usable_docs
        )

        augmented = (
            "Answer the user's question using only the evidence "
            "in the following retrieved excerpts.\n"
            "Do not invent facts or citations. If the excerpts do not "
            "support an answer, set is_knowledge_present to False and "
            "explain that the available materials do not contain enough "
            "information.\n\n"
            f"Retrieved excerpts:\n{kb_context}\n\n"
            f"User question: {query}"
        )

        lc_messages = (
            [
                SystemMessage(
                    content=(
                        f"{prompt_str}\n"
                        f"You must respond as {expert_name}.\n"
                        "GROUNDING REQUIREMENT: Use only facts supported "
                        "by the retrieved excerpts. Do not claim that a "
                        "fact is supported when the excerpts do not "
                        "establish it."
                    )
                )
            ]
            + history
            + [HumanMessage(content=augmented)]
        )

        return structured_llm.invoke(lc_messages), []

    else:
        # Fast mode: Direct Structured LLM call (no ReAct, no retrieval)
        lc_messages = (
            [SystemMessage(content=f"{prompt_str}\nYou must respond as {expert_name}.")]
            + history
            + [HumanMessage(content=query)]
        )
        return structured_llm.invoke(lc_messages), []

from src.utils.state import AgentState

def notes_agent_node(state: AgentState):
    query = state["sub_queries"][0].query if state["sub_queries"] else state["query"]
    mode = "planning" if state.get("planning", False) else "rag"
    res, trace = run_expert("notes_agent", query, mode=mode, messages=state.get("messages", []))
    return {"results": {"notes_agent": res, "notes_agent_trace": trace}}

def books_agent_node(state: AgentState):
    query = state["sub_queries"][0].query if state["sub_queries"] else state["query"]
    mode = "planning" if state.get("planning", False) else "rag"
    res, trace = run_expert("books_agent", query, mode=mode, messages=state.get("messages", []))
    return {"results": {"books_agent": res, "books_agent_trace": trace}}
