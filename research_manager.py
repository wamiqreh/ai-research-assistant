"""Research manager: an orchestrator agent that uses sub-agents as tools and handoffs."""

import asyncio
import json
from typing import Any

from agents import Agent, Runner, RunConfig, function_tool, handoff, RunContextWrapper, trace, gen_trace_id
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from search_agent import search_agent
from clarifier_agent import clarifier_agent
from planner_agent import (
    PlannerInput,
    build_planner_prompt,
    planner_agent,
    WebSearchItem,
    WebSearchPlan,
)
from writer_agent import writer_agent, ReportData
from email_agent import email_agent


class ResearchContext:
    """Context passed through the run; holds progress queue and trace_id for a single trace."""

    def __init__(
        self,
        progress_queue: asyncio.Queue[str] | None = None,
        trace_id: str | None = None,
    ):
        self.progress_queue = progress_queue
        self.trace_id = trace_id


def _emit_progress(ctx: RunContextWrapper[ResearchContext] | None, message: str) -> None:
    if ctx and ctx.context and ctx.context.progress_queue:
        try:
            ctx.context.progress_queue.put_nowait(message)
        except asyncio.QueueFull:
            pass


def _run_config(ctx: RunContextWrapper[ResearchContext] | None) -> RunConfig | None:
    """RunConfig so nested Runner.run calls use the same trace."""
    if ctx and ctx.context and ctx.context.trace_id:
        return RunConfig(trace_id=ctx.context.trace_id)
    return None


# --- Handoffs: progress on transfer ---


def _on_handoff_to_clarifier(ctx: RunContextWrapper[ResearchContext] | None) -> None:
    _emit_progress(ctx, "Asking a few questions to focus the research…")


def _on_handoff_to_email(ctx: RunContextWrapper[ResearchContext] | None) -> None:
    _emit_progress(ctx, "Sending email…")


# --- Tools that wrap sub-agents and report progress ---


@function_tool
async def plan_searches(
    ctx: RunContextWrapper[ResearchContext],
    query: str,
    questions_and_answers: str,
) -> str:
    """Create a search plan from the main query and the user's answers to clarifying questions. Input must be JSON: {"questions": [...], "answers": [...]}. Extract questions and answers from the conversation if the user was asked by the Clarifier agent."""
    _emit_progress(ctx, "Planning web searches...")
    data = json.loads(questions_and_answers)
    questions = data.get("questions", [])
    answers = data.get("answers", [])
    input_data = PlannerInput(query=query, clarifications=questions, answers=answers)
    prompt = build_planner_prompt(input_data)
    result = await Runner.run(planner_agent, prompt, run_config=_run_config(ctx))
    plan = result.final_output_as(WebSearchPlan)
    return plan.model_dump_json()


@function_tool
async def run_search(
    ctx: RunContextWrapper[ResearchContext],
    search_term: str,
    reason: str,
    progress_label: str | None = None,
) -> str:
    """Run one web search and return a short summary. Use progress_label to describe this step (e.g. 'Search 2/5')."""
    if progress_label:
        _emit_progress(ctx, progress_label)
    try:
        result = await Runner.run(
            search_agent,
            f"Search term: {search_term}\nReason for searching: {reason}",
            run_config=_run_config(ctx),
        )
        return str(result.final_output)
    except Exception:
        return ""


@function_tool
async def write_report(
    ctx: RunContextWrapper[ResearchContext],
    query: str,
    search_results_json: str,
) -> str:
    """Write the full research report from the original query and search result summaries. search_results_json is a JSON array of strings."""
    _emit_progress(ctx, "Writing report...")
    results = json.loads(search_results_json)
    input_text = f"Original query: {query}\nSummarized search results: {results}"
    result = await Runner.run(writer_agent, input_text, run_config=_run_config(ctx))
    report_data = result.final_output_as(ReportData)
    return report_data.model_dump_json()


MANAGER_INSTRUCTIONS = f"""You are a research coordinator. You help the user get a deep research report on a topic.

{RECOMMENDED_PROMPT_PREFIX}

Flow:
1. When the user first gives a research query, transfer to the Clarifier agent so they can ask 2–3 clarifying questions in a natural way. When the user has answered, you will get control back.
2. When you have the user's answers (from the conversation after the Clarifier handed back), use plan_searches with the original query and their answers. Pass JSON: {{"questions": [...], "answers": [...]}}. You can infer the questions from what the Clarifier asked, or use short placeholders.
3. For each search in the plan, call run_search with search_term, reason, and progress_label like "Search 2/5".
4. Call write_report with the query and all search result strings as a JSON array.
5. When the report is ready, transfer to the Email agent so they can send it. Include the full report (markdown_report from write_report) in your message when you transfer. When they hand back, reply to the user with the full report in markdown so they see it in the chat.

If the user has already provided answers to clarifying questions in the conversation, skip step 1 and go straight to planning and searching. Use the full conversation to extract the original query and the user's answers.
Always show the full report to the user at the end."""


research_manager_agent = Agent(
    name="ResearchManager",
    instructions=MANAGER_INSTRUCTIONS,
    model="gpt-4o-mini",
    tools=[
        plan_searches,
        run_search,
        write_report,
    ],
    handoffs=[
        handoff(
            agent=clarifier_agent,
            on_handoff=_on_handoff_to_clarifier,
        ),
        handoff(
            agent=email_agent,
            on_handoff=_on_handoff_to_email,
        ),
    ],
)

# Hand back to manager when clarifier or email agent are done
clarifier_agent.handoffs = [research_manager_agent]
email_agent.handoffs = [research_manager_agent]


async def run_research(
    user_message: str,
    conversation_history: list[tuple[str, str]],
    progress_queue: asyncio.Queue[str],
) -> tuple[str, str]:
    """
    Run the manager agent with the given message and history.
    Puts progress updates into progress_queue.
    Returns (assistant_reply, report_markdown).
    report_markdown is the final report if one was produced, else "".
    """
    trace_id = gen_trace_id()
    ctx = ResearchContext(progress_queue=progress_queue, trace_id=trace_id)

    # Build input: previous turns as history + new user message
    input_items: list[Any] = []
    for user, assistant in conversation_history:
        if user:
            input_items.append({"type": "message", "role": "user", "content": user})
        if assistant:
            input_items.append({"type": "message", "role": "assistant", "content": assistant})
    input_items.append({"type": "message", "role": "user", "content": user_message})

    # Single string input is simpler; multi-turn can use session later
    input_str = user_message
    if conversation_history:
        # Prepend recent context so the agent sees the thread
        recent = conversation_history[-5:]
        context = "\n\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in recent if u or a
        )
        input_str = f"[Previous context]\n{context}\n\n[Current message]\n{user_message}"

    with trace("Research trace", trace_id=trace_id):
        try:
            progress_queue.put_nowait(
                f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            )
        except asyncio.QueueFull:
            pass

        result = await Runner.run(
            research_manager_agent,
            input_str,
            context=ctx,
            max_turns=30,
            run_config=RunConfig(trace_id=trace_id),
        )

    final_output = result.final_output
    text = final_output if isinstance(final_output, str) else (final_output or "")
    # Final assistant reply is the report when research is complete
    return text, text
