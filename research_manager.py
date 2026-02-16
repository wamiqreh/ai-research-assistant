"""Research manager: an orchestrator agent that uses sub-agents as tools and handoffs."""

import asyncio
import json
from typing import Any

from agents import Agent, Runner, function_tool, RunContextWrapper, trace, gen_trace_id

from search_agent import search_agent
from clarifier_agent import clarifier_agent, Clarifications
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
    """Context passed through the run; holds progress queue for UI updates."""

    def __init__(self, progress_queue: asyncio.Queue[str] | None = None):
        self.progress_queue = progress_queue


def _emit_progress(ctx: RunContextWrapper[ResearchContext] | None, message: str) -> None:
    if ctx and ctx.context and ctx.context.progress_queue:
        try:
            ctx.context.progress_queue.put_nowait(message)
        except asyncio.QueueFull:
            pass


# --- Tools that wrap sub-agents and report progress ---


@function_tool
async def get_clarifications(
    ctx: RunContextWrapper[ResearchContext],
    query: str,
) -> str:
    """Get 2–3 clarifying questions to ask the user for this research query. Call this first when the user has not yet answered any clarifying questions."""
    _emit_progress(ctx, "Thinking of a few questions to better focus the research...")
    result = await Runner.run(clarifier_agent, query)
    clarifications = result.final_output_as(Clarifications)
    return json.dumps({"questions": clarifications.questions})


@function_tool
async def plan_searches(
    ctx: RunContextWrapper[ResearchContext],
    query: str,
    questions_and_answers: str,
) -> str:
    """Create a search plan from the main query and the user's answers to clarifying questions. Input must be JSON: {\"questions\": [...], \"answers\": [...]}."""
    _emit_progress(ctx, "Planning web searches...")
    data = json.loads(questions_and_answers)
    questions = data.get("questions", [])
    answers = data.get("answers", [])
    input_data = PlannerInput(query=query, clarifications=questions, answers=answers)
    prompt = build_planner_prompt(input_data)
    result = await Runner.run(planner_agent, prompt)
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
    result = await Runner.run(writer_agent, input_text)
    report_data = result.final_output_as(ReportData)
    return report_data.model_dump_json()


@function_tool
async def send_report_email(
    ctx: RunContextWrapper[ResearchContext],
    markdown_report: str,
) -> str:
    """Send the final report by email. Call this after write_report."""
    _emit_progress(ctx, "Sending email...")
    await Runner.run(email_agent, markdown_report)
    return "Email sent."


MANAGER_INSTRUCTIONS = """You are a research coordinator. You help the user get a deep research report on a topic.

Flow:
1. When the user first gives a research query, use get_clarifications to get 2–3 short questions. Then ask the user those questions in a natural, conversational way (one message). Do not list "Question 1", "Question 2" mechanically—phrase them as a friendly assistant would.
2. When the user has answered (in the same chat), use plan_searches with the original query and their answers (pass questions and answers as JSON: {"questions": [...], "answers": [...]}).
3. For each search in the plan, call run_search with search_term, reason, and progress_label like "Search 2/5".
4. Call write_report with the query and all search result strings as a JSON array.
5. Call send_report_email with the markdown_report from the write_report output.
6. Reply to the user with the full report in markdown (the markdown_report from write_report). That is your final response.

If the user has already provided answers to your clarifying questions in the conversation, skip step 1 and go straight to planning and searching. Use the full conversation to extract the original query and the user's answers.
Always return the full report at the end so the user sees it in the chat."""


research_manager_agent = Agent(
    name="ResearchManager",
    instructions=MANAGER_INSTRUCTIONS,
    model="gpt-4o-mini",
    tools=[
        get_clarifications,
        plan_searches,
        run_search,
        write_report,
        send_report_email,
    ],
)


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
    ctx = ResearchContext(progress_queue=progress_queue)

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
        )

    final_output = result.final_output
    text = final_output if isinstance(final_output, str) else (final_output or "")
    # Final assistant reply is the report when research is complete
    return text, text
