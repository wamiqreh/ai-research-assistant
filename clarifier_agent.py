from pydantic import BaseModel, Field
from agents import Agent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# Used when clarifier is invoked as a tool (e.g. get_clarifications) for structured output
class Clarifications(BaseModel):
    questions: list[str] = Field(
        description="A list of questions to ask the user to provide more information to help you research the query."
    )

# Conversational clarifier for handoff: asks questions in natural language, then hands back to ResearchManager
CLARIFIER_HANDOFF_INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}

You are a helpful research assistant. The user has asked for a research report.
Your job is to ask 2–3 short clarifying questions in a friendly, natural way so the research can be better focused.
Ask your questions in one message—do not list "Question 1", "Question 2" mechanically.
When the user has answered your questions, hand back to the Research Manager so they can continue with the research."""

clarifier_agent = Agent(
    name="ClarifierAgent",
    instructions=CLARIFIER_HANDOFF_INSTRUCTIONS,
    model="gpt-4o-mini",
    handoff_description="Transfer here when you need to ask the user 2–3 clarifying questions before planning the research. They will answer in chat; then you get control back.",
)