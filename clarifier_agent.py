import string
from pydantic import BaseModel, Field
from agents import Agent

INSTRUCTIONS = f"You are a helpful research assistant. Given a query, come up with a set of 3 clarifications to ask the user to provide more information to help you research the query."

class Clarifications(BaseModel):
    questions: list[str] = Field(description="A list of questions to ask the user to provide more information to help you research the query.")

clarifier_agent = Agent(
    name="ClarifierAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=Clarifications,
)