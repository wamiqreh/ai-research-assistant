from typing import List
from pydantic import BaseModel, Field
from agents import Agent


HOW_MANY_SEARCHES = 1

INSTRUCTIONS = f"You are a helpful research assistant. Given a query, come up with a set of web searches \
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for. \
Use the prior clarfications and answers to guide your search planning. The questions and answers are in the same order"


class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")

class PlannerInput(BaseModel):
    answers: list[str] = Field(description="A list of answers to the questions provided by the user.")
    query: str = Field(description="The search user is searching for.")
    clarifications: list[str] = Field(description="A set of prior or related questions to inform planning.")


planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,
)


def build_planner_prompt(input_data: PlannerInput) -> str:
    prompt = f"Main query: {input_data.query}\n\n"
    
    for i, (q, a) in enumerate(zip(input_data.clarifications, input_data.answers), 1):
        prompt += f"Question {i}: {q}\nAnswer {i}: {a}\n"
    
    prompt += "\nBased on the above, create a WebSearchPlan in the required format."
    return prompt