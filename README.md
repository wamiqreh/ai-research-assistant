# AI Research Assistant

A deep research assistant that uses AI agents to plan web searches, gather information, write detailed reports, and send results via email.

## Features

- **Planner Agent**: Creates a set of web searches to best answer your research query
- **Search Agent**: Performs web searches and produces concise summaries
- **Writer Agent**: Synthesizes search results into a cohesive, detailed report (5-10 pages)
- **Email Agent**: Sends the report as a nicely formatted HTML email

Built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/), Gradio for the UI, and SendGrid for email delivery.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

### 1. Clone and install dependencies with uv

```bash
# Install uv if you haven't already
# Windows (PowerShell):
irm https://astral.sh/uv/install.ps1 | iex

# Install project dependencies
uv sync
```

### 2. Configure environment variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (required) |
| `SENDGRID_API_KEY` | SendGrid API key for email delivery (required for email feature) |

For email, update the sender and recipient in `email_agent.py` to your verified SendGrid sender and desired recipient.

## Running the project

### With uv (recommended)

```bash
uv run ai-research-assistant
```

Or run the module directly:

```bash
uv run python deep_research.py
```

### With pip

```bash
pip install -r requirements.txt
python deep_research.py
```

The Gradio UI will open in your browser. Enter a research topic, click **Run**, and the assistant will:

1. Plan relevant web searches
2. Execute the searches and collect results
3. Write a detailed report
4. Send the report via email

View traces of the research process at [platform.openai.com](https://platform.openai.com/traces/).

## Project Structure

```
├── deep_research.py      # Gradio UI entry point
├── research_manager.py   # Orchestrates the research workflow
├── planner_agent.py      # Plans web searches for the query
├── search_agent.py       # Performs web searches and summarizes
├── writer_agent.py       # Writes the final report
├── email_agent.py        # Sends report via SendGrid
└── pyproject.toml        # Project config and dependencies
```

## License

MIT
