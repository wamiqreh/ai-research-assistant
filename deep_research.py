import asyncio
import gradio as gr
from dotenv import load_dotenv
from research_manager import run_research

load_dotenv(override=True)

# Gradio Chatbot expects list of {"role": "user"|"assistant", "content": "..."}
def _history_to_messages(history: list[tuple[str, str]]) -> list[dict]:
    out = []
    for user, assistant in history:
        if user:
            out.append({"role": "user", "content": user})
        if assistant:
            out.append({"role": "assistant", "content": assistant})
    return out


def _messages_to_history(messages: list) -> list[tuple[str, str]]:
    """Accept Gradio chatbot value (list of dicts with role/content) and return list of (user, assistant) tuples."""
    if not messages:
        return []
    history = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "") or ""
            if not isinstance(content, str):
                content = str(content)
            if role == "user":
                history.append((content, ""))
            else:
                if history and history[-1][1] == "":
                    history[-1] = (history[-1][0], content)
                else:
                    history.append(("", content))
        else:
            # legacy (user, assistant) tuple
            history.append((m[0], m[1]) if len(m) >= 2 else (str(m[0]), ""))
    return history


async def chat_turn(
    message: str,
    history: list[tuple[str, str]],
    progress_holder: str,
):
    """Handle one user message: run manager agent, stream progress, then return reply and report.
    Yields (chatbot_messages, progress_text, report) so the UI can show progress in real time."""
    if not (message or "").strip():
        yield _history_to_messages(history), progress_holder, ""
        return

    progress_queue: asyncio.Queue[str] = asyncio.Queue()
    progress_lines: list[str] = []

    run_task = asyncio.create_task(
        run_research(message, history, progress_queue)
    )

    while not run_task.done():
        try:
            line = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
            progress_lines.append(line)
            progress_text = "\n".join(progress_lines)
            yield _history_to_messages(history), progress_text, ""
        except asyncio.TimeoutError:
            yield _history_to_messages(history), "\n".join(progress_lines), ""
            continue

    while not progress_queue.empty():
        try:
            progress_lines.append(progress_queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    reply, report_md = run_task.result()
    new_history = history + [(message, reply)]
    progress_text = "\n".join(progress_lines) if progress_lines else ""

    yield _history_to_messages(new_history), progress_text, report_md


# Single conversational UI: one place to type, no separate buttons
with gr.Blocks(title="Deep Research Assistant") as ui:
    gr.Markdown(
        "# Deep Research Assistant\n"
        "Describe what you want to research. I may ask a few short questions to focus the report, then I’ll run the research and show the report here."
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        height=320,
    )
    progress_box = gr.Textbox(
        label="Progress",
        lines=4,
        max_lines=6,
        interactive=False,
        placeholder="Progress will appear here while research runs…",
    )
    report_box = gr.Markdown(
        label="Report",
        value="",
        elem_id="report",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="e.g. Compare the top 3 project management tools for small teams",
            show_label=False,
            scale=9,
            container=False,
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    async def respond(message: str, history: list, progress: str):
        history_tuples = _messages_to_history(history)
        if not (message or "").strip():
            yield _history_to_messages(history_tuples), progress, ""
            return
        async for chat_messages, progress_text, report in chat_turn(
            message, history_tuples, progress
        ):
            yield chat_messages, progress_text, report

    def clear_msg():
        return ""

    submit_btn.click(
        respond,
        inputs=[msg, chatbot, progress_box],
        outputs=[chatbot, progress_box, report_box],
    ).then(clear_msg, outputs=[msg])
    msg.submit(
        respond,
        inputs=[msg, chatbot, progress_box],
        outputs=[chatbot, progress_box, report_box],
    ).then(clear_msg, outputs=[msg])


def main():
    ui.launch(inbrowser=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
