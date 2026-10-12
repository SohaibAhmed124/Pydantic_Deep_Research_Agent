import asyncio
import os
import tempfile
from typing import Tuple
import gradio as gr
from dotenv import load_dotenv

from agent import get_model_name, quick_agent, run_deep_research
from telemetry import ResearchTelemetry

# Reload environment
load_dotenv(override=True)

ACTIVE_MODEL = get_model_name()


def check_api_key() -> Tuple[bool, str]:
    """Check if a valid OpenAI API key is configured."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "your_openai_api_key_here":
        return (
            False,
            "⚠️ **OpenAI API Key is missing!**\n\n"
            "Please open your `.env` file and set `OPENAI_API_KEY=your-actual-api-key`, "
            "then restart the application.",
        )
    return True, ""


# ==============================================================================
# Modern, Minimalist ChatGPT/Gemini-Style Design & Typography
# ==============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Font & Reset */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

code, pre, .terminal-window, .stat-value, .status-log {
    font-family: 'JetBrains Mono', Consolas, Monaco, monospace !important;
}

body, .gradio-container {
    background-color: #0b0e14 !important;
    color: #e2e8f0 !important;
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 1rem 1rem !important;
}

/* Minimalist Top App Bar */
.minimal-app-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 0.25rem 1rem 0.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    margin-bottom: 1rem;
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-title {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0;
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}

.badge-live {
    background: rgba(16, 185, 129, 0.12);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.badge-model {
    background: rgba(99, 102, 241, 0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.25);
}

/* Sleek Minimal Chat Window */
.chat-window {
    background: #0f131a !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px -2px rgba(0, 0, 0, 0.5) !important;
    overflow: hidden !important;
}

.chat-window .message-row {
    padding: 0.85rem 1rem !important;
}

.chat-window .message-bubble-border {
    border: none !important;
}

/* Agent Status Box in Chat */
.agent-status-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}

.status-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.95rem;
    font-weight: 600;
    color: #e0e7ff;
    margin-bottom: 0.4rem;
}

.pulse-indicator {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #6366f1;
    display: inline-block;
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7);
    animation: status-pulse 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
}

@keyframes status-pulse {
    to {
        box-shadow: 0 0 0 10px rgba(99, 102, 241, 0);
    }
}

.status-step {
    font-size: 0.88rem;
    color: #cbd5e1;
    margin-bottom: 0.4rem;
    font-weight: 500;
}

.status-log {
    font-size: 0.78rem;
    color: #94a3b8;
    background: rgba(15, 23, 42, 0.7);
    padding: 4px 10px;
    border-radius: 6px;
    display: inline-block;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Research Complete Badge */
.research-complete-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.28);
    color: #6ee7b7;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 1.25rem;
}

/* Chat Input Bar */
.chat-input-bar {
    margin-top: 0.85rem !important;
    gap: 8px !important;
}

.chat-input-bar textarea, .chat-input-bar input {
    background-color: #121620 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease !important;
}

.chat-input-bar textarea:focus, .chat-input-bar input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

.send-btn {
    background: #4f46e5 !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    transition: all 0.2s ease !important;
}

.send-btn:hover {
    background: #4338ca !important;
    transform: translateY(-1px) !important;
}

/* Telemetry Accordion Drawer */
.telemetry-drawer {
    margin-top: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    background: #0f131a !important;
    transition: all 0.25s ease !important;
    overflow: hidden !important;
}

.telemetry-drawer .label-wrap {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #94a3b8 !important;
    padding: 0.5rem 0.75rem !important;
}

.telemetry-drawer .label-wrap:hover {
    color: #e2e8f0 !important;
}

/* Metric Cards Grid Inside Accordion */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin: 0.5rem 0 1rem 0;
}

.stat-card {
    background: rgba(18, 24, 38, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    transition: all 0.2s ease;
}

.stat-icon {
    font-size: 1.1rem;
    margin-bottom: 0.2rem;
}

.stat-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
}

.stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-top: 2px;
}

.stat-sub {
    font-size: 0.68rem;
    color: #64748b;
    margin-top: 2px;
}

/* Monospace Live Console Box Inside Accordion */
.terminal-window {
    background: #07090e !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
    color: #93c5fd !important;
    max-height: 220px !important;
    overflow-y: auto !important;
    white-space: pre-wrap !important;
}

/* Markdown styling inside chat */
.chat-window h1, .chat-window h2, .chat-window h3 {
    color: #f8fafc !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07) !important;
    padding-bottom: 0.3rem !important;
    margin-top: 1.2rem !important;
}

.chat-window a {
    color: #818cf8 !important;
    text-decoration: underline !important;
    text-underline-offset: 2px !important;
}

.chat-window a:hover {
    color: #a5b4fc !important;
}

.chat-window table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 1rem 0 !important;
}

.chat-window th, .chat-window td {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 6px 12px !important;
    text-align: left !important;
}

.chat-window th {
    background-color: rgba(30, 41, 59, 0.5) !important;
}
"""


# ==============================================================================
# Chat Handlers
# ==============================================================================
async def chat_deep_research(user_message: str, chat_history: list, model_choice: str):
    """
    Handle deep research initiated in conversational chat window.
    Streams agent status live in chat bubble, then synthesizes full report.
    Yields: (cleared_input, chat_history, stat_cards_html, tool_audit_md, token_breakdown_md, terminal_logs, report_file)
    """
    if not user_message or not user_message.strip():
        blank_telemetry = ResearchTelemetry()
        yield (
            "",
            chat_history,
            blank_telemetry.format_stat_cards_html(),
            "*No tool audit recorded.*",
            "*No token breakdown recorded.*",
            "System ready.",
            None,
        )
        return

    is_valid_key, key_err = check_api_key()
    if not is_valid_key:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": f"❌ **Configuration Error:**\n\n{key_err}"})
        blank_telemetry = ResearchTelemetry()
        yield (
            "",
            chat_history,
            blank_telemetry.format_stat_cards_html(),
            "*No tool audit recorded.*",
            "*No token breakdown recorded.*",
            key_err,
            None,
        )
        return

    selected_model = model_choice or ACTIVE_MODEL
    current_telemetry = ResearchTelemetry(model_name=selected_model)
    current_telemetry.log("INFO", f"Queued query: '{user_message}' | Model: {selected_model}")

    # Add user message to history
    chat_history.append({"role": "user", "content": user_message})

    # Initial live assistant status
    initial_status = f"""
<div class="agent-status-card">
    <div class="status-header">
        <span class="pulse-indicator"></span>
        <span>Initializing Deep Research Agent...</span>
    </div>
    <div class="status-step">Analyzing intent and entity scope...</div>
    <div class="status-log">Active Model: {selected_model.replace('openai:', '')}</div>
</div>
"""
    chat_history.append({"role": "assistant", "content": initial_status})

    yield (
        "",
        chat_history,
        current_telemetry.format_stat_cards_html(),
        current_telemetry.format_tool_audit_markdown(),
        current_telemetry.format_token_breakdown_markdown(),
        current_telemetry.format_logs_terminal(),
        None,
    )

    current_progress_msg = "🔍 Analyzing intent and resolving entities..."

    def on_progress(msg: str, step_pct: float, tel: ResearchTelemetry):
        nonlocal current_telemetry, current_progress_msg
        current_telemetry = tel
        current_progress_msg = msg

    # Launch background research task
    research_task = asyncio.create_task(
        run_deep_research(
            user_input=user_message.strip(),
            model_name=selected_model,
            progress_callback=on_progress,
        )
    )

    # Stream real-time status into the chat bubble
    while not research_task.done():
        latest_log = current_telemetry.logs[-1] if current_telemetry.logs else "Executing pipeline..."
        status_html = f"""
<div class="agent-status-card">
    <div class="status-header">
        <span class="pulse-indicator"></span>
        <span>Deep Research in Progress...</span>
    </div>
    <div class="status-step">{current_progress_msg}</div>
    <div class="status-log">{latest_log}</div>
</div>
"""
        chat_history[-1]["content"] = status_html
        yield (
            "",
            chat_history,
            current_telemetry.format_stat_cards_html(),
            current_telemetry.format_tool_audit_markdown(),
            current_telemetry.format_token_breakdown_markdown(),
            current_telemetry.format_logs_terminal(),
            None,
        )
        await asyncio.sleep(0.35)

    try:
        final_report, completed_telemetry = await research_task

        # Prepare downloadable markdown file
        clean_title = "".join(c for c in user_message[:25] if c.isalnum() or c in ("-", "_")).strip() or "report"
        temp_dir = tempfile.gettempdir()
        report_filename = os.path.join(temp_dir, f"Deep_Research_{clean_title}.md")
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report)

        completion_badge = f"""
<div class="research-complete-badge">
    ✅ <b>Deep Research Completed</b> · <i>{completed_telemetry.elapsed_seconds}s</i> · <i>{completed_telemetry.total_tokens:,} tokens (${completed_telemetry.total_cost_usd:.4f})</i> · <i>{completed_telemetry.search_count} searches</i>
</div>
"""
        chat_history[-1]["content"] = f"{completion_badge}\n\n{final_report}"

        yield (
            "",
            chat_history,
            completed_telemetry.format_stat_cards_html(),
            completed_telemetry.format_tool_audit_markdown(),
            completed_telemetry.format_token_breakdown_markdown(),
            completed_telemetry.format_logs_terminal(),
            report_filename,
        )

    except Exception as e:
        err_msg = str(e)
        if "model" in err_msg.lower() and ("not found" in err_msg.lower() or "does not exist" in err_msg.lower()):
            err_msg += "\n\n💡 *Tip*: Model not supported on your OpenAI key. Switch to `openai:gpt-4o-mini` in Model Settings."
        current_telemetry.log("ERROR", f"Execution failed: {err_msg}")
        chat_history[-1]["content"] = f"❌ **Error during deep research:**\n\n{err_msg}"
        yield (
            "",
            chat_history,
            current_telemetry.format_stat_cards_html(),
            current_telemetry.format_tool_audit_markdown(),
            current_telemetry.format_token_breakdown_markdown(),
            current_telemetry.format_logs_terminal(),
            None,
        )


def clear_deep_research():
    return [], None


async def chat_quick_agent(user_message: str, chat_history: list, pydantic_history: list):
    """Fast, concise chat responses with streaming."""
    if not user_message or not user_message.strip():
        yield "", chat_history, pydantic_history
        return

    is_valid_key, key_err = check_api_key()
    if not is_valid_key:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": key_err})
        yield "", chat_history, pydantic_history
        return

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})
    yield "", chat_history, pydantic_history

    try:
        async with quick_agent.run_stream(
            user_message, message_history=pydantic_history
        ) as response:
            async for chunk in response.stream_text():
                chat_history[-1]["content"] = chunk
                yield "", chat_history, pydantic_history

            if hasattr(response, "all_messages"):
                pydantic_history = response.all_messages()
    except Exception as e:
        chat_history[-1]["content"] = f"❌ **Error:** {e}"
        yield "", chat_history, pydantic_history


def clear_quick_chat():
    return [], []


def update_active_model(model_name: str):
    global ACTIVE_MODEL, quick_agent
    cleaned = model_name.strip()
    if cleaned:
        ACTIVE_MODEL = cleaned
        from agent import Agent, CONCISE_SYSTEM_PROMPT
        quick_agent = Agent(model=ACTIVE_MODEL, system_prompt=CONCISE_SYSTEM_PROMPT)
        return f"✅ Active model updated to: `{ACTIVE_MODEL}`"
    return "⚠️ Model name cannot be empty."


# ==============================================================================
# UI Layout
# ==============================================================================
with gr.Blocks(title="Deep Research Agent") as demo:
    # Inject Custom Modern CSS
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")

    # Minimal App Bar
    gr.HTML(
        f"""
        <div class="minimal-app-bar">
            <div class="brand-wrapper">
                <span style="font-size: 1.4rem;">🧠</span>
                <div>
                    <h1 class="brand-title">Deep Research Agent</h1>
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="brand-badge badge-live">● Agent Ready</span>
                <span class="brand-badge badge-model">{ACTIVE_MODEL.replace('openai:', '')}</span>
            </div>
        </div>
        """
    )

    # Collapsible Model Configuration
    with gr.Accordion("⚙️ Model Settings", open=False):
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=[
                    "openai:gpt-6-luna",
                    "openai:gpt-4o-mini",
                    "openai:gpt-4o",
                    "openai:gpt-5.6-luna",
                ],
                value=ACTIVE_MODEL,
                label="Select AI Model",
                allow_custom_value=True,
                scale=4,
            )
            apply_model_btn = gr.Button("Update Model", variant="secondary", scale=1)
        model_status = gr.Markdown()
        apply_model_btn.click(update_active_model, inputs=[model_dropdown], outputs=[model_status])

    with gr.Tabs():
        # ======================================================================
        # Tab 1: Deep Research (Conversational LLM Chat)
        # ======================================================================
        with gr.Tab("🔬 Deep Research"):
            research_chatbot = gr.Chatbot(
                label="",
                elem_classes=["chat-window"],
                height=560,
                buttons=["copy"],
                render_markdown=True,
                placeholder="Ask any research topic or stock ticker (e.g. NVDA, Solid-State EV Batteries)...",
            )

            with gr.Row(elem_classes=["chat-input-bar"]):
                research_input = gr.Textbox(
                    placeholder="Ask any research topic or stock ticker (e.g. NVDA, Solid-State EV Batteries)...",
                    show_label=False,
                    lines=1,
                    max_lines=4,
                    scale=9,
                    autofocus=True,
                )
                research_send_btn = gr.Button("➔", variant="primary", scale=1, elem_classes=["send-btn"])

            with gr.Row():
                clear_research_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm", scale=2)
                download_file = gr.File(label="📥 Download Latest Report (.md)", interactive=False, scale=8)

            # Discreet Collapsible Slide-Out Telemetry Drawer
            with gr.Accordion("📊 Research Telemetry & Tool Audit (Click to expand)", open=False, elem_classes=["telemetry-drawer"]):
                initial_telemetry = ResearchTelemetry(model_name=ACTIVE_MODEL)
                metrics_display = gr.HTML(initial_telemetry.format_stat_cards_html())

                with gr.Tabs():
                    with gr.Tab("🛠️ Tool Audit & Web Scrapes"):
                        tool_audit_table = gr.Markdown("*No tool calls executed yet.*")
                    with gr.Tab("📊 Token & Cost Breakdown"):
                        token_breakdown_table = gr.Markdown("*No tokens consumed yet.*")
                    with gr.Tab("💻 Event Log"):
                        terminal_console = gr.Code(
                            value="[System Ready] Awaiting deep research request...",
                            language=None,
                            lines=8,
                            interactive=False,
                        )

            # Event Wireup for Deep Research Chat
            research_input.submit(
                chat_deep_research,
                inputs=[research_input, research_chatbot, model_dropdown],
                outputs=[
                    research_input,
                    research_chatbot,
                    metrics_display,
                    tool_audit_table,
                    token_breakdown_table,
                    terminal_console,
                    download_file,
                ],
            )
            research_send_btn.click(
                chat_deep_research,
                inputs=[research_input, research_chatbot, model_dropdown],
                outputs=[
                    research_input,
                    research_chatbot,
                    metrics_display,
                    tool_audit_table,
                    token_breakdown_table,
                    terminal_console,
                    download_file,
                ],
            )
            clear_research_btn.click(
                clear_deep_research,
                outputs=[research_chatbot, download_file],
            )

        # ======================================================================
        # Tab 2: Direct Concise Chat
        # ======================================================================
        with gr.Tab("💬 Direct Concise Chat"):
            pydantic_chat_state = gr.State([])
            quick_chatbot = gr.Chatbot(
                label="",
                elem_classes=["chat-window"],
                height=520,
                buttons=["copy"],
                placeholder="Ask a quick question for a direct, short answer...",
            )

            with gr.Row(elem_classes=["chat-input-bar"]):
                chat_input = gr.Textbox(
                    placeholder="Ask a quick question for a direct, short answer...",
                    show_label=False,
                    lines=1,
                    scale=9,
                )
                chat_send_btn = gr.Button("➔", variant="primary", scale=1, elem_classes=["send-btn"])

            clear_quick_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

            chat_input.submit(
                chat_quick_agent,
                inputs=[chat_input, quick_chatbot, pydantic_chat_state],
                outputs=[chat_input, quick_chatbot, pydantic_chat_state],
            )
            chat_send_btn.click(
                chat_quick_agent,
                inputs=[chat_input, quick_chatbot, pydantic_chat_state],
                outputs=[chat_input, quick_chatbot, pydantic_chat_state],
            )
            clear_quick_btn.click(
                clear_quick_chat,
                outputs=[quick_chatbot, pydantic_chat_state],
            )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
