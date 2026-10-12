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
# CSS & Modern Styling
# ==============================================================================
CUSTOM_CSS = """
/* Global Container & Typography */
body, .gradio-container {
    background-color: #080c14 !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Header & Brand */
.brand-header {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.2) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
}

.brand-title {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.25rem 0;
}

.brand-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-model {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.badge-live {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Metric Cards Grid */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 1.25rem 0;
}

.stat-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.stat-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
}

.stat-icon {
    font-size: 1.3rem;
    margin-bottom: 0.25rem;
}

.stat-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f8fafc;
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    letter-spacing: -0.02em;
}

.stat-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-top: 2px;
}

.stat-sub {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}

/* Monospace Live Console Box */
.terminal-window {
    background: #040711 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
    color: #93c5fd !important;
    max-height: 240px !important;
    overflow-y: auto !important;
    white-space: pre-wrap !important;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
}

/* Form Controls & Inputs */
input, textarea {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

input:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* Report Formatting */
.report-content {
    background: #0b1120;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 2rem;
    line-height: 1.7;
    color: #e2e8f0;
}

.report-content h1, .report-content h2, .report-content h3 {
    color: #f8fafc;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.3rem;
}

.report-content a {
    color: #818cf8;
    text-decoration: none;
}

.report-content a:hover {
    text-decoration: underline;
}
"""


# ==============================================================================
# Deep Research Handler
# ==============================================================================
async def execute_deep_research(user_query: str, model_choice: str):
    """
    Execute deep research with live streaming of metrics, terminal logs, and final report.
    Yields: (metrics_html, terminal_logs, report_md, tool_audit_md, token_breakdown_md, file_download)
    """
    if not user_query or not user_query.strip():
        blank_telemetry = ResearchTelemetry()
        yield (
            blank_telemetry.format_stat_cards_html(),
            "⚠️ Please enter a stock ticker (e.g. NVDA) or a research topic.",
            "",
            "*No tool audit recorded.*",
            "*No token breakdown recorded.*",
            None,
        )
        return

    is_valid_key, key_err = check_api_key()
    if not is_valid_key:
        blank_telemetry = ResearchTelemetry()
        yield (
            blank_telemetry.format_stat_cards_html(),
            f"❌ Configuration Error:\n{key_err}",
            "",
            "*No tool audit recorded.*",
            "*No token breakdown recorded.*",
            None,
        )
        return

    selected_model = model_choice or ACTIVE_MODEL
    current_telemetry = ResearchTelemetry(model_name=selected_model)
    current_telemetry.log("INFO", f"Task queued: '{user_query}' | Engine: Pydantic AI | Model: {selected_model}")

    def on_progress(msg: str, step_pct: float, tel: ResearchTelemetry):
        nonlocal current_telemetry
        current_telemetry = tel

    # Launch background research task
    research_task = asyncio.create_task(
        run_deep_research(
            user_input=user_query.strip(),
            model_name=selected_model,
            progress_callback=on_progress,
        )
    )

    # Stream intermediate progress
    while not research_task.done():
        yield (
            current_telemetry.format_stat_cards_html(),
            current_telemetry.format_logs_terminal(),
            "*Generating comprehensive research report... Check live console above.*",
            current_telemetry.format_tool_audit_markdown(),
            current_telemetry.format_token_breakdown_markdown(),
            None,
        )
        await asyncio.sleep(0.4)

    try:
        final_report, completed_telemetry = await research_task

        # Save to markdown file for download
        clean_title = "".join(c for c in user_query[:25] if c.isalnum() or c in ("-", "_")).strip() or "report"
        temp_dir = tempfile.gettempdir()
        report_filename = os.path.join(temp_dir, f"Deep_Research_{clean_title}.md")
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report)

        yield (
            completed_telemetry.format_stat_cards_html(),
            completed_telemetry.format_logs_terminal(),
            final_report,
            completed_telemetry.format_tool_audit_markdown(),
            completed_telemetry.format_token_breakdown_markdown(),
            report_filename,
        )

    except Exception as e:
        err_msg = str(e)
        if "model" in err_msg.lower() and ("not found" in err_msg.lower() or "does not exist" in err_msg.lower()):
            err_msg += "\n\n💡 *Tip*: Model not supported on your key. Switch to `openai:gpt-4o-mini` in Model Settings."
        current_telemetry.log("ERROR", f"Execution failed: {err_msg}")
        yield (
            current_telemetry.format_stat_cards_html(),
            current_telemetry.format_logs_terminal(),
            f"❌ **Error during deep research:**\n\n{err_msg}",
            current_telemetry.format_tool_audit_markdown(),
            current_telemetry.format_token_breakdown_markdown(),
            None,
        )


# ==============================================================================
# Quick Chat Handler
# ==============================================================================
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


def clear_chat():
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
with gr.Blocks(title="Pydantic AI Deep Research Engine") as demo:
    # Inject Custom CSS
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")

    # Modern Header
    gr.HTML(
        f"""
        <div class="brand-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <h1 class="brand-title">🧠 Pydantic AI Deep Research Engine</h1>
                    <p class="brand-subtitle">
                        Multi-agent autonomous intelligence powered by <b>DuckDuckGo Web Search</b> & <b>Pydantic AI</b>
                    </p>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span class="brand-badge badge-live">● Live Telemetry</span>
                    <span class="brand-badge badge-model">{ACTIVE_MODEL.replace('openai:', '')}</span>
                </div>
            </div>
        </div>
        """
    )

    with gr.Accordion("⚙️ Engine & Model Configuration", open=False):
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=[
                    "openai:gpt-6-luna",
                    "openai:gpt-4o-mini",
                    "openai:gpt-4o",
                    "openai:gpt-5.6-luna",
                ],
                value=ACTIVE_MODEL,
                label="Active AI Model",
                allow_custom_value=True,
                scale=4,
            )
            apply_model_btn = gr.Button("Update Model", variant="secondary", scale=1)
        model_status = gr.Markdown()
        apply_model_btn.click(update_active_model, inputs=[model_dropdown], outputs=[model_status])

    with gr.Tabs():
        # ======================================================================
        # Tab 1: Deep Research Engine
        # ======================================================================
        with gr.Tab("🔬 Autonomous Deep Research"):
            with gr.Row():
                research_input = gr.Textbox(
                    placeholder="Enter stock ticker (e.g., NVDA, TSLA, AAPL) or subject (e.g., 'Solid-State Battery Breakthroughs 2026')...",
                    label="Research Query or Stock Ticker",
                    lines=1,
                    scale=8,
                    autofocus=True,
                )
                research_btn = gr.Button("🚀 Run Deep Research", variant="primary", scale=2)

            gr.Examples(
                examples=[
                    "NVDA",
                    "TSLA",
                    "AAPL",
                    "Commercial Viability of Solid-State EV Batteries in 2026",
                    "Quantum Computing Post-Quantum Cryptography Impact",
                ],
                inputs=research_input,
            )

            # Live Telemetry Stat Ribbon
            initial_telemetry = ResearchTelemetry(model_name=ACTIVE_MODEL)
            metrics_display = gr.HTML(initial_telemetry.format_stat_cards_html())

            # Real-Time Monospace Developer Console
            gr.Markdown("### 💻 Real-Time Execution Console")
            terminal_console = gr.Code(
                value="[System Ready] Awaiting user query to initiate autonomous research...",
                language=None,
                lines=8,
                interactive=False,
                label="Live Multi-Agent Event Stream",
            )

            # Structured Results Tabs
            with gr.Tabs():
                with gr.Tab("📄 Structured Research Report"):
                    with gr.Row():
                        download_btn = gr.File(label="📥 Download Generated Report (.md)", interactive=False)
                    report_markdown = gr.Markdown(
                        value="*Your synthesized report will be rendered here upon completion.*",
                        elem_classes=["report-content"],
                    )

                with gr.Tab("🛠️ Tool Audit & Web Scrapes"):
                    gr.Markdown("### Audit of All Executed Tool Calls (DuckDuckGo & Page Scrapes)")
                    tool_audit_table = gr.Markdown("*No tool calls executed yet.*")

                with gr.Tab("📊 Token & Cost Breakdown"):
                    gr.Markdown("### LLM Token Accounting & Estimated Cost by Pipeline Stage")
                    token_breakdown_table = gr.Markdown("*No tokens consumed yet.*")

            # Wire up execution
            research_btn.click(
                execute_deep_research,
                inputs=[research_input, model_dropdown],
                outputs=[
                    metrics_display,
                    terminal_console,
                    report_markdown,
                    tool_audit_table,
                    token_breakdown_table,
                    download_btn,
                ],
            )
            research_input.submit(
                execute_deep_research,
                inputs=[research_input, model_dropdown],
                outputs=[
                    metrics_display,
                    terminal_console,
                    report_markdown,
                    tool_audit_table,
                    token_breakdown_table,
                    download_btn,
                ],
            )

        # ======================================================================
        # Tab 2: Quick Concise Chat
        # ======================================================================
        with gr.Tab("💬 Direct Concise Chat"):
            pydantic_chat_state = gr.State([])
            quick_chatbot = gr.Chatbot(label="Direct Q&A", height=450)

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask a quick question for a direct, short answer...",
                    show_label=False,
                    lines=1,
                    scale=8,
                )
                chat_send_btn = gr.Button("Send ➔", variant="primary", scale=1)

            clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

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
            clear_chat_btn.click(
                clear_chat,
                outputs=[quick_chatbot, pydantic_chat_state],
            )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
