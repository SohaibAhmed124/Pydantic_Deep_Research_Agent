# 🔬 Pydantic Deep Research Agent

An autonomous, multi-step deep research agent built with **Pydantic AI** and powered by **OpenAI** and **DuckDuckGo**, packaged in a modern, interactive **Gradio** web interface.

The agent investigates any given topic or stock ticker, autonomously formulates research angles, performs targeted web searches, scrapes relevant evidence, and synthesizes structured, authoritative reports with primary source citations and live telemetry.

---

## ✨ Features

- **Autonomous Research Pipeline**: Deconstructs high-level queries or stock tickers into targeted, non-overlapping research angles.
- **Concurrent Web Discovery & Scraping**: Uses DuckDuckGo search with rate-limited, asynchronous page extraction.
- **Evidence-Based Synthesis**: Synthesizes findings with structured citations, primary source verification, and risk analysis.
- **Live Execution Console**: Real-time developer log streaming search queries, scrape events, and agent milestones.
- **Token & Cost Accounting**: Tracks prompt and completion tokens with dynamic cost estimation based on the active model.
- **Granular Tool Audit**: Inspects every tool call, search query, HTTP status, and scraping latency.
- **Modern Gradio Web UI**: Includes a custom obsidian dark theme, one-click Markdown report export, and a quick concise chat mode.

---

## 🛠️ Technologies Used

- **[Pydantic AI](https://ai.pydantic.dev/)** — Type-safe, structured agentic orchestration and tool calling.
- **[OpenAI](https://github.com/openai/openai-python)** — Language model backend for intent resolution, research planning, and synthesis.
- **[Gradio](https://www.gradio.app/)** — Interactive web UI with real-time streaming, telemetry metrics, and multi-tab controls.
- **[DuckDuckGo Search (`ddgs`)](https://github.com/deedy5/duckduckgo_search)** — Unofficial async DuckDuckGo search integration.
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) & [Markdownify](https://github.com/matthewwithanm/python-markdownify)** — HTML parsing, cleaning, and conversion to Markdown for LLM consumption.
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** — Environment variable configuration.

---

## 🚀 How to Use

### 1. Prerequisites & Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/<YOUR-USERNAME>/Pydantic_Deep_Research_Agent.git
cd Pydantic_Deep_Research_Agent

# (Optional) Create and activate a virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root (see `.env.example`):

```ini
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=openai:gpt-4o
```

> **Note**: You can use `openai:gpt-4o`, `openai:gpt-4o-mini`, or any supported OpenAI model. The Gradio UI also provides a model selector dropdown.

### 3. Run the Application

Launch the Gradio web dashboard:

```bash
python app.py
```

Once running, open your browser at:
```text
http://127.0.0.1:7860
```

### 4. Running a Research Session

1. Go to the **🔬 Autonomous Deep Research** tab.
2. Enter a stock ticker (e.g. `NVDA`, `TSLA`) or any topic (e.g. `Commercial Viability of Solid-State EV Batteries in 2026`).
3. Click **🚀 Run Deep Research**.
4. Monitor the live terminal logs and telemetry stats in real-time.
5. Review the comprehensive report and download it as a `.md` file using **📥 Download Generated Report**.
