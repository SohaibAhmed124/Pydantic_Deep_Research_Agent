import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5.6-luna": (0.50, 2.00),
    "gpt-6-luna": (0.50, 2.00),
    "default": (0.50, 2.00),
}


def get_pricing_for_model(model_name: str) -> tuple[float, float]:
    """Return (input_price_per_1M, output_price_per_1M) for the model."""
    name_lower = model_name.lower()
    for key, (in_p, out_p) in MODEL_PRICING_PER_1M.items():
        if key in name_lower:
            return in_p, out_p
    return MODEL_PRICING_PER_1M["default"]


@dataclass
class ToolRecord:
    timestamp: str
    tool_type: str  # "DuckDuckGo Search" | "Web Scraper"
    target: str     # query or URL
    status: str     # "SUCCESS" | "FAILED" | "SNIPPET_FALLBACK"
    duration_ms: float
    details: str


@dataclass
class StageUsage:
    stage_name: str
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float


@dataclass
class ResearchTelemetry:
    model_name: str = "openai:gpt-6-luna"
    logs: List[str] = field(default_factory=list)
    tool_calls: List[ToolRecord] = field(default_factory=list)
    stage_usages: List[StageUsage] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def log(self, level: str, message: str):
        """Append a timestamped log entry."""
        now = datetime.datetime.now().strftime("%H:%M:%S")
        log_line = f"[{now}] [{level.upper()}] {message}"
        self.logs.append(log_line)

    def record_tool(
        self,
        tool_type: str,
        target: str,
        status: str,
        duration_ms: float,
        details: str = "",
    ):
        """Record an executed tool call."""
        now = datetime.datetime.now().strftime("%H:%M:%S")
        record = ToolRecord(
            timestamp=now,
            tool_type=tool_type,
            target=target,
            status=status,
            duration_ms=round(duration_ms, 1),
            details=details,
        )
        self.tool_calls.append(record)

    def record_stage_usage(self, stage_name: str, usage_obj: Any, model_name: Optional[str] = None):
        """Extract and record tokens and cost from Pydantic AI usage object."""
        model = model_name or self.model_name
        in_p, out_p = get_pricing_for_model(model)

        reqs = getattr(usage_obj, "requests", 1) if usage_obj else 1
        in_tok = getattr(usage_obj, "input_tokens", getattr(usage_obj, "request_tokens", 0)) if usage_obj else 0
        out_tok = getattr(usage_obj, "output_tokens", getattr(usage_obj, "response_tokens", 0)) if usage_obj else 0
        tot_tok = getattr(usage_obj, "total_tokens", in_tok + out_tok) if usage_obj else (in_tok + out_tok)

        cost = (in_tok * (in_p / 1_000_000.0)) + (out_tok * (out_p / 1_000_000.0))

        stage = StageUsage(
            stage_name=stage_name,
            requests=reqs,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            cost_usd=cost,
        )
        self.stage_usages.append(stage)

    def finish(self):
        """Mark end time."""
        if not self.end_time:
            self.end_time = time.time()

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.stage_usages)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.stage_usages)

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.stage_usages)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.stage_usages)

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 1)

    @property
    def search_count(self) -> int:
        return sum(1 for t in self.tool_calls if t.tool_type == "DuckDuckGo Search")

    @property
    def scraped_count(self) -> int:
        return sum(1 for t in self.tool_calls if t.tool_type == "Web Scraper" and t.status == "SUCCESS")

    def format_stat_cards_html(self) -> str:
        """Render modern CSS metric cards."""
        tot_tok = f"{self.total_tokens:,}"
        cost_str = f"${self.total_cost_usd:.4f}"
        searches = str(self.search_count)
        scraped = str(self.scraped_count)
        elapsed = f"{self.elapsed_seconds}s"

        return f"""
        <div class="telemetry-grid">
            <div class="stat-card">
                <div class="stat-icon">🪙</div>
                <div class="stat-value">{tot_tok}</div>
                <div class="stat-label">Total Tokens</div>
                <div class="stat-sub">{self.total_input_tokens:,} in / {self.total_output_tokens:,} out</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">💵</div>
                <div class="stat-value">{cost_str}</div>
                <div class="stat-label">Estimated Cost</div>
                <div class="stat-sub">{self.model_name.replace('openai:', '')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🔍</div>
                <div class="stat-value">{searches}</div>
                <div class="stat-label">DDG Searches</div>
                <div class="stat-sub">Targeted queries</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📄</div>
                <div class="stat-value">{scraped}</div>
                <div class="stat-label">Pages Scraped</div>
                <div class="stat-sub">Markdown parsed</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⏱️</div>
                <div class="stat-value">{elapsed}</div>
                <div class="stat-label">Total Duration</div>
                <div class="stat-sub">End-to-end time</div>
            </div>
        </div>
        """

    def format_logs_terminal(self) -> str:
        """Return the raw terminal log string."""
        if not self.logs:
            return "System ready. Awaiting research request..."
        return "\n".join(self.logs)

    def format_tool_audit_markdown(self) -> str:
        """Render a clean Markdown table of executed tool calls."""
        if not self.tool_calls:
            return "*No tool calls recorded yet.*"

        rows = []
        rows.append("| Time | Tool | Target / Query | Status | Latency | Details |")
        rows.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for t in self.tool_calls:
            # Escape pipes for table formatting
            target_clean = t.target.replace("|", "&#124;")
            if len(target_clean) > 60:
                target_clean = target_clean[:57] + "..."
            details_clean = t.details.replace("|", "&#124;")
            status_badge = f"`{t.status}`"
            rows.append(f"| {t.timestamp} | **{t.tool_type}** | {target_clean} | {status_badge} | {t.duration_ms}ms | {details_clean} |")
        return "\n".join(rows)

    def format_token_breakdown_markdown(self) -> str:
        """Render a stage-by-stage token and cost breakdown table."""
        if not self.stage_usages:
            return "*No LLM token consumption recorded yet.*"

        rows = []
        rows.append("| Pipeline Stage | Requests | Input Tokens | Output Tokens | Total Tokens | Estimated Cost |")
        rows.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for s in self.stage_usages:
            rows.append(f"| **{s.stage_name}** | {s.requests} | {s.input_tokens:,} | {s.output_tokens:,} | {s.total_tokens:,} | ${s.cost_usd:.4f} |")
        rows.append(f"| **TOTAL** | **{sum(s.requests for s in self.stage_usages)}** | **{self.total_input_tokens:,}** | **{self.total_output_tokens:,}** | **{self.total_tokens:,}** | **${self.total_cost_usd:.4f}** |")
        return "\n".join(rows)
