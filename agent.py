import asyncio
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from pydantic_ai import Agent

from research_types import (
    AngleDeepDive,
    DiscoveryPlan,
    EntityIntent,
    EvidenceItem,
    ResearchAngle,
)
from search_tools import ddg_search_async, fetch_page_markdown_async
from telemetry import ResearchTelemetry

# Load environment variables
load_dotenv(override=True)


def get_model_name() -> str:
    """Resolve the OpenAI model name from the environment."""
    model_name = os.getenv("OPENAI_MODEL", "openai:gpt-6-luna").strip()
    if not model_name:
        model_name = "openai:gpt-6-luna"
    if not any(model_name.startswith(p) for p in ("openai:", "openai-chat:", "openai-decisions:")):
        model_name = f"openai:{model_name}"
    return model_name


FALLBACK_MODEL = "openai:gpt-4o-mini"


# ==============================================================================
# Prompts
# ==============================================================================
INTENT_SYSTEM_PROMPT = """
You are an expert financial and technical intelligence analyst.
Analyze the user's input to determine if it is a stock ticker symbol (e.g. NVDA, TSLA, AAPL, MSFT) or a general topic/query.
1. If it is a stock ticker:
   - Set is_ticker = True
   - Extract the uppercase ticker symbol
   - Resolve the exact company name (e.g. 'NVDA' -> 'NVIDIA Corporation')
   - Identify the primary sector and core technologies (e.g. 'Semiconductors, GPUs, AI Accelerators')
   - Formulate an initial discovery search query that will find recent financial and strategic overviews.
2. If it is a general query or technology topic:
   - Set is_ticker = False, ticker = None
   - Resolve the primary entity / subject name
   - Identify the technology domain or category
   - Formulate an initial discovery search query to understand recent breakthroughs and state-of-the-art.
Return strictly the structured EntityIntent output.
"""

DISCOVERY_SYSTEM_PROMPT = """
You are an expert research strategist.
Based on the entity context and initial DuckDuckGo search snippets, construct exactly 3 or 4 distinct, non-overlapping research angles for a deep dive.
- For a stock/company, ensure the angles cover:
  1. SWOT Analysis / Competitive Position & Moat
  2. Last 12 Months Stock Performance & Market Drivers
  3. Competition and Market Positioning
  4. Latest Quarterly Financial Results and Forward Guidance
- For general technical queries, generate 3 or 4 comprehensive, non-overlapping pillars:
  1. Fundamental Technology Architecture & Current Capabilities
  2. Commercial Adoption & Leading Market Implementations
  3. Key Bottlenecks, Risks, and Limitations
  4. Future Outlook & Next-Gen Innovations
Each angle must have a targeted DuckDuckGo search query tailored to yield factual data, metrics, or primary sources.
"""

DEEP_DIVE_SYSTEM_PROMPT = """
You are an investigative research analyst.
Examine the provided search snippets and extracted webpage texts for a specific research angle.
Extract key facts, verifiable numbers, dates, metrics, and claims.
Track which source (Title and URL) supports each claim.
Identify any risks, headwinds, uncertainties, or conflicting/contradictory claims across sources.
Prefer primary sources (e.g. investor relations, SEC filings, official statements) and reputable news outlets.
Return strictly the structured AngleDeepDive output.
"""

SYNTHESIS_SYSTEM_PROMPT = """
You are a senior research director authoring an authoritative, comprehensive intelligence report.
Given the entity context and the deep dive results across all angles, produce a single detailed, highly structured Markdown report.

The report MUST include:
# [Entity / Query Title]: Deep Research Report
*Sector/Domain:* [Domain] | *Generated:* [Current Date/Context]

## Executive Summary
A concise executive synthesis capturing the overarching state, key investment/technology thesis, and primary takeaways.

## Key Findings by Research Angle
For EACH research angle investigated:
### [Angle Title]
- **Overview & Context:** Narrative detailing the core findings.
- **Key Takeaways:** 3-5 high-impact bullet points.
- **Evidence & Supporting Citations:** Direct bullet points linking specific facts and metrics to their sources using markdown links: `[Source Title](URL)`. Example: `- Data Center revenue grew 112% year-over-year ([NVIDIA Q4 FY25 Release](https://investor.nvidia.com/...))`.

## Risks, Uncertainties & Conflicting Information
Identify headwinds, regulatory risks, technological hurdles, or differing opinions/projections observed across the sources.

## What to Watch Next
A concrete checklist of upcoming catalysts, earnings dates, regulatory milestones, or technical checkpoints to monitor.

## References & Sources
A consolidated table or bulleted list of all consulted sources with clickable markdown links.

Quality rules:
- Be thorough, objective, and dense with facts and data points.
- Always include the actual URLs provided in the evidence as clickable markdown links.
- Avoid vague generalizations; cite exact figures, percentages, dates, and names where present.
"""


def _create_pydantic_agents(model_name: str):
    """Instantiate the 4 specialized agents for the given model."""
    intent_agent = Agent(
        model=model_name,
        output_type=EntityIntent,
        system_prompt=INTENT_SYSTEM_PROMPT,
    )
    discovery_agent = Agent(
        model=model_name,
        output_type=DiscoveryPlan,
        system_prompt=DISCOVERY_SYSTEM_PROMPT,
    )
    deep_dive_agent = Agent(
        model=model_name,
        output_type=AngleDeepDive,
        system_prompt=DEEP_DIVE_SYSTEM_PROMPT,
    )
    synthesis_agent = Agent(
        model=model_name,
        output_type=str,
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
    )
    return intent_agent, discovery_agent, deep_dive_agent, synthesis_agent


async def _execute_with_model_fallback(agent_call_fn: Callable[[Agent], Any], model_name: str):
    """Execute an agent call, returning (result_object, used_model)."""
    try:
        current_agents = _create_pydantic_agents(model_name)
        res = await agent_call_fn(current_agents)
        return res, model_name
    except Exception as e:
        err_str = str(e).lower()
        if model_name != FALLBACK_MODEL and ("not found" in err_str or "does not exist" in err_str or "404" in err_str):
            print(f"[Model Fallback] Model '{model_name}' not available. Falling back to '{FALLBACK_MODEL}'...")
            fallback_agents = _create_pydantic_agents(FALLBACK_MODEL)
            res = await agent_call_fn(fallback_agents)
            return res, FALLBACK_MODEL
        raise e


async def run_single_angle_deep_dive(
    angle: ResearchAngle,
    is_ticker: bool,
    semaphore: asyncio.Semaphore,
    model_name: str,
    telemetry: ResearchTelemetry,
) -> AngleDeepDive:
    """Execute research, scraping, and fact-extraction for a single angle."""
    async with semaphore:
        telemetry.log("AGENT", f"Starting Deep Dive on Angle: '{angle.angle_title}'")

        # 1. Targeted DuckDuckGo search
        search_results = await ddg_search_async(
            angle.search_query,
            max_results=5,
            is_financial=is_ticker,
            telemetry=telemetry,
        )

        # 2. Fetch page markdown for top 2 accessible links
        fetched_contents = []
        for item in search_results[:2]:
            url = item.get("href")
            if url:
                md_content = await fetch_page_markdown_async(
                    url, timeout=5.0, max_chars=3000, telemetry=telemetry
                )
                if md_content:
                    fetched_contents.append(f"Source: {item.get('title')}\nURL: {url}\nContent:\n{md_content}\n---")

        snippets_block = "\n".join(
            f"- [{r.get('title')}]({r.get('href')}): {r.get('body')}"
            for r in search_results
        )
        pages_block = "\n\n".join(fetched_contents) if fetched_contents else "No full page text extracted; rely on snippets."

        prompt = f"""
Research Angle: {angle.angle_title}
Focus Area: {angle.focus_area}
Search Query Used: {angle.search_query}

=== SEARCH SNIPPETS ===
{snippets_block}

=== EXTRACTED PAGE TEXTS ===
{pages_block}

Extract key findings, exact facts/numbers with citations, and any risks or conflicting info.
"""

        async def _call(agents):
            _, _, deep_dive_agent, _ = agents
            return await deep_dive_agent.run(prompt)

        res, used_model = await _execute_with_model_fallback(_call, model_name)
        usage = res.usage() if hasattr(res, "usage") and callable(res.usage) else getattr(res, "usage", None)
        telemetry.record_stage_usage(f"Deep Dive: {angle.angle_title[:22]}", usage, used_model)
        tot_tok = getattr(usage, 'total_tokens', 0) if usage else 0
        telemetry.log("INFO", f"Completed Angle '{angle.angle_title}' ({tot_tok:,} tokens)")
        return res.output


async def run_deep_research(
    user_input: str,
    model_name: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float, ResearchTelemetry], Any]] = None,
) -> Tuple[str, ResearchTelemetry]:
    """
    End-to-end Deep Research Orchestrator with telemetry instrumentation.
    Returns (final_markdown_report, telemetry_object).
    """
    selected_model = model_name or get_model_name()
    telemetry = ResearchTelemetry(model_name=selected_model)

    def update_progress(message: str, step: float):
        telemetry.log("INFO", message)
        if progress_callback:
            progress_callback(message, step, telemetry)

    telemetry.log("INFO", f"Deep Research initialized for query: '{user_input}' (Model: {selected_model})")

    # --------------------------------------------------------------------------
    # Step 1: Intent & Entity Detection
    # --------------------------------------------------------------------------
    update_progress("🔍 Step 1/4: Analyzing input & detecting entities...", 0.10)

    async def _detect_intent(agents):
        intent_agent, _, _, _ = agents
        return await intent_agent.run(f"Input: {user_input.strip()}")

    intent_res, used_model = await _execute_with_model_fallback(_detect_intent, selected_model)
    intent: EntityIntent = intent_res.output
    usage = intent_res.usage() if hasattr(intent_res, "usage") and callable(intent_res.usage) else getattr(intent_res, "usage", None)
    telemetry.record_stage_usage("Step 1: Intent & Entity Detection", usage, used_model)

    entity_desc = f"{intent.entity_name} ({intent.ticker})" if intent.is_ticker else intent.entity_name
    telemetry.log("AGENT", f"Entity Resolved: {entity_desc} | Sector: {intent.core_domain_or_sector}")

    # --------------------------------------------------------------------------
    # Step 2: Initial Discovery Search & Angle Planning
    # --------------------------------------------------------------------------
    update_progress(f"🌐 Step 2/4: Initial discovery search for '{entity_desc}'...", 0.25)
    discovery_results = await ddg_search_async(
        intent.discovery_query,
        max_results=5,
        is_financial=intent.is_ticker,
        telemetry=telemetry,
    )

    snippets_text = "\n".join(
        f"- {r.get('title')}: {r.get('body')} ({r.get('href')})"
        for r in discovery_results
    )

    discovery_prompt = f"""
Entity: {entity_desc}
Sector / Context: {intent.core_domain_or_sector}
Is Ticker: {intent.is_ticker}

Initial Discovery Results:
{snippets_text}

Generate 3-4 distinct, non-overlapping research angles and targeted search queries.
"""

    async def _plan_discovery(agents):
        _, discovery_agent, _, _ = agents
        return await discovery_agent.run(discovery_prompt)

    plan_res, used_model = await _execute_with_model_fallback(_plan_discovery, selected_model)
    plan: DiscoveryPlan = plan_res.output
    usage = plan_res.usage() if hasattr(plan_res, "usage") and callable(plan_res.usage) else getattr(plan_res, "usage", None)
    telemetry.record_stage_usage("Step 2: Angle Planning", usage, used_model)

    telemetry.log("AGENT", f"Formulated {len(plan.angles)} Research Angles: {[a.angle_title for a in plan.angles]}")

    # --------------------------------------------------------------------------
    # Step 3: Parallel Deep Dives
    # --------------------------------------------------------------------------
    num_angles = len(plan.angles)
    update_progress(f"⚡ Step 3/4: Executing parallel deep dives across {num_angles} research angles...", 0.50)

    semaphore = asyncio.Semaphore(2)
    tasks = [
        run_single_angle_deep_dive(
            angle=angle,
            is_ticker=intent.is_ticker,
            semaphore=semaphore,
            model_name=selected_model,
            telemetry=telemetry,
        )
        for angle in plan.angles
    ]

    deep_dive_results: List[AngleDeepDive] = await asyncio.gather(*tasks)

    # --------------------------------------------------------------------------
    # Step 4: Final Synthesis
    # --------------------------------------------------------------------------
    update_progress("📝 Step 4/4: Synthesizing comprehensive research report...", 0.85)

    deep_dives_summary = []
    for dd in deep_dive_results:
        evidence_str = "\n".join(
            f"  * {ev.claim_or_metric} (Source: [{ev.source_title}]({ev.source_url}))"
            for ev in dd.evidence_bullets
        )
        findings_str = "\n".join(f"  * {f}" for f in dd.key_findings)
        risks_str = "\n".join(f"  * {r}" for r in dd.risks_or_conflicts) if dd.risks_or_conflicts else "  * None noted."

        deep_dives_summary.append(f"""
### Research Angle: {dd.angle_title}
**Summary:** {dd.summary}

**Key Findings:**
{findings_str}

**Evidence with Citations:**
{evidence_str}

**Risks / Uncertainties:**
{risks_str}
""")

    synthesis_prompt = f"""
Entity: {entity_desc}
Core Domain/Sector: {intent.core_domain_or_sector}
Is Stock Ticker: {intent.is_ticker}
Discovery Context: {plan.summary_of_context}

=== DEEP DIVE RESEARCH FINDINGS ===
{"".join(deep_dives_summary)}

Produce the complete, authoritative Markdown report following the specified structure.
Ensure all evidence bullets feature clickable markdown links `[Source Title](URL)`.
"""

    async def _synthesize(agents):
        _, _, _, synthesis_agent = agents
        return await synthesis_agent.run(synthesis_prompt)

    synth_res, used_model = await _execute_with_model_fallback(_synthesize, selected_model)
    final_report: str = synth_res.output
    usage = synth_res.usage() if hasattr(synth_res, "usage") and callable(synth_res.usage) else getattr(synth_res, "usage", None)
    telemetry.record_stage_usage("Step 4: Report Synthesis", usage, used_model)

    telemetry.finish()
    update_progress(f"✅ Deep Research Complete! Total Tokens: {telemetry.total_tokens:,} | Cost: ${telemetry.total_cost_usd:.4f}", 1.0)
    return final_report, telemetry


# ==============================================================================
# Quick Concise Agent (Preserved for Instant Chat)
# ==============================================================================
CONCISE_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Always answer questions in a short, direct, and concise way. "
    "Provide only key information and essential facts without unnecessary filler."
)

quick_agent = Agent(
    model=get_model_name(),
    system_prompt=CONCISE_SYSTEM_PROMPT,
)
