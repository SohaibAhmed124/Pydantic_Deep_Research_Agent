from typing import List, Optional
from pydantic import BaseModel, Field


class EntityIntent(BaseModel):
    """Detection of whether user input is a stock ticker or general query."""
    is_ticker: bool = Field(
        description="True if the input is a stock ticker symbol (e.g., NVDA, AAPL, TSLA), False if a general topic."
    )
    ticker: Optional[str] = Field(
        default=None,
        description="The uppercase ticker symbol if is_ticker is True, otherwise None."
    )
    entity_name: str = Field(
        description="Resolved company or subject name (e.g., 'NVIDIA Corporation', 'Quantum Computing')."
    )
    core_domain_or_sector: str = Field(
        description="Industry sector, technology domain, or context (e.g., 'Semiconductors & AI Hardware')."
    )
    discovery_query: str = Field(
        description="Optimized initial DuckDuckGo search query to discover the current state and top news."
    )


class ResearchAngle(BaseModel):
    """An angle or sub-topic for deep dive investigation."""
    angle_title: str = Field(
        description="Concise, descriptive title for the research angle (e.g., 'SWOT Analysis', 'Competitive Positioning')."
    )
    search_query: str = Field(
        description="Targeted DuckDuckGo search query with relevant keywords, preferring primary financial/official terms if applicable."
    )
    focus_area: str = Field(
        description="Specific metrics, claims, or questions to answer for this angle."
    )


class DiscoveryPlan(BaseModel):
    """Collection of 3 to 4 distinct, non-overlapping research angles."""
    summary_of_context: str = Field(
        description="Brief summary of the initial discovery findings."
    )
    angles: List[ResearchAngle] = Field(
        description="Exactly 3 or 4 relevant, non-overlapping research angles.",
        min_length=3,
        max_length=4,
    )


class EvidenceItem(BaseModel):
    """A verified fact, metric, or claim paired with its source citation."""
    claim_or_metric: str = Field(
        description="Specific fact, numerical metric, quote, or finding extracted from the source."
    )
    source_title: str = Field(
        description="Title of the source publication or webpage."
    )
    source_url: str = Field(
        description="URL of the source where this claim was found."
    )


class AngleDeepDive(BaseModel):
    """Synthesized deep dive for a single research angle."""
    angle_title: str = Field(
        description="The title of the angle."
    )
    summary: str = Field(
        description="Comprehensive summary of findings for this angle."
    )
    key_findings: List[str] = Field(
        description="Bullet points of the most critical takeaways."
    )
    evidence_bullets: List[EvidenceItem] = Field(
        description="Specific claims and data points with their direct source citations."
    )
    risks_or_conflicts: List[str] = Field(
        description="Uncertainties, conflicting information, or headwind risks identified in this area."
    )
