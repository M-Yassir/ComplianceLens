from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    timestamp: float = Field(..., description="Unix timestamp of health check")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated RAG response or refusal text")
    route: str = Field(..., description="Routing decision (POLICY_LOOKUP, VERSION_DIFF, HYBRID_BOTH, etc.)")
    route_reasoning: str = Field(..., description="Explanation for routing selection")
    abstained: bool = Field(..., description="True if system refused to answer due to out-of-scope or ungrounded context")
    citation_valid: bool = Field(..., description="True if citation guardrail verified statutory references")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved source document chunks")
    execution_time_sec: float = Field(..., description="Latency in seconds")


class DiffResponse(BaseModel):
    answer: str = Field(..., description="Generated cross-version comparison analysis")
    sections_analyzed: List[str] = Field(default_factory=list, description="Statutory sections evaluated")
    abstained: bool = Field(..., description="True if system abstained from answering")
    execution_time_sec: float = Field(..., description="Latency in seconds")