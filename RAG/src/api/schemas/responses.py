from typing import List, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    timestamp: float = Field(..., description="Unix timestamp of health check")


class QueryResponse(BaseModel):
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Generated answer or refusal response")
    route: str = Field(..., description="Routing decision made by the router (e.g. POLICY_LOOKUP, VERSION_DIFF)")
    route_reasoning: str = Field(..., description="Reasoning behind the router decision")
    abstained: bool = Field(..., description="True if the system refused to answer due to low confidence/out-of-scope")
    citation_valid: bool = Field(..., description="True if citation verification passed")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved source document snippets and metadata")
    execution_time_sec: float = Field(..., description="Latency in seconds")