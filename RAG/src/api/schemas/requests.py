from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query (policy lookup, version comparison, or general question)", min_length=3)
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for Langfuse tracing")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional key-value metadata for telemetry")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What was changed concerning Opt-out Preference Signals between 2023 and 2026?",
                "tags": ["api-production"],
                "metadata": {"user_id": "usr_123"}
            }
        }
    }