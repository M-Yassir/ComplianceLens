from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User regulatory or policy question", min_length=3)
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for tracing (e.g. ['production'])")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional key-value metadata for telemetry")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the requirements for dark patterns under CCPA §1798.100?",
                "tags": ["prod-api"],
                "metadata": {"user_id": "usr_123"}
            }
        }
    }


class DiffRequest(BaseModel):
    query: str = Field(..., description="Cross-version comparison query", min_length=3)
    sections: Optional[List[str]] = Field(default=None, description="Optional explicit statutory sections to compare (e.g. ['1798.100'])")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How did CCPA §1798.100 change between 2023 and 2026?",
                "sections": ["1798.100"]
            }
        }
    }