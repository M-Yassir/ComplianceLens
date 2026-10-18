from typing import Optional
from src.rag.pipeline import RAGPipeline

# Global singleton to keep models loaded across HTTP requests
_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance