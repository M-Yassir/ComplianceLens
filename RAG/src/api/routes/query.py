import time
from fastapi import APIRouter, Depends, HTTPException, status
from src.api.schemas.requests import QueryRequest
from src.api.schemas.responses import QueryResponse
from src.api.dependencies import get_pipeline
from src.rag.pipeline import RAGPipeline, RAGResponse

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_rag(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline)
) -> QueryResponse:
    """Unified query endpoint that automatically routes, retrieves, and generates answers."""
    start_time = time.time()
    try:
        rag_resp: RAGResponse = pipeline.query(
            user_query=request.query,
            tags=request.tags or ["api"],
            metadata=request.metadata or {}
        )
        exec_time = round(time.time() - start_time, 3)

        return QueryResponse(
            query=rag_resp.query,
            answer=rag_resp.answer,
            route=rag_resp.routing_decision.route.value,
            route_reasoning=rag_resp.routing_decision.reasoning,
            abstained=rag_resp.abstention_decision.should_abstain,
            citation_valid=rag_resp.citation_verification.is_valid,
            sources=rag_resp.source_documents,
            execution_time_sec=exec_time,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG execution error: {str(e)}"
        )