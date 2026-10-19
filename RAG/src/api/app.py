from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langfuse import get_client

from src.api.routes import health, query
from src.api.dependencies import get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up RAG pipeline (models, vector index connections) on startup
    get_pipeline()
    yield
    # Flush buffered Langfuse traces on server shutdown
    try:
        get_client().flush()
    except Exception:
        pass


app = FastAPI(
    title="CCPA Policy RAG API",
    description="Unified API for CCPA compliance policy lookup, version diffs, and guardrail enforcement.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "CCPA RAG API is online. Docs available at /docs"}