#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

echo "Building folder structure safely..."

# 1. Create Directories
mkdir -p data/raw data/processed
mkdir -p eval
mkdir -p src/api/schemas src/api/routes
mkdir -p src/parsers
mkdir -p src/chunkers
mkdir -p src/versioning
mkdir -p src/embeddings
mkdir -p src/vector_store
mkdir -p src/retrieval
mkdir -p src/guardrails
mkdir -p src/generation
mkdir -p tests

# 2. Create Root Files (touch will NOT overwrite existing contents)
touch .dockerignore .env.example Dockerfile docker-compose.yml cli.py

# 3. Create Evaluation Harness Files
touch eval/golden_dataset.json eval/eval_harness.py

# 4. Create Source Python Modules and Packages
touch src/__init__.py src/config.py

touch src/api/__init__.py src/api/app.py src/api/dependencies.py
touch src/api/schemas/__init__.py src/api/schemas/requests.py src/api/schemas/responses.py
touch src/api/routes/__init__.py src/api/routes/health.py src/api/routes/query.py src/api/routes/diff.py src/api/routes/ingest.py

touch src/parsers/__init__.py src/parsers/pdf_vector_parser.py
touch src/chunkers/__init__.py src/chunkers/legal_chunker.py
touch src/versioning/__init__.py src/versioning/aligner.py src/versioning/card_generator.py
touch src/embeddings/__init__.py src/embeddings/embedder.py
touch src/vector_store/__init__.py src/vector_store/index_manager.py
touch src/retrieval/__init__.py src/retrieval/router.py src/retrieval/retriever.py src/retrieval/reranker.py
touch src/guardrails/__init__.py src/guardrails/citation_verifier.py src/guardrails/abstention.py
touch src/generation/__init__.py src/generation/prompt_templates.py src/generation/llm_service.py

# 5. Create Unit Test Suite Files
touch tests/__init__.py tests/test_api.py tests/test_parser.py tests/test_lineage.py

echo "Structure created successfully! Existing files remain completely untouched."