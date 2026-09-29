# CCPA/CPRA Compliance RAG

A production-grade Retrieval-Augmented Generation system for querying California privacy law (CCPA/CPRA and its implementing regulations), with verified citations, confidence-gated abstention, and a version-tracking feature that answers not just "what's the rule" but "what changed, and when."

This is not a general-purpose "upload any PDF" chatbot. It's deliberately narrow: one regulatory domain, parsed and structured with full knowledge of its actual legal hierarchy (Title → Division → Chapter → Article → Section → Subsection), so that citations, retrieval, and version comparisons can all be trusted rather than guessed at.

> **This tool provides decision support, not legal advice.** Outputs should be verified against the official regulatory text and reviewed by qualified counsel before being relied upon.

---

## What it does

- **Answers compliance questions with verified citations.** Every claim in a response is checked against the retrieved source text before it's returned — a citation the system can't verify against retrieved content doesn't reach the user.
- **Knows when to abstain.** If retrieval confidence falls below a calibrated threshold, the system says so instead of guessing.
- **Tracks regulatory change over time.** The current regulation (Sept 2025 CPPA amendment, effective Jan 1, 2026) and its immediate predecessor (March 2023 base regulations) are both indexed, section-aligned, and diffable — including cases where amendments renumbered, split, or merged provisions rather than just editing them in place.

## Why this exists

Compliance teams spend real, recurring time answering questions like "does this apply to us" and "what changed in the last update" — usually by manually reading dense, cross-referenced regulatory text. This project is a working demonstration of doing that with retrieval-augmented generation, built with the rigor a compliance use case actually demands: nothing gets stated without a checked citation, and nothing gets flagged as "changed" without validated alignment.

## Architecture

```
PDF (redline, strikethrough/underline markup)
        │
        ▼
  Vector-aware parser (src/parsers)       ── separates N-1 and N text from a single
        │                                     redlined source document
        ▼
  Structure-aware chunker (src/chunkers)  ── parent-child chunking along statutory
        │                                     boundaries (§ → subsection → sub-item)
        ▼
  Embedder + Vector Store (src/embeddings, src/vector_store)
        │
        ▼
  Retriever + Reranker (src/retrieval)    ── hybrid search, cross-encoder rerank,
        │                                     semantic router (policy / diff / both)
        ▼
  Guardrails (src/guardrails)             ── citation verification, confidence-gated
        │                                     abstention
        ▼
  Generation (src/generation)             ── cited, grounded response

  Versioning pipeline (src/versioning), run offline:
  aligner.py   → maps provisions across versions (canonical_lineage_id), even when
                 renumbered, split, or merged
  card_generator.py → pre-writes verified "what changed" summaries, indexed
                 alongside regular content for retrieval like any other chunk
```

Key design decisions, and why:

- **One vector store, not two.** N-1 and N chunks live in the same collection, distinguished by a `version` metadata field. Comparisons are a metadata filter, not a separate system.
- **Diffs are precomputed, not generated live per query.** Alignment and "what changed" summaries are built once, offline, and verified once — not regenerated (and re-risked) on every user question. Live LLM comparison is a fallback for pairs the precomputed set doesn't cover, not the primary mechanism.
- **Lineage identity survives renumbering.** A `canonical_lineage_id` tracks a provision's identity across versions independently of its section/subsection label, because CCPA amendments do relabel provisions (e.g., inserting a new definition shifts every subsequent lettered definition).
- **Parent-child retrieval.** Small child chunks (150-400 tokens) are embedded for precise matching; retrieval returns the larger parent chunk (up to ~1,200 tokens) so the LLM sees full context, not a fragment.

## Project structure

```
├── cli.py                  # CLI entrypoint (ingestion, eval, server)
├── data/
│   ├── raw/                 # Source PDF(s)
│   └── processed/           # Parsed text, chunks, glossary, parent payloads
├── eval/
│   ├── golden_dataset.json  # Labeled Q&A set for scoring retrieval + faithfulness
│   └── eval_harness.py
├── src/
│   ├── api/                 # FastAPI app, routes, request/response schemas
│   ├── parsers/              # Vector-based PDF redline parser
│   ├── chunkers/              # Structure-aware statutory chunker
│   ├── embeddings/
│   ├── vector_store/
│   ├── retrieval/            # Retriever, reranker, semantic router
│   ├── guardrails/            # Citation verification, abstention
│   ├── versioning/            # Cross-version alignment + changelog card generation
│   └── schemas/               # Shared Pydantic models (chunks, lineage records)
├── tests/
└── pyproject.toml           # Managed with uv
```

## Setup

```bash
uv sync
cp .env.example .env   # fill in embedding/LLM provider credentials
```

## Usage

```bash
# Parse and chunk a new document version
uv run cli.py ingest --pdf data/raw/<file>.pdf --version 2026-09

# Run the alignment + changelog generation step (offline, once per new version)
uv run cli.py align --source-version 2023-03 --target-version 2026-09

# Run the evaluation harness against the golden dataset
uv run cli.py eval

# Start the API server
uv run uvicorn src.api.app:app --reload
```

## API

| Endpoint | Description |
|---|---|
| `POST /api/v1/query` | Natural-language question. Internally routed to policy text, changelog, or both, depending on intent. |
| `POST /api/v1/diff` | Structured comparison of a specific provision between two named versions, by lineage ID. |
| `GET /health` | Liveness check. |

## Evaluation

`eval/golden_dataset.json` holds a hand-labeled question set spanning single-clause lookups, cross-reference chains, and version-comparison questions. `eval_harness.py` reports retrieval recall, faithfulness (are claims actually supported by retrieved text), and citation accuracy (does every cited section genuinely exist in what was retrieved). Confidence-abstention thresholds are calibrated against this set rather than assumed.

## Data source and licensing

Regulatory text is sourced from the California Privacy Protection Agency (cppa.ca.gov), a public California state government body; California regulatory text is public record. This project's own code is licensed under [LICENSE — fill in, e.g., MIT].

## Status

- [x] PDF redline parsing (strikethrough/underline separation via vector-line detection)
- [ ] Structure-aware chunking (in progress)
- [ ] Hybrid retrieval + reranking
- [ ] Citation verification + confidence-gated abstention
- [ ] Cross-version alignment + changelog generation
- [ ] Evaluation harness run against golden dataset