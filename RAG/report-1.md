# System Architecture & Diagnostic Report: Regulatory RAG Pipeline

---

## 1. Executive Summary

This report documents the current architecture, component design, and failure modes of the Regulatory Retrieval-Augmented Generation (RAG) system built for statutory compliance workloads (e.g., California Consumer Privacy Act / CCPA 2026 regulations).

While the pipeline functions smoothly on explicit citation-based queries, empirical testing reveals a critical failure mode: **semantic-only queries (queries omitting explicit section numbers like "§ 7123") consistently fail to retrieve relevant regulatory context**, triggering default abstentions or incorrect "no relevant information" responses from the generation model.

---

## 2. System Architecture & Component Breakdown

```
[User Query]
     │
     ▼
┌───────────────────────────┐
│       Query Router        │ ──► Dynamic Mode Switch (Heuristic vs. LLM Router)
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│     Hybrid Retrieval      │ ──► Dense Vector (Gemini Embeddings) + Sparse BM25
└─────────────┬─────────────┘
              │ (Top-20 Candidates)
              ▼
┌───────────────────────────┐
│    Cross-Encoder Rerank   │ ──► Scoring & Diversity Filtering (max 2/section)
└─────────────┬─────────────┘
              │ (Top-8 Reranked)
              ▼
┌───────────────────────────┐
│     LLM Generation        │ ──► Gemini 3.1 Flash Lite (AFC-enabled Generation)
└─────────────────────────┘

```

### Key Parameters & Signature

```python
def __init__(
    self,
    llm: Optional[Any] = None,
    router: Optional[QueryRouter] = None,
    use_llm_router: bool = False,
    min_rerank_score: float = 0.35,
    top_k_candidates: int = 20,
    top_k_reranked: int = 8,
    max_per_section: int = 2,
):

```

### Component Details

| Component | Technology / Implementation | Core Responsibility |
| --- | --- | --- |
| **Query Router** | Heuristic fallback + Dynamic LLM Router | Categorizes query intent (e.g., `POLICY_LOOKUP`), determines execution mode, and injects section-targeting logic when explicit anchors exist. |
| **Embedding Model** | `gemini-embedding-001` | Generates dense vector representations for both indexed regulatory chunks and incoming user queries. |
| **Vector Index** | ChromaDB Collection (`/api/v2/tenants/...`) | Stores chunked regulatory text along with metadata anchors (`section`, `part`, `clause`). |
| **Cross-Encoder Reranker** | Neural Cross-Encoder (`top_k_reranked=8`) | Re-scores top 20 candidates retrieved by hybrid search to assess true query-passage pair relevance. |
| **Diversity Filter** | Post-Rerank Window Deduplication (`max_per_section=2`) | Suppresses redundant chunks from identical sections to preserve diversity in the top context window. |
| **Generator** | Gemini 3.1 Flash Lite | Generates structured compliance answers grounded strictly in retrieved chunks using Automatic Function Calling (AFC). |

---

## 3. Critical Failure Analysis: The "Citation-Free Query" Bottleneck

When tested on high-level semantic queries lacking explicit section citations (e.g., *"Does standard SMS-based MFA fulfill authentication requirements for contractors?"* instead of *"What does § 7123(c)(1)(A) say about MFA?"*), the system exhibits systemic failure.

### Root Cause Mechanics

```
┌────────────────────────────────────────────────────────────────────────┐
│ Query: "Is SMS-based MFA compliant for contractors?"                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Dense Embeddings + BM25 retrieve 20 chunks containing:                │
│ "contractor", "compliance", "consumer", "written contract", "business" │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Reranker scores general rights sections higher due to repetitive       │
│ regulatory keywords across all sections:                              │
│ • § 7022 (Deletion)   ── Score: 0.97  [INCORRECT]                     │
│ • § 7023 (Correction) ── Score: 0.95  [INCORRECT]                     │
│ • § 7011 (Notice)     ── Score: 0.94  [INCORRECT]                     │
│ • § 7123 (Cyber Audit)── Score: 0.65  [CORRECT - DROPPED FROM TOP 8]   │
└────────────────────────────────────────────────────────────────────────┘

```

1. **Boilerplate Keyword Contamination (Lexical Noise):**
Regulatory texts share an extremely high density of overlapping vocabulary (`business`, `consumer`, `service provider`, `contractor`, `personal information`). When a query lacks an exact section anchor, both sparse BM25 and dense vector search surface chunks from unrelated sections (§ 7011, § 7022, § 7023, § 7026).
2. **Cross-Encoder False Positives:**
Because Cross-Encoders evaluate raw text overlap alongside semantic intent, general regulatory boilerplate inflates the rerank scores of non-target sections above `0.80`. Target chunks containing the actual substantive answer (e.g., § 7123 audit provisions) receive lower relative scores (`0.60 - 0.70`) and fall outside the `top_k_reranked = 8` cutoff window.
3. **In-Window Diversity Filter Bypass:**
The `max_per_section = 2` constraint prevents single-section dominance, but when 6–8 *distinct* non-relevant sections all score `>0.75` due to shared legal boilerplate, the top 8 positions fill up before the correct section is ever evaluated.
4. **Brittle Abstention & Strict Framing:**
Because the system prompt instructs the generator to rely *only* on provided context, the absence of the correct chunk in the top 8 leads directly to a false abstention (*"The provided text does not explicitly address..."*).

---

## 4. Production Hardening Roadmap

To resolve this issue systematically across all regulatory topics without relying on hardcoded section patches, the pipeline requires architectural enhancements at the chunking, expansion, and indexing layers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROPOSED PRODUCTION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
 1. CONTEXTUAL CHUNKING ──────► Prepends Section Header & Scope to every Chunk
                                       │
                                       ▼
 2. HYDE / QUERY EXPANSION ───► Translates "SMS MFA" ──► "Authentication, Passwords, Cybersecurity Audit Requirements"
                                       │
                                       ▼
 3. TWO-TIERED RERANKING ─────► First-stage section clustering, Second-stage chunk selection

```

### Roadmap Initiatives

* **Contextual Chunk Header Injection (Indexing Layer)**
* *Action:* Mutate chunk payloads prior to embedding. Every chunk must prepended with parent section metadata.
* *Example Payload:* `[Section: § 7123 Cybersecurity Audit | Subject: Authentication & MFA] The cybersecurity audit must assess...`
* *Impact:* Forces vector embeddings to capture high-level section context alongside local chunk text.


* **Hypothetical Document Embeddings (HyDE) / Query Transformation (Router Layer)**
* *Action:* Before passing semantic queries to hybrid retrieval, use a lightweight prompt to expand the query into statutory legal language.
* *Example:* Query *"SMS MFA for contractors"* expands to *"Cybersecurity program authentication requirements, multi-factor authentication, phishing-resistant controls, § 7123"*.


* **Hierarchical (Parent-Child) Indexing**
* *Action:* Retrieve at the granular child level (e.g., subsection `(c)(1)(A)`), but deliver the parent section context (`§ 7123` full scope) to the LLM context window.


* **Custom Cross-Encoder Calibration / Negative Sampling**
* *Action:* Fine-tune or score-calibrate the reranker using hard negative legal pairs to penalize cross-section keyword matches when the specific sub-domain (e.g., Cybersecurity vs. Opt-Out) does not align.