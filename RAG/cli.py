#!/usr/bin/env python3
"""ComplianceLens command line.

Run from the RAG/ directory:

    python cli.py doctor                  check the environment and the index
    python cli.py ingest                  parse -> chunk -> index (full build)
    python cli.py ingest --stage index --reindex
    python cli.py stats                   chunk statistics per version
    python cli.py ask "..."               ask one question through the full pipeline
    python cli.py eval --llm-judge --max-samples 150
    python cli.py sweep                   offline retrieval/abstention tuning
    python cli.py serve                   start the FastAPI server

`eval` and `sweep` forward every extra argument to the scripts in eval/, so
their own flags (python cli.py eval --help) stay the single source of truth.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PDF_NAME = "ccpa_updates_cyber_risk_admt_appr_text.pdf"
VERSIONS = ("2023", "2026")
INDEXED_VERSION = "2026"  # only the current version is searched by meaning

# Indexing is rate limited by index_manager: 10 chunks every 6 seconds.
INDEX_BATCH = 10
INDEX_DELAY_SEC = 6.0


# --------------------------------------------------------------------------- output helpers
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def ok(msg: str) -> None:
    print(f"{_c('32', 'ok  ')} {msg}")


def warn(msg: str) -> None:
    print(f"{_c('33', 'warn')} {msg}")


def fail(msg: str) -> None:
    print(f"{_c('31', 'FAIL')} {msg}")


def info(msg: str) -> None:
    print(f"{_c('2', '...')} {msg}")


def heading(msg: str) -> None:
    print(f"\n{_c('1', msg)}")


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _settings():
    from src.config import settings

    return settings


def _processed_dir() -> Path:
    return _settings().PROCESSED_DATA_DIR


def _find_pdf(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    raw_dir = ROOT / "data" / "raw"
    default = raw_dir / DEFAULT_PDF_NAME
    if default.exists():
        return default
    pdfs = sorted(raw_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else default


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- ingest
def _stage_parse(pdf: Path) -> bool:
    from src.parsers.pdf_vector_parser import parse_and_split_pdf

    heading("1/3  Parse the redlined PDF")
    if not pdf.exists():
        fail(f"PDF not found: {pdf}")
        return False
    out = _processed_dir()
    info(f"{pdf.name} -> {out}")
    res = parse_and_split_pdf(pdf_path=pdf, output_dir=out)
    m = res["metrics"]
    ok(f"pages: {m['pages_processed']}")
    ok(
        f"words: normal {m['normal_words']:,} | struck (2023 only) {m['strikethrough_words']:,} | "
        f"underlined (2026 only) {m['underline_words']:,} | margin stripped {m['margin_words_stripped']:,}"
    )
    if m["strikethrough_words"] == 0 or m["underline_words"] == 0:
        warn("no struck or underlined words detected: is this really a redlined PDF?")
        return False
    return True


def _stage_chunk() -> bool:
    from src.chunkers.legal_chunker import run_chunker_pipeline

    heading("2/3  Chunk both versions")
    out = _processed_dir()
    for version in VERSIONS:
        src_txt = out / f"version_{version}.txt"
        if not src_txt.exists():
            fail(f"missing {src_txt.name}: run `ingest --stage parse` first")
            return False
        run_chunker_pipeline(src_txt, out, version_tag=version)
    return True


def _stage_index(reindex: bool) -> bool:
    heading(f"3/3  Index the {INDEXED_VERSION} child chunks into Chroma")
    chunks_file = _processed_dir() / f"chunks_{INDEXED_VERSION}.json"
    if not chunks_file.exists():
        fail(f"missing {chunks_file.name}: run `ingest --stage chunk` first")
        return False

    n = len(_load_json(chunks_file))
    eta = math.ceil(n / INDEX_BATCH) * INDEX_DELAY_SEC
    info(f"{n} chunks, rate limited to {INDEX_BATCH} per {INDEX_DELAY_SEC:.0f}s: about {_fmt_duration(eta)}")

    from src.vector_store.index_manager import VectorIndexManager, load_and_index_file

    try:
        manager = VectorIndexManager()
    except Exception as e:  # connection problems are the common failure here
        fail(f"cannot connect to Chroma: {e}")
        info("run `python cli.py doctor` to see which host and port are being used")
        return False

    if reindex:
        before = manager.count()
        manager.delete_version(INDEXED_VERSION)
        info(f"--reindex: removed version {INDEXED_VERSION} chunks (collection had {before})")

    started = time.time()
    load_and_index_file(chunks_file, manager)
    ok(f"collection '{manager.collection_name}' now holds {manager.count()} chunks ({_fmt_duration(time.time() - started)})")
    return True


def cmd_ingest(args: argparse.Namespace) -> int:
    pdf = _find_pdf(args.pdf)
    stages = {"parse": [0], "chunk": [1], "index": [2], "all": [0, 1, 2]}[args.stage]
    steps = [
        lambda: _stage_parse(pdf),
        _stage_chunk,
        lambda: _stage_index(args.reindex),
    ]
    for i in stages:
        if not steps[i]():
            return 1
    print()
    ok("ingest finished")
    return 0


# --------------------------------------------------------------------------- stats
def _summ(tokens: List[int]) -> str:
    t = sorted(tokens)
    if not t:
        return "n=0"
    p95 = t[int(0.95 * (len(t) - 1))]
    return f"n={len(t)}  median={statistics.median(t):.0f}  p95={p95}  max={t[-1]}"


def cmd_stats(args: argparse.Namespace) -> int:
    proc = _processed_dir()
    status = 0
    for v in VERSIONS:
        paths = {k: proc / f"{k}_{v}.json" for k in ("chunks", "parents", "glossary")}
        missing = [p.name for p in paths.values() if not p.exists()]
        heading(f"Version {v}")
        if missing:
            fail(f"missing: {', '.join(missing)} (run `ingest --stage chunk`)")
            status = 1
            continue
        chunks = _load_json(paths["chunks"])
        parents = _load_json(paths["parents"])
        glossary = _load_json(paths["glossary"])
        print(f"  child chunks   {_summ([c['token_count'] for c in chunks])}")
        print(f"  parent chunks  {_summ([p['token_count'] for p in parents])}")
        print(f"  glossary terms {len(glossary)}")
        print(f"  sections       {len({c['section'] for c in chunks})}")
        print(f"  children > 400 tokens: {sum(c['token_count'] > 400 for c in chunks)}")
        print(f"  fallback-split children: {sum(bool(c.get('is_fallback_split')) for c in chunks)}")
    return status


# --------------------------------------------------------------------------- ask
def _print_answer(resp, elapsed: float, verbose: bool) -> None:
    route = resp.routing_decision.route.value
    abst = resp.abstention_decision
    cit = resp.citation_verification
    refused = bool(abst.should_abstain)

    heading("Question")
    print(f"  {resp.query}")
    print()
    print(f"  route      {route}  {_c('2', '(' + resp.routing_decision.reasoning + ')')}")
    status = _c("33", "REFUSED") if refused else _c("32", "ANSWERED")
    print(f"  result     {status}")
    cit_txt = "valid" if cit.is_valid else _c("31", "INVALID: " + cit.summary)
    print(f"  citations  {cit_txt}")
    print(f"  latency    {elapsed:.1f}s")

    heading("Answer")
    print(resp.answer.strip())

    sources = resp.source_documents or []
    if sources:
        heading("Sources")
        for s in sources:
            score = s.get("rerank_score")
            score_txt = f"  score {score:.4f}" if isinstance(score, (int, float)) and score else ""
            print(f"  § {s.get('section')}  {s.get('section_title') or ''}{score_txt}")
            if verbose and s.get("snippet"):
                print(f"      {_c('2', s['snippet'])}")

    if verbose:
        heading("Decision detail")
        print(f"  abstention reason : {abst.reason}")
        print(f"  top score         : {abst.confidence_score}")
        print(f"  score margin      : {abst.score_margin}")
        status_attr = getattr(resp, "answer_status", None)
        if status_attr:
            print(f"  answer status     : {status_attr}")
        ctx = getattr(resp, "context_sources", None)
        if refused and ctx:
            print("  retrieved (not shown to the user):")
            for s in ctx:
                print(f"    § {s.get('section')}  {s.get('section_title') or ''}  score {s.get('rerank_score')}")


def cmd_ask(args: argparse.Namespace) -> int:
    question = " ".join(args.question).strip()
    if len(question) < 3:
        fail("the question must be at least 3 characters")
        return 2

    # Library code prints progress to stdout; keep stdout clean when --json is used.
    out_ctx = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    with out_ctx:
        from src.rag.pipeline import RAGPipeline

        info("loading pipeline (models, parent maps, BM25 index)...")
        kwargs = {"use_llm_router": True} if args.llm_router else {}
        pipeline = RAGPipeline(**kwargs)
        started = time.time()
        resp = pipeline.query(question, tags=args.tag or ["cli"], metadata={"source": "cli"})
        elapsed = time.time() - started

    # A short-lived process must flush buffered Langfuse traces before exiting.
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass

    if args.json:
        payload = resp.model_dump(mode="json")
        payload["execution_time_sec"] = round(elapsed, 3)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_answer(resp, elapsed, args.verbose)
    return 0


# --------------------------------------------------------------------------- eval / sweep (pass-through)
def _run_script(script: str, extra: List[str]) -> int:
    path = ROOT / "eval" / script
    if not path.exists():
        fail(f"{path} not found")
        return 1
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(path), *extra]
    info("running: " + " ".join(["python", f"eval/{script}", *extra]))
    return subprocess.call(cmd, cwd=ROOT, env=env)


def cmd_eval(args: argparse.Namespace, extra: List[str]) -> int:
    return _run_script("eval_harness.py", extra)


def cmd_sweep(args: argparse.Namespace, extra: List[str]) -> int:
    if "--sweep" not in extra:
        extra = ["--sweep", *extra]
    return _run_script("eval_retrieval_tuning.py", extra)


# --------------------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    problems = 0
    s = _settings()

    heading("Configuration")
    env_file = ROOT / ".env"
    if env_file.exists():
        ok(f".env found ({env_file})")
    else:
        warn(f"no .env at {env_file}: only environment variables and defaults are used")
    for name in ("EMBEDDING_API_KEY", "LLM_API_KEY"):
        if getattr(s, name):
            ok(f"{name} is set")
        else:
            fail(f"{name} is empty")
            problems += 1
    print(f"     embeddings : {s.EMBEDDING_PROVIDER} / {s.EMBEDDING_MODEL} / {s.EMBEDDING_DIMENSION}d")
    print(f"     llm        : {s.LLM_PROVIDER} / {s.LLM_MODEL}")
    print(f"     chroma     : {s.CHROMA_HOST}:{s.CHROMA_PORT} (collection '{s.CHROMA_COLLECTION_NAME}')")

    heading("Processed data")
    proc = s.PROCESSED_DATA_DIR
    chunk_counts = {}
    for v in VERSIONS:
        for kind in ("version", "chunks", "parents", "glossary"):
            ext = "txt" if kind == "version" else "json"
            p = proc / f"{kind}_{v}.{ext}"
            if p.exists():
                ok(p.name)
                if kind == "chunks":
                    chunk_counts[v] = len(_load_json(p))
            else:
                fail(f"{p.name} missing (run `python cli.py ingest`)")
                problems += 1
    pdf = _find_pdf(None)
    if pdf.exists():
        ok(f"source PDF: {pdf.name}")
    else:
        warn(f"source PDF not found in {ROOT / 'data' / 'raw'} (only needed for `ingest --stage parse`)")

    heading("Chroma")
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.HttpClient(
            host=s.CHROMA_HOST, port=s.CHROMA_PORT, settings=ChromaSettings(anonymized_telemetry=False)
        )
        client.heartbeat()
        ok(f"server reachable at {s.CHROMA_HOST}:{s.CHROMA_PORT}")
        names = [getattr(c, "name", c) for c in client.list_collections()]
        if s.CHROMA_COLLECTION_NAME not in names:
            fail(f"collection '{s.CHROMA_COLLECTION_NAME}' does not exist (run `python cli.py ingest --stage index`)")
            problems += 1
        else:
            col = client.get_collection(s.CHROMA_COLLECTION_NAME)
            count = col.count()
            expected = chunk_counts.get(INDEXED_VERSION)
            if expected is None:
                warn(f"collection holds {count} chunks (no chunks_{INDEXED_VERSION}.json to compare)")
            elif count == expected:
                ok(f"collection holds {count} chunks, matching chunks_{INDEXED_VERSION}.json")
            else:
                fail(f"collection holds {count} chunks but chunks_{INDEXED_VERSION}.json has {expected}: re-run `ingest --stage index --reindex`")
                problems += 1

            if count:
                other = col.get(where={"version": "2023"}, limit=1)
                if other.get("ids"):
                    warn("2023 chunks are present in the vector index; the design expects only 2026 to be searchable by meaning")
                sample = col.get(limit=1, include=["embeddings"])
                emb = sample.get("embeddings")
                if emb is not None and len(emb):
                    dim = len(emb[0])
                    if dim == s.EMBEDDING_DIMENSION:
                        ok(f"stored vectors are {dim}-dimensional, matching EMBEDDING_DIMENSION")
                    else:
                        fail(f"stored vectors are {dim}-d but EMBEDDING_DIMENSION is {s.EMBEDDING_DIMENSION}: the index was built with another model/config")
                        problems += 1
    except Exception as e:
        fail(f"Chroma check failed: {str(e)[:160]}")
        info("from the host the compose port is usually 8001; inside the compose network it is chroma:8000")
        problems += 1

    if args.deep:
        heading("Live API calls")
        try:
            from src.vector_store.embeddings import get_embedding_function

            vec = get_embedding_function().embed_query("opt-out preference signal")
            if len(vec) == s.EMBEDDING_DIMENSION:
                ok(f"embedding call works ({len(vec)}-d)")
            else:
                fail(f"embedding returned {len(vec)}-d, expected {s.EMBEDDING_DIMENSION}")
                problems += 1
        except Exception as e:
            fail(f"embedding call failed: {str(e)[:160]}")
            problems += 1
        try:
            from src.generation.llm_service import get_llm, _get_text_content

            reply = _get_text_content(get_llm(temperature=0).invoke("Reply with the single word: pong").content)
            ok(f"LLM call works ({reply.strip()[:20]!r})")
        except Exception as e:
            fail(f"LLM call failed: {str(e)[:160]}")
            problems += 1

    heading("Langfuse")
    if s.LANGFUSE_ENABLED and s.LANGFUSE_PUBLIC_KEY:
        try:
            with urllib.request.urlopen(s.LANGFUSE_HOST.rstrip("/") + "/api/public/health", timeout=3) as r:
                ok(f"{s.LANGFUSE_HOST} reachable (HTTP {r.status})")
        except Exception as e:
            warn(f"{s.LANGFUSE_HOST} not reachable ({str(e)[:80]}); tracing will be skipped, the app still works")
    else:
        info("tracing disabled or keys not set")

    print()
    if problems:
        fail(f"{problems} problem(s) found")
        return 1
    ok("everything looks good" + ("" if args.deep else " (use --deep to also test the embedding and LLM APIs)"))
    return 0


# --------------------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload, app_dir=str(ROOT))
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="ComplianceLens: build the index, ask questions, evaluate and serve.",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    ing = sub.add_parser("ingest", help="parse the PDF, chunk both versions, index 2026 into Chroma")
    ing.add_argument("--stage", choices=["all", "parse", "chunk", "index"], default="all")
    ing.add_argument("--pdf", help="path to the redlined PDF (default: data/raw/*.pdf)")
    ing.add_argument("--reindex", action="store_true", help="delete the 2026 chunks from Chroma before indexing")
    ing.set_defaults(func=cmd_ingest)

    st = sub.add_parser("stats", help="chunk, parent and glossary statistics per version")
    st.set_defaults(func=cmd_stats)

    ask = sub.add_parser("ask", help="ask one question through the full pipeline")
    ask.add_argument("question", nargs="+", help="the question (quotes optional)")
    ask.add_argument("--json", action="store_true", help="print the full response as JSON only")
    ask.add_argument("-v", "--verbose", action="store_true", help="show snippets, scores and the abstention reasoning")
    ask.add_argument("--llm-router", action="store_true", help="use the LLM router instead of the heuristic one")
    ask.add_argument("--tag", action="append", help="Langfuse tag (repeatable)")
    ask.set_defaults(func=cmd_ask)

    ev = sub.add_parser("eval", add_help=False, help="run eval/eval_harness.py (all other flags are forwarded)")
    ev.set_defaults(func=cmd_eval, passthrough=True)

    sw = sub.add_parser("sweep", add_help=False, help="run eval/eval_retrieval_tuning.py --sweep (flags forwarded)")
    sw.set_defaults(func=cmd_sweep, passthrough=True)

    doc = sub.add_parser("doctor", help="check configuration, processed files, Chroma and Langfuse")
    doc.add_argument("--deep", action="store_true", help="also make one real embedding call and one LLM call")
    doc.set_defaults(func=cmd_doctor)

    sv = sub.add_parser("serve", help="start the FastAPI server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=cmd_serve)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    if getattr(args, "passthrough", False):
        return args.func(args, extra)
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())