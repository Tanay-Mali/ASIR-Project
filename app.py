# app.py
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import uuid
import datetime
import threading
import time

# Import your pipeline components
from classifier.query_classifier import QueryClassifier
from router.router import Router
from reranker.reranker import Reranker
from rag_config import RAGConfig
from generation.generator import Generator, build_prompt, filter_chunks_by_score
from google.genai import types

# Import your real database loaders
from retriever.vector_retriever import load_index as load_vector_index
from retriever.keyword_retriever import load_keyword_index
from retriever.hierarchical_retriever import load_hierarchical_index
from graph_retriever.pipeline import GraphRetrieverPipeline
from config.core_config import GRAPH_PICKLE_PATH
from classifier.xgboost_classifier import check_and_retrain_xgboost
from feedback.failure_classifier import classify_failure
from feedback.label_generator import generate_training_label
from feedback.logger import (
    load_chunks_for_query,
    load_query_signals,
    log_explicit_feedback,
    log_knowledge_gap,
    log_query_signals,
    log_user_error_query,
    save_training_label,
    update_feedback_diagnostics,
)
from feedback.semantic_audit import run_semantic_audit
from feedback.signal_collector import QuerySignals, auto_signals
from router.weight_store import WeightUpdateConfig, apply_credit_attribution, apply_weekly_decay

# Import the adapters
from retriever.adapters import (
    make_vector_adapter, 
    make_keyword_adapter, 
    make_hierarchical_adapter, 
    make_graph_adapter
)


# ==========================================
# X-RAY DEBUG HELPER
# ==========================================
def print_xray_stage(stage_name: str, data: str):
    """Helper to print highly visible debug dividers in the terminal."""
    print(f"\n{'-'*80}")
    print(f"🔍 STAGE: {stage_name.upper()}")
    print(f"{'-'*80}")
    print(data)
    print(f"{'-'*80}\n")


app = FastAPI(title="ASIR-RAG API")

# Serve the background image as a static asset
@app.get("/bg-vortex.jpg")
async def serve_bg_image():
    return FileResponse("WhatsApp Image 2026-06-18 at 20.32.14.jpeg", media_type="image/jpeg")

# ────────────────────────────────────────────────────────────────────────
# 1. Load Databases at Startup
# ────────────────────────────────────────────────────────────────────────
print("Initializing ASIR-RAG Core Pipeline...")
cfg = RAGConfig()

print("Loading Vector DB...")
v_index = load_vector_index(cfg)

print("Loading Keyword DB...")
bm25_idx, bm25_chks = load_keyword_index(cfg.keyword_dir)

print("Loading Hierarchical DB...")
h_index = load_hierarchical_index(cfg)

print("Loading Knowledge Graph...")
graph_pipeline = GraphRetrieverPipeline(GRAPH_PICKLE_PATH)
real_graph_retriever = make_graph_adapter(graph_pipeline)

# 2. Wrap them in adapters
real_vector_retriever = make_vector_adapter(v_index)
real_bm25_retriever   = make_keyword_adapter(bm25_idx, bm25_chks)
real_hier_retriever   = make_hierarchical_adapter(h_index)


# ────────────────────────────────────────────────────────────────────────
# 3. Initialize Pipeline Engines
# ────────────────────────────────────────────────────────────────────────
classifier = QueryClassifier()
reranker = Reranker()
llm_generator = Generator()  

router = Router(
    retrieval_registry={
        "vector":       real_vector_retriever,
        "bm25":         real_bm25_retriever,
        "hierarchical": real_hier_retriever,
        "graph":        real_graph_retriever,
    },
    reranker=reranker
)

print("Pipeline Ready!")
EMBEDDER = graph_pipeline._embedder
_QUERY_CONTEXT: dict[str, dict] = {}

def _compute_chunk_ranks(chunks) -> dict:
    out = {}
    for i, c in enumerate(chunks or []):
        strat = getattr(c, "strategy", None)
        if not strat:
            continue
        rank = i + 1
        if strat not in out or rank < out[strat]:
            out[strat] = rank
    return out


def _weekly_decay_worker():
    while True:
        try:
            time.sleep(7 * 24 * 60 * 60)
            apply_weekly_decay(router._weights, WeightUpdateConfig())
        except Exception:
            continue


threading.Thread(target=_weekly_decay_worker, daemon=True).start()

# ────────────────────────────────────────────────────────────────────────
# Data Models & Endpoints
# ────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str = "web_user"

class FeedbackRequest(BaseModel):
    query_id: str
    rating: str
    failure_reason: str = "none"
    query_type: str | None = None
    strategies_used: list[str] | None = None

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the base HTML frontend."""
    try:
        with open("asir base.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: 'asir base.html' not found in root directory.</h1>"

# ────────────────────────────────────────────────────────────────────────
# SSE Streaming Endpoint
# ────────────────────────────────────────────────────────────────────────

@app.post("/api/stream")
async def stream_pipeline(request: Request):
    """Streams pipeline stages as newline-delimited JSON events."""
    import json as _json
    body = await request.json()
    query = body.get("query", "").strip()
    session_id = body.get("session_id", "web_user")

    if not query:
        async def _err():
            yield _json.dumps({"step": "error", "error": "Empty query"}) + "\n"
        return StreamingResponse(_err(), media_type="text/plain")

    def generate_events():
        import json
        try:
            # ── STEP 1: Classifier ──
            yield json.dumps({"step": "classifier", "status": "active"}) + "\n"
            co = classifier.classify_text(query, session_id)
            yield json.dumps({
                "step": "classifier", "status": "done",
                "query_type": co.query_type,
                "confidence": round(co.confidence, 4),
                "query_id": co.query_id,
            }) + "\n"

            # ── STEP 2: Router ──
            yield json.dumps({"step": "router", "status": "active"}) + "\n"
            decision, merged = router.route(co)
            primary_strats = [f"{s.strategy} (w: {s.weight})" for s in getattr(decision, 'primary_strategies', [])]
            yield json.dumps({
                "step": "router", "status": "done",
                "routing_mode": decision.routing_mode,
                "strategies": primary_strats,
            }) + "\n"

            # ── STEP 3: Retriever ──
            yield json.dumps({"step": "retriever", "status": "done",
                "chunks_count": len(merged.chunks),
                "strategies_used": merged.strategies_used,
            }) + "\n"

            # ── STEP 4: Reranker ──
            yield json.dumps({"step": "reranker", "status": "done",
                "top_score": round(float(getattr(merged.chunks[0], 'score', 0.0)), 4) if merged.chunks else 0,
            }) + "\n"

            # ── STEP 5: Sources ──
            sources_list = [{
                "n": getattr(c, "source", "unknown"),
                "s": str(getattr(c, "score", 0.0)),
                "path": getattr(c, "relation_paths", None),
                "snippet": getattr(c, "content", "")  # <-- CRITICAL FIX
            } for c in merged.chunks]

            yield json.dumps({
                "step": "sources",
                "sources": sources_list,
                "label": f"{decision.routing_mode} ({', '.join(merged.strategies_used)})",
                "conf": round(co.confidence * 100),
                "strategies_used": merged.strategies_used,
            }) + "\n"

            # --- X-RAY: PRINT CHUNKS TO TERMINAL ---
            debug_text = f"Total chunks after Merging/Reranking: {len(merged.chunks)}\n\n"
            for i, chunk in enumerate(merged.chunks[:5]): # Look at top 5 final chunks
                section_meta = getattr(chunk, 'section', 'N/A')
                safe_content = getattr(chunk, 'content', '')[:200].replace('\n', ' ')

                # Safe score extraction to prevent formatting crashes
                raw_score = getattr(chunk, 'final_score', None)
                if raw_score is None:
                    raw_score = getattr(chunk, 'score', 0.0)
                try:
                    display_score = float(raw_score)
                except (TypeError, ValueError):
                    display_score = 0.0

                strat = getattr(chunk, 'strategy', 'unknown')

                debug_text += f"  Rank {i+1} | Source Strat: {strat} | Final Score: {display_score:.4f}\n"
                debug_text += f"  File: {getattr(chunk, 'source', 'unknown')} | Section: {section_meta}\n"
                debug_text += f"  Content: {safe_content}...\n\n"

            print_xray_stage("STREAMING ENDPOINT: TOP CHUNKS", debug_text)

            # ── STEP 6: Generator (streaming) ──
            yield json.dumps({"step": "generator", "status": "active"}) + "\n"

            if decision.routing_mode == "direct_llm":
                # Direct LLM — use streaming
                try:
                    stream_response = llm_generator.client.models.generate_content_stream(
                        model=llm_generator.cfg.model_name,
                        contents=query
                    )
                    full_answer = ""
                    for chunk in stream_response:
                        token = chunk.text or ""
                        if token:
                            full_answer += token
                            yield json.dumps({"step": "generator", "status": "streaming", "token": token}) + "\n"
                    ans = full_answer.strip()
                    used_sources = []
                except Exception as e:
                    ans = f"Error: {str(e)}"
                    used_sources = []
                    yield json.dumps({"step": "generator", "status": "streaming", "token": ans}) + "\n"
            else:
                # RAG path — build prompt, stream from Gemini
                generator_chunks = [{
                    "text": c.content,
                    "score": c.score,
                    "source": c.source,
                    "chunk_index": c.metadata.get("chunk_index", -1) if hasattr(c, 'metadata') else -1
                } for c in merged.chunks]

                filtered = filter_chunks_by_score(generator_chunks, -999.0)
                prompt = build_prompt(query, filtered)

                try:
                    stream_response = llm_generator.client.models.generate_content_stream(
                        model=llm_generator.cfg.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            max_output_tokens=llm_generator.cfg.max_output_tokens,
                            temperature=llm_generator.cfg.temperature,
                        )
                    )
                    full_answer = ""
                    for chunk in stream_response:
                        token = chunk.text or ""
                        if token:
                            full_answer += token
                            yield json.dumps({"step": "generator", "status": "streaming", "token": token}) + "\n"
                    ans = full_answer.strip()
                    used_sources = filtered
                except Exception as e:
                    # Gemini failed — try fallback (non-streaming)
                    if llm_generator.cfg.use_fallback:
                        from generation.generator import call_fallback_llm
                        ans = call_fallback_llm(prompt, llm_generator.cfg.fallback_model)
                        yield json.dumps({"step": "generator", "status": "streaming", "token": ans}) + "\n"
                        used_sources = filtered
                    else:
                        ans = f"Error: {str(e)}"
                        yield json.dumps({"step": "generator", "status": "streaming", "token": ans}) + "\n"
                        used_sources = []

            yield json.dumps({"step": "generator", "status": "done", "query_id": co.query_id}) + "\n"

            # ── Signal logging (async, non-blocking to stream) ──
            try:
                chunk_ranks = _compute_chunk_ranks(merged.chunks)
                signals = auto_signals(query, ans, merged.chunks, EMBEDDER, co.query_type)
                qs = QuerySignals(
                    query_id=co.query_id,
                    session_id=session_id,
                    query_text=query,
                    timestamp=datetime.datetime.utcnow().isoformat(),
                    query_type=co.query_type,
                    classifier_used=getattr(co, "classified_by", "unknown"),
                    classifier_conf=co.confidence,
                    routing_mode=decision.routing_mode,
                    strategies_used=merged.strategies_used,
                    fallback_triggered=merged.fallback_used,
                    top_chunk_scores=[float(getattr(c, "score", 0.0)) for c in merged.chunks],
                    reranker_top=float(getattr(merged.chunks[0], "score", 0.0)) if merged.chunks else 0.0,
                    reranker_gap=(
                        float(getattr(merged.chunks[0], "score", 0.0)) -
                        float(getattr(merged.chunks[1], "score", 0.0))
                    ) if len(merged.chunks) > 1 else 0.0,
                    chunks_filtered=0,
                    chunk_ranks=chunk_ranks,
                    llm_not_found=signals["llm_not_found"],
                    grounding_score=signals["grounding_score"],
                    citation_valid=signals["citation_valid"],
                    uncited_answer=signals["uncited_answer"],
                    hallucinated_refs=signals["hallucinated_refs"],
                    answer_text=ans,
                    qa_similarity=signals["qa_similarity"],
                    max_chunk_sim=signals["max_chunk_sim"],
                    mean_chunk_sim=signals["mean_chunk_sim"],
                    max_query_chunk_sim=signals["max_query_chunk_sim"],
                )
                log_query_signals({
                    **qs.__dict__,
                    "chunks_json": [
                        {
                            "content": getattr(c, "content", ""),
                            "source": getattr(c, "source", "unknown"),
                            "score": float(getattr(c, "score", 0.0)),
                            "strategy": getattr(c, "strategy", ""),
                        }
                        for c in (merged.chunks or [])
                    ],
                })
                _QUERY_CONTEXT[co.query_id] = {"chunks": merged.chunks}
            except Exception:
                pass

            yield json.dumps({"step": "done"}) + "\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"step": "error", "error": str(e)}) + "\n"

    return StreamingResponse(generate_events(), media_type="text/plain")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Processes the query and returns the expected JSON for the UI."""
    try:
        co = classifier.classify_text(req.query, req.session_id)
        decision, merged = router.route(co)
        
        # ────────────────────────────────────────────────────────────────────────
        # X-RAY STAGE 1: ROUTER DECISION
        # ────────────────────────────────────────────────────────────────────────
        # Extract the strategies the router decided to use
        primary_strats = [f"{s.strategy} (w: {s.weight})" for s in getattr(decision, 'primary_strategies', [])]
        print_xray_stage("Router Decision", 
            f"Query: '{req.query}'\n"
            f"Classified Type: {co.query_type} (Confidence: {co.confidence:.2f})\n"
            f"Routing Mode: {decision.routing_mode}\n"
            f"Selected Strategies: {primary_strats}"
        )

        # ────────────────────────────────────────────────────────────────────────
        # X-RAY STAGE 3: POST-MERGER FINAL RANKING
        # ────────────────────────────────────────────────────────────────────────
        debug_text = f"Total chunks after Merging/Reranking: {len(merged.chunks)}\n\n"
        for i, chunk in enumerate(merged.chunks[:5]): # Look at top 5 final chunks
            section_meta = getattr(chunk, 'section', 'N/A')
            
            # Handle newline replacement outside the f-string 
            safe_content = chunk.content[:200].replace('\n', ' ')
            
            # ✅ FIXED: Safely handle missing final_score
            display_score = chunk.final_score if chunk.final_score is not None else chunk.score
            
            debug_text += f"  Rank {i+1} | Source Strat: {chunk.strategy} | Final Score: {display_score:.4f}\n"
            debug_text += f"  File: {chunk.source} | Section: {section_meta}\n"
            debug_text += f"  Content: {safe_content}...\n\n"
            
        print_xray_stage("Post-Merger Final Ranking", debug_text)

        # ────────────────────────────────────────────────────────────────────────
        # 4. Generate the Final Answer 
        # ────────────────────────────────────────────────────────────────────────
        if decision.routing_mode == "direct_llm":
            # For simple factual queries, bypass RAG and ask Gemini directly
            response = llm_generator.client.models.generate_content(
                model=llm_generator.cfg.model_name,
                contents=req.query
            )
            ans = response.text.strip()
            used_sources = []
        else:
            # Convert the Router's chunks into the dictionary format the generator expects
            generator_chunks = [{
                "text": c.content,
                "score": c.score,
                "source": c.source,
                "chunk_index": c.metadata.get("chunk_index", -1) if hasattr(c, 'metadata') else -1
            } for c in merged.chunks]

            # Pass to generator. Set threshold extremely low (-999.0) because 
            # our Cross-Encoder Re-ranker already handled quality control!
            ans, used_sources = llm_generator.generate(
                question=req.query, 
                chunks=generator_chunks, 
                score_threshold=-999.0
            )
            print("\n--- DEBUG: FULL ANSWER ---")
            print(ans)
            print("--------------------------\n")

        try:
            chunk_ranks = _compute_chunk_ranks(merged.chunks)
            signals = auto_signals(req.query, ans, merged.chunks, EMBEDDER, co.query_type)
            qs = QuerySignals(
                query_id=co.query_id,
                session_id=req.session_id,
                query_text=req.query,
                timestamp=datetime.datetime.utcnow().isoformat(),
                query_type=co.query_type,
                classifier_used=getattr(co, "classified_by", "unknown"),
                classifier_conf=co.confidence,
                routing_mode=decision.routing_mode,
                strategies_used=merged.strategies_used,
                fallback_triggered=merged.fallback_used,
                top_chunk_scores=[float(getattr(c, "score", 0.0)) for c in merged.chunks],
                reranker_top=float(getattr(merged.chunks[0], "score", 0.0)) if merged.chunks else 0.0,
                reranker_gap=(
                    float(getattr(merged.chunks[0], "score", 0.0)) -
                    float(getattr(merged.chunks[1], "score", 0.0))
                ) if len(merged.chunks) > 1 else 0.0,
                chunks_filtered=0,
                chunk_ranks=chunk_ranks,
                llm_not_found=signals["llm_not_found"],
                grounding_score=signals["grounding_score"],
                citation_valid=signals["citation_valid"],
                uncited_answer=signals["uncited_answer"],
                hallucinated_refs=signals["hallucinated_refs"],
                answer_text=ans,
                qa_similarity=signals["qa_similarity"],
                max_chunk_sim=signals["max_chunk_sim"],
                mean_chunk_sim=signals["mean_chunk_sim"],
                max_query_chunk_sim=signals["max_query_chunk_sim"],
            )
            log_query_signals(
                {
                    **qs.__dict__,
                    "chunks_json": [
                        {
                            "content": getattr(c, "content", ""),
                            "source": getattr(c, "source", "unknown"),
                            "score": float(getattr(c, "score", 0.0)),
                            "strategy": getattr(c, "strategy", ""),
                        }
                        for c in (merged.chunks or [])
                    ],
                }
            )
            _QUERY_CONTEXT[co.query_id] = {"chunks": merged.chunks}
        except Exception:
            pass
 
        return {
            "query_id": co.query_id,
            "type": co.query_type,
            "label": f"{decision.routing_mode} ({', '.join(merged.strategies_used)})",
            "conf": round(co.confidence * 100),
            "sources": [{

                "n": c.source,

                "s": str(c.score),

                "path": c.relation_paths if hasattr(c, "relation_paths") else None

            } for c in merged.chunks],
            "ans": ans,
            "strategies_used": merged.strategies_used
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """
    Stores explicit feedback and runs failure classification / attribution pipeline.
    """
    try:
        ts = datetime.datetime.utcnow().isoformat()
        log_explicit_feedback(req.query_id, req.rating, req.failure_reason, ts)
        rating_value = req.rating if req.rating in {"helpful", "not_helpful"} else "not_helpful"
        reason_value = (
            req.failure_reason
            if req.failure_reason in {"irrelevant", "incomplete", "not_in_doc", "wrong_citation", "none"}
            else "none"
        )

        stored = load_query_signals(req.query_id)
        if stored is None:
            return {"status": "accepted", "note": "query_signals_not_found"}

        chunks = load_chunks_for_query(req.query_id)
        if not chunks:
            chunks = _QUERY_CONTEXT.get(req.query_id, {}).get("chunks", [])
        audit = run_semantic_audit(stored.__dict__)

        failure_class, confidence = classify_failure(
            rating=rating_value,
            failure_reason=reason_value,
            llm_not_found=stored.llm_not_found,
            grounding_score=stored.grounding_score,
            citation_valid=stored.citation_valid,
            uncited_answer=stored.uncited_answer,
            qa_similarity=audit["qa_similarity"],
            max_chunk_sim=audit["max_chunk_sim"],
            rapid_rephrase=getattr(stored, "rapid_rephrase", False),
            dwell_time=getattr(stored, "dwell_time_seconds", 0.0),
            fallback_used=getattr(stored, "fallback_triggered", False),
        )

        apply_credit_attribution(
            failure_class=failure_class,
            confidence=confidence,
            strategies_used=getattr(stored, "strategies_used", []),
            chunk_ranks=getattr(stored, "chunk_ranks", {}),
            weight_store=router._weights,
            query_type=stored.query_type,
            config=WeightUpdateConfig(),
        )

        label = generate_training_label(
            failure_class=failure_class,
            confidence=confidence,
            query_text=stored.query_text,
            query_type=stored.query_type,
            signals=stored.__dict__,
        )
        if label.get("label") is not None:
            save_training_label(label)
            check_and_retrain_xgboost()

        update_feedback_diagnostics(
            query_id=req.query_id,
            failure_class=failure_class,
            label_confidence=confidence,
            qa_similarity=audit["qa_similarity"],
            max_chunk_sim=audit["max_chunk_sim"],
            mean_chunk_sim=audit["mean_chunk_sim"],
            true_query_type=label.get("label") if isinstance(label, dict) else None,
        )

        if failure_class == "system_error_knowledge_gap":
            log_knowledge_gap(stored.query_type, stored.query_text, req.query_id)
        if failure_class == "user_error":
            log_user_error_query(stored.query_text, stored.query_type)

        return {"status": "success", "failure_class": failure_class, "confidence": round(confidence, 4)}
    except Exception as e:
        import logging
        logging.error("Feedback pipeline error: %s", e)
        return {"status": "ok", "note": "feedback logged with errors"}

# ── TEMPORARY DASHBOARD — remove this block to disable ───────────────
from fastapi.responses import HTMLResponse
from admin.db_reader import (
    get_feedback_summary, get_recent_feedback, get_grounding_distribution,
    get_knowledge_gaps, get_label_stats, get_current_weights, get_weight_history
)


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    with open("admin/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/admin/data")
async def admin_data():
    return {
        "summary":         get_feedback_summary(),
        "recent_feedback": get_recent_feedback(20),
        "grounding_dist":  get_grounding_distribution(),
        "knowledge_gaps":  get_knowledge_gaps(10),
        "label_stats":     get_label_stats(),
        "current_weights": get_current_weights(),
        "weight_history":  get_weight_history(50),
    }


# ── END TEMPORARY DASHBOARD ───────────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
