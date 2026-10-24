"""
generator.py
Connects retrieved chunks to Gemini and generates a final answer.
"""

import os
import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ==========================================
# X-RAY DEBUG HELPER
# ==========================================
def print_xray_stage(stage_name: str, data: str):
    """Helper to print highly visible debug dividers"""
    print(f"\n{'-'*80}")
    print(f"🔍 STAGE: {stage_name.upper()}")
    print(f"{'-'*80}")
    print(data)
    print(f"{'-'*80}\n")


# ---------------------------
# Fallback LLM (Ollama)
# ---------------------------
def call_fallback_llm(prompt: str, model: str = "llama3") -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        return response.json().get("response", "").strip()

    except Exception as e:
        return f"❌ Fallback LLM error: {str(e)}"


# ---------------------------
# Config
# ---------------------------
@dataclass
class GeneratorConfig:
    api_key: str = os.getenv("LLM_API_KEY")   # ✅ FIXED
    model_name: str = "gemini-3.1-flash-lite"

    fallback_model: str = "llama3"
    use_fallback: bool = True

    max_output_tokens: int = 4192
    temperature: float = 0.2
    min_score_threshold: float = 0.25
    hierarchical_score_threshold: float = 0.25
    min_chunks_required: int = 1


# ---------------------------
# Convert Nodes
# ---------------------------
def convert_nodes_to_chunks(nodes: list) -> list[dict]:
    chunks = []
    for node in nodes:
        chunks.append({
            "text": node.node.text,
            "score": round(node.score, 4),
            "source": node.node.metadata.get("source", "unknown"),
            "chunk_index": node.node.metadata.get("chunk_index", -1),
            "total_chunks": node.node.metadata.get("total_chunks", -1),
            "retriever": "vector",
        })
    return chunks


# ---------------------------
# Score filter
# ---------------------------
def filter_chunks_by_score(chunks: list, min_score: float) -> list:
    filtered = []
    for c in chunks:
        # dict path
        if isinstance(c, dict):
            score = c.get("score", 0.0)
        else:
            # RetrievedChunk: prefer rerank_score from metadata if present
            score = c.metadata.get("rerank_score") 
            if score is None:
                score = getattr(c, "score", 0.0)

        if score >= min_score:
            filtered.append(c)

    removed = len(chunks) - len(filtered)
    if removed > 0:
        print(f"  ⚠️ Filtered out {removed} chunk(s) below score {min_score}")

    return filtered


# ---------------------------
# Prompt builder (ENHANCED FOR ANTI-HALLUCINATION)
# ---------------------------
def build_prompt(question: str, chunks: list) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        # Handle both RetrievedChunk objects and legacy plain dicts
        if isinstance(chunk, dict):
            text   = chunk.get("text", "").strip()
            score  = chunk.get("score", 0.0)
            source = chunk.get("source", "unknown")
            chunk_index = chunk.get("chunk_index", i)
        else:
            text   = getattr(chunk, "content", "").strip()
            score  = getattr(chunk, "score", 0.0)
            source = getattr(chunk, "source", "unknown")
            chunk_index = chunk.metadata.get("chunk_index", i)

        if not text:            # skip truly empty chunks
            continue

        context_parts.append(
            f"[Chunk {chunk_index} | Source: {source} | Score: {score:.4f}]\n{text}"
        )

    context_block = "\n\n".join(context_parts)

    return f"""You are an expert legal and document assistant. Your task is to answer the user's question based STRICTLY and EXCLUSIVELY on the provided context blocks.

CRITICAL RULES:
1. NO EXTERNAL KNOWLEDGE: Do not use any prior knowledge, internet search, or assumptions. 
2. STRICT GROUNDING: If the answer cannot be completely derived from the text below, you must reply EXACTLY with: "I could not find enough information in the document to answer this." Do not attempt to guess or partially answer if the core information is missing.
3. SYNTHESIZE MULTIPLE SOURCES: You are receiving up to 15 chunks from different legal documents. Synthesize the answer accurately. If different laws apply to the query (e.g., IT Act vs. DPDP Act), specify which rule comes from which act.
4. ACCURACY OVER LENGTH: Be precise and factual. Do not add conversational filler.

CONTEXT:
=========================
{context_block}
=========================

QUESTION: {question}

ANSWER:
"""


# ---------------------------
# Generator
# ---------------------------
class Generator:

    def __init__(self, cfg: GeneratorConfig = None):
        self.cfg = cfg or GeneratorConfig()
        self._setup_gemini()

    def _setup_gemini(self):
        if not self.cfg.api_key:
            raise ValueError("❌ GEMINI_API_KEY missing in .env")

        self.client = genai.Client(api_key=self.cfg.api_key)
        print(f"  ✅ Gemini connected — {self.cfg.model_name}")

    def generate(
        self,
        question: str,
        chunks: list[dict],
        score_threshold: float = None,
    ) -> tuple[str, list[dict]]:

        threshold = score_threshold or self.cfg.min_score_threshold

        print(f"\n{'─'*50}")
        print(f"🔍 Question: {question}")
        print(f"📦 Chunks received: {len(chunks)}")

        # Step 1 — filter
        filtered = filter_chunks_by_score(chunks, threshold)

        if len(filtered) < self.cfg.min_chunks_required:
            return "I could not find relevant information.", []

        print(f"✅ {len(filtered)} chunk(s) passed filter")

        # --- X-RAY: PRINT CHUNKS GOING TO LLM ---
        debug_chunks = ""
        for i, chunk in enumerate(filtered[:5]): # Print top 5 to terminal for monitoring
            text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "content", "")
            source = chunk.get("source", "unknown") if isinstance(chunk, dict) else getattr(chunk, "source", "unknown")
            debug_chunks += f"Rank {i+1} | Source: {source}\n{text[:200]}...\n\n"
        print_xray_stage("TOP 5 CHUNKS PASSED TO LLM", debug_chunks)


        # Step 2 — prompt
        prompt = build_prompt(question, filtered)
        
        # --- X-RAY: PRINT FINAL PROMPT ---
        print_xray_stage("FINAL PROMPT (Truncated)", prompt[:1000] + "\n\n...[TRUNCATED]")

        # Step 3 — Gemini API Call (RESTORED)
        print(f"🤖 Calling Gemini API...")

        try:
            response = self.client.models.generate_content(
                model=self.cfg.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=self.cfg.max_output_tokens,
                    temperature=self.cfg.temperature,
                )
            )

            answer = (response.text or "").strip()

            if not answer:
                raise ValueError("Empty Gemini response")

            print("✅ Gemini success")
            return answer + "\n\n[Gemini]", filtered

        except Exception as e:
            print(f"⚠️ Gemini failed: {str(e)}")

            # Step 4 — Fallback
            if self.cfg.use_fallback:
                print(f"🔄 Switching to {self.cfg.fallback_model}")

                try:
                    answer = call_fallback_llm(prompt, self.cfg.fallback_model)

                    if not answer:
                        raise ValueError("Empty fallback response")

                    print("✅ Fallback success")
                    return answer + "\n\n[Llama3 fallback]", filtered

                except Exception as fallback_error:
                    print(f"❌ Fallback failed: {str(fallback_error)}")
                    return "Both Gemini and fallback failed.", []

            return f"❌ Gemini error: {str(e)}", []

    def show_prompt(self, question: str, chunks: list[dict]):
        filtered = filter_chunks_by_score(chunks, self.cfg.min_score_threshold)
        prompt = build_prompt(question, filtered)
        print(prompt)


# ---------------------------
# Test
# ---------------------------
if __name__ == "__main__":
    from retriever.vector_retriever import load_index, retrieve_chunks
    from ingestion.chunking import RAGConfig

    cfg = RAGConfig()
    index = load_index(cfg)

    q = "What is ReTreever?"

    nodes = retrieve_chunks(q, index, cfg)
    chunks = convert_nodes_to_chunks(nodes)

    gen = Generator()
    ans, src = gen.generate(q, chunks)

    print("\nANSWER:\n", ans)
