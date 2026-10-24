"""
main.py
───────
Entry point for the RAG pipeline.

Run this file to ask questions against your documents.

Usage:
    python main.py
"""

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from retriever.vector_retriever import load_index, retrieve_chunks
from retriever.keyword_retriever import keyword_retrieve
# from retriever.hierarchical_retriever import retrieve
from ingestion.keyword_indexer import load_keyword_index
from generation.generator import Generator, GeneratorConfig, convert_nodes_to_chunks
from retriever.hierarchical_retriever import load_hierarchical_index, hierarchical_retrieve
from config import RAGConfig
from ingestion.chunking import get_embed_model

# ---------------------------
# Retriever context — all resources in one object
# ---------------------------

@dataclass
class RetrieverContext:
    index: any
    bm25: any
    keyword_chunks: list
    hierarchical_index: any    # ← changed from hierarchical_collection

# ---------------------------
# Conversation history
# ---------------------------

class ConversationHistory:
    def __init__(self, max_turns: int = 5):
        self.history   = []
        self.max_turns = max_turns

    def add(self, question: str, answer: str) -> None:
        self.history.append({
            "question": question,
            "answer": answer
        })
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self) -> str:
        if not self.history:
            return ""
        parts = []
        for i, turn in enumerate(self.history):
            parts.append(
                f"Previous Q{i+1}: {turn['question']}\n"
                f"Previous A{i+1}: {turn['answer'][:300]}..."
            )
        return "CONVERSATION HISTORY (for context only):\n" + "\n\n".join(parts)

    def is_empty(self) -> bool:
        return len(self.history) == 0

    def clear(self) -> None:
        self.history = []
        print("  🗑️  Conversation history cleared.")


# ---------------------------
# Retriever selection
# ---------------------------

def select_retriever() -> str:
    print("\n" + "─" * 60)
    print("  Select retriever:")
    print("  [1] Vector       — semantic similarity")
    print("  [2] Keyword      — exact word match")
    print("  [3] Hierarchical — multi-level context")
    print("─" * 60)

    while True:
        choice = input("  Enter 1, 2 or 3: ").strip()
        if choice == "1":
            print("  ✅ Using Vector retriever.")
            return "vector"
        elif choice == "2":
            print("  ✅ Using Keyword retriever.")
            return "keyword"
        elif choice == "3":
            print("  ✅ Using Hierarchical retriever.")
            return "hierarchical"
        else:
            print("  ⚠️  Please enter 1, 2 or 3.")


# ---------------------------
# Single query — full pipeline
# ---------------------------

def ask(
    question: str,
    retriever_type: str,
    ctx: RetrieverContext,      # ← single object instead of 3 separate params
    generator: Generator,
    rag_cfg: RAGConfig,
    history: ConversationHistory,
) -> tuple[str, list[dict], float, float]:
    """
    Full pipeline for one question.

    Router integration point:
        When router is ready, replace the if/else block below with:
            chunks = router.route(question, ctx, rag_cfg)
        ask() signature stays exactly the same.
    """

    # Step 1 — retrieve chunks (timed)
    t0 = time.time()

    if retriever_type == "vector":
        nodes  = retrieve_chunks(question, ctx.index, rag_cfg)
        chunks = convert_nodes_to_chunks(nodes)

    elif retriever_type == "keyword":
        chunks = keyword_retrieve(
            query   = question,
            bm25    = ctx.bm25,
            chunks  = ctx.keyword_chunks,
            top_k   = rag_cfg.similarity_top_k,
            min_score = 0.0,
        )
    elif retriever_type == "hierarchical":
        chunks = hierarchical_retrieve(
            query = question,
            index = ctx.hierarchical_index,
            cfg   = rag_cfg,
        )

    retrieval_time = time.time() - t0

    if not chunks:
        return "No relevant chunks found.", [], retrieval_time, 0.0

    # Step 2 — add conversation context if history exists
    question_with_context = question
    if not history.is_empty():
        question_with_context = (
            f"{history.get_context()}\n\n"
            f"CURRENT QUESTION:\n{question}"
        )

    # Step 3 — generate answer (timed)
    # t0 = time.time()

    # answer, sources = generator.generate(question_with_context, chunks)
    # generation_time = time.time() - t0
    # Step 3 — generate answer with correct threshold per retriever
    gen_cfg=GeneratorConfig()
    t0 = time.time()

    if retriever_type == "hierarchical":
        answer, sources = generator.generate(
            question_with_context,
            chunks,
            score_threshold = gen_cfg.hierarchical_score_threshold,
        )
    else:
        answer, sources = generator.generate(
            question_with_context,
            chunks,
        )

    generation_time = time.time() - t0

    # Step 4 — save to history
    history.add(question, answer)

    return answer, sources, retrieval_time, generation_time


# ---------------------------
# Display helpers
# ---------------------------

def display_startup_banner(retriever_type: str) -> None:
    print("\n" + "═" * 60)
    print("  🤖 RAG PIPELINE — INTERACTIVE MODE")
    print(f"  Retriever : {retriever_type.upper()}")
    print("  ─────────────────────────────────")
    print("  Commands:")
    print("    'exit' or 'quit' → stop")
    print("    'switch'         → switch retriever")
    print("    'history'        → show conversation history")
    print("    'clear'          → clear conversation history")
    print("    'sources'        → toggle sources display")
    print("    'stats'          → toggle timing stats")
    print("═" * 60)


def display_result(
    answer: str,
    sources: list[dict],
    retrieval_time: float,
    generation_time: float,
    show_sources: bool,
    show_stats: bool,
    retriever_type: str,
) -> None:

    print("\n" + "═" * 60)
    print(f"  💬 ANSWER  [via {retriever_type} retriever]")
    print("═" * 60)
    print(answer)

    if show_sources and sources:
        print("\n" + "─" * 60)
        print("  📚 SOURCES USED")
        print("─" * 60)
        for i, chunk in enumerate(sources):
            preview = chunk["text"][:200].replace("\n", " ").strip()
            print(f"\n  [{i+1}] File     : {chunk['source']}")
            print(f"       Chunk    : {chunk['chunk_index']} of {chunk['total_chunks']}")
            print(f"       Score    : {chunk['score']}")
            print(f"       Retriever: {chunk.strategy}")
            print(f"       Text     : {preview}...")

    if show_stats:
        print("\n" + "─" * 60)
        print("  ⏱️  STATS")
        print("─" * 60)
        print(f"  Retrieval : {retrieval_time:.2f}s")
        print(f"  Generation: {generation_time:.2f}s")
        print(f"  Total     : {retrieval_time + generation_time:.2f}s")

    print("═" * 60)


# ---------------------------
# Interactive loop
# ---------------------------

def interactive_mode(
    ctx: RetrieverContext,
    generator: Generator,
    rag_cfg: RAGConfig,
) -> None:

    retriever_type = select_retriever()
    display_startup_banner(retriever_type)

    history      = ConversationHistory(max_turns=5)
    show_sources = True
    show_stats   = True

    while True:
        print()
        try:
            question = input("  ❓ Your question: ").strip()
        except KeyboardInterrupt:
            print("\n\n  👋 Interrupted. Goodbye!")
            break

        if not question:
            print("  ⚠️  Please type a question.")
            continue

        # Commands
        if question.lower() in ["exit", "quit", "q"]:
            print("\n  👋 Exiting. Goodbye!")
            break

        if question.lower() == "switch":
            retriever_type = select_retriever()
            history.clear()
            display_startup_banner(retriever_type)
            continue

        if question.lower() == "history":
            if history.is_empty():
                print("  ℹ️  No history yet.")
            else:
                print("\n  📜 CONVERSATION HISTORY")
                for i, turn in enumerate(history.history):
                    print(f"\n  Q{i+1}: {turn['question']}")
                    print(f"  A{i+1}: {turn['answer'][:200]}...")
            continue

        if question.lower() == "clear":
            history.clear()
            continue

        if question.lower() == "sources":
            show_sources = not show_sources
            print(f"  ℹ️  Sources {'ON' if show_sources else 'OFF'}.")
            continue

        if question.lower() == "stats":
            show_stats = not show_stats
            print(f"  ℹ️  Stats {'ON' if show_stats else 'OFF'}.")
            continue

        # Run pipeline
        try:
            answer, sources, retrieval_time, generation_time = ask(
                question       = question,
                retriever_type = retriever_type,
                ctx            = ctx,
                generator      = generator,
                rag_cfg        = rag_cfg,
                history        = history,
            )

            display_result(
                answer          = answer,
                sources         = sources,
                retrieval_time  = retrieval_time,
                generation_time = generation_time,
                show_sources    = show_sources,
                show_stats      = show_stats,
                retriever_type  = retriever_type,
            )

        except Exception as e:
            print(f"\n  ❌ Error: {str(e)}")
            print("  ℹ️  Exiting due to error.")
            break


# ---------------------------
# Main
# ---------------------------

def main():
    print("🚀 Starting RAG pipeline...")
    print("─" * 60)

    # Single config object — paths come from here
    rag_cfg = RAGConfig()
    gen_cfg = GeneratorConfig()

    # Load vector index
    print("\n📦 Loading vector index...")
    try:
        index = load_index(rag_cfg)
    except Exception as e:
        print(f"❌ Failed to load vector index: {e}")
        print("   Run ingest.py first.")
        return
    print("✅ Vector index loaded.")

    # Load keyword index
    print("\n🔑 Loading keyword index...")
    try:
        bm25, keyword_chunks = load_keyword_index(rag_cfg.keyword_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run ingest.py first.")
        return
    print("✅ Keyword index loaded.")

    # Load hierarchical index
    print("\n🏗️  Loading hierarchical index...")
    try:
        h_index = load_hierarchical_index(rag_cfg)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    except ValueError as e:
        print(f"❌ {e}")
        return
    print("✅ Hierarchical index loaded.")

    ctx = RetrieverContext(
        index              = index,
        bm25               = bm25,
        keyword_chunks     = keyword_chunks,
        hierarchical_index = h_index,
    )

    # Setup Gemini
    print("\n🤖 Setting up Gemini...")
    try:
        generator = Generator(gen_cfg)
    except ValueError as e:
        print(e)
        return
    print("✅ Gemini connected.")

    print("\n✅ All resources loaded successfully.")
    print("─" * 60)

    # Start interactive loop
    interactive_mode(ctx, generator, rag_cfg)


if __name__ == "__main__":
    main()

