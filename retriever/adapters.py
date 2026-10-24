"""
retriever/adapters.py
=====================
Wraps the real retrieval engines into the Registry Pattern required by the Router.
Converts native outputs (LlamaIndex nodes, dicts) into RetrievedChunk dataclasses.
"""

from shared.models import StrategyCall, RetrievedChunk

def make_vector_adapter(vector_index):
    """Adapter for vector_retriever.py"""
    def retrieve(sc: StrategyCall) -> list[RetrievedChunk]:
        # Use LlamaIndex retriever
        retriever = vector_index.as_retriever(similarity_top_k=sc.top_k)
        nodes = retriever.retrieve(sc.query_text)
        
        chunks = []
        for i, node in enumerate(nodes):
            chunks.append(RetrievedChunk(
                chunk_id=node.node_id,
                content=node.text,
                source=node.metadata.get("source", "unknown"),
                page_number=node.metadata.get("page_label", 1), # Fallback to 1 if not a PDF
                score=float(node.score) if node.score else 0.0,
                rank=i + 1,
                strategy="vector",
                metadata=node.metadata
            ))
        return chunks
    return retrieve

def make_keyword_adapter(bm25_index, bm25_chunks):
    from retriever.keyword_retriever import keyword_retrieve

    def retrieve(sc: StrategyCall) -> list[RetrievedChunk]:
        results = keyword_retrieve(
            query=sc.query_text,
            bm25=bm25_index,
            chunks=bm25_chunks,
            top_k=sc.top_k,
            min_score=0.0
        )

        chunks = []
        for i, res in enumerate(results):
            # Handle both plain dict (correct) and accidental RetrievedChunk (stale pickle)
            if isinstance(res, dict):
                chunk_id = res.get("node_id", f"bm25_{i}")
                content  = res["text"]
                source   = res["source"]
                score    = res["score"]
                meta     = {"chunk_index": res["chunk_index"],
                            "total_chunks": res["total_chunks"]}
            else:
                # res is already a RetrievedChunk — pass through directly
                chunks.append(res)
                continue

            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=content,
                source=source,
                page_number=1,
                score=score,
                rank=i + 1,
                strategy="bm25",
                metadata=meta
            ))
        return chunks
    return retrieve

def make_hierarchical_adapter(hierarchical_index):
    """Adapter for hierarchical_retriever.py"""
    # Import the real retrieval function
    from retriever.hierarchical_retriever import hierarchical_retrieve
    from rag_config import RAGConfig
    
    # We create a dummy config just to pass the top_k to your existing function
    cfg = RAGConfig()
    
    def retrieve(sc: StrategyCall) -> list[RetrievedChunk]:
        cfg.similarity_top_k = sc.top_k
        results = hierarchical_retrieve(
            query=sc.query_text, 
            index=hierarchical_index, 
            cfg=cfg
        )
        
        chunks = []
        for i, res in enumerate(results):
            chunks.append(RetrievedChunk(
                # Use a hash of the text as an ID since hierarchy nodes might not have one easily accessible here
                chunk_id=f"hier_{hash(res['text']) % 1000000}", 
                content=res["text"],
                source=res["source"],
                page_number=1,
                score=res["score"],
                rank=i + 1,
                strategy="hierarchical",
                metadata={"level": res["level"]}
            ))
        return chunks
    return retrieve


def make_graph_adapter(graph_pipeline):

    def retrieve(sc: StrategyCall):
        print("🔥 GRAPH RETRIEVER CALLED")
        return graph_pipeline.retrieve(

            query=sc.query_text,

            strategy_call=sc

        )

    return retrieve
