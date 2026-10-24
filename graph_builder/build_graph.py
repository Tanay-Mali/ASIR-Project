"""
graph_builder/build_graph.py
==============================
Hardened Orchestrator for the ASIR-RAG Graph Construction.
Ingests text chunks, extracts entities/relations, and persists the NetworkX graph.
"""

import logging
import pickle
from pathlib import Path
from typing import List

import networkx as nx
from tqdm import tqdm

import config.core_config as C
from graph_builder.entity_extractor import extract_entities
from extraction.relationship_extractor import extract_all_relations

log = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self, output_path: str = C.GRAPH_PICKLE_PATH):
        self.output_path = Path(output_path)
        self.graph = nx.DiGraph()
        
        # [BUG FIX 2] Safe load format handling
        if self.output_path.exists():
            log.info("Loading existing graph for incremental update...")
            try:
                with open(self.output_path, "rb") as f:
                    data = pickle.load(f)
                
                loaded = data if isinstance(data, nx.DiGraph) else data.get("graph")
                if isinstance(loaded, nx.DiGraph):
                    self.graph = loaded
                else:
                    log.warning("Invalid graph format in file. Starting fresh.")
            except Exception as e:
                log.error("Failed to load existing graph: %s. Starting fresh.", e)

    def process_chunks(self, chunks: List[dict]):
        """
        Processes a list of text chunks. 
        Expects chunk format: {"id": str, "text": str, "metadata": dict, "legal_citations": list}
        """
        log.info("Starting graph construction for %d chunks...", len(chunks))

        for chunk in tqdm(chunks, desc="Building Graph"):
            text = chunk.get("text", "")
            chunk_id = chunk.get("id", "unknown")
            # ✅ ADDED: Pull citations from the chunk (passed down from document_loader)
            citations = chunk.get("legal_citations", []) 

            # 1. Extract Entities
            entities = extract_entities(text)
            if not entities:
                continue

            # 2. Add/Update Nodes
            for ent in entities:
                if not self.graph.has_node(ent.canonical):
                    self.graph.add_node(
                        ent.canonical, 
                        label=ent.label, 
                        chunk_ids=[], 
                        chunk_texts=[],
                        legal_citations=[]  # ✅ ADDED: Initialize citations list
                    )
                else:
                    if "label" not in self.graph.nodes[ent.canonical]:
                        self.graph.nodes[ent.canonical]["label"] = ent.label
                
                node_data = self.graph.nodes[ent.canonical]
                
                if chunk_id not in node_data["chunk_ids"]:
                    node_data["chunk_ids"].append(chunk_id)
                    node_data["chunk_texts"].append(text[:500]) 
                    
                    # ✅ ADDED: Safely append new citations without duplicating
                    existing_citations = node_data.setdefault("legal_citations", [])
                    for cit in citations:
                        if cit not in existing_citations:
                            existing_citations.append(cit)

            # 3. Extract and Add Relations
            relations = extract_all_relations(text, entities)
            for rel in relations:
                u, v = rel.subject, rel.object_
                
                if u == v:
                    continue
                
                if self.graph.has_edge(u, v):
                    new_weight = self.graph[u][v]["edge_weight"] + (rel.confidence * 0.5)
                    self.graph[u][v]["edge_weight"] = min(new_weight, 5.0)
                    self.graph[u][v]["frequency"] += 1
                    
                    if "methods" in self.graph[u][v]:
                        self.graph[u][v]["methods"].add(rel.method)
                    else:
                        self.graph[u][v]["methods"] = {rel.method}
                else:
                    self.graph.add_edge(
                        u, v, 
                        relation=rel.relation, 
                        edge_weight=rel.confidence, 
                        frequency=1,
                        methods={rel.method}
                    )
            
            log.debug("Processed chunk %s → %d entities, %d relations", 
                      chunk_id, len(entities), len(relations))

        self._save_graph()

    def _save_graph(self):
        """Persists the graph to disk."""
        log.info("Saving graph to %s...", self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "graph": self.graph,
            "version": "2.0",
            "domain": "Digital Rights and Laws"
        }
        
        try:
            with open(self.output_path, "wb") as f:
                pickle.dump(payload, f)
            
            log.info("Graph build complete. Final Nodes: %d, Edges: %d", 
                     self.graph.number_of_nodes(), self.graph.number_of_edges())
        except Exception as e:
            log.error("Failed to save graph: %s", e)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = GraphBuilder()
    
    # Representative Digital Rights test cases
    test_chunks = [
        {
            "id": "chunk_001", 
            "text": "The GDPR empowers the Data Principal with the Right to Erasure."
        },
        {
            "id": "chunk_002", 
            "text": "The EDPB provides guidelines on how the GDPR should be enforced by fiduciaries."
        }
    ]
    
    builder.process_chunks(test_chunks)
