"""
ingestion_graph.py
==================
Reads real PDFs from data_file/, extracts chunk units, 
and builds the Knowledge Graph.
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from graph_builder.document_loader import load_pdf
from graph_builder.build_graph import GraphBuilder
import config.core_config as C

def ingest_pdfs_to_graph(data_dir: str = "data_file"):
    data_path = Path(data_dir)
    if not data_path.exists():
        log.error(f"Directory {data_dir} not found!")
        return

    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        log.warning(f"No PDFs found in {data_dir}")
        return

    log.info(f"Found {len(pdf_files)} PDFs. Starting extraction...")

    all_graph_chunks = []

    # 1. Load and process all PDFs
    for pdf in pdf_files:
        log.info(f"Processing: {pdf.name}")
        text_units = load_pdf(pdf)
        
        # Convert TextUnits into the dict format GraphBuilder expects
        for unit in text_units:
            chunk_dict = {
                "id": unit.chunk_id,
                # We feed the broader 'context' to the NLP extractor for better relation building
                "text": unit.context, 
                "metadata": {
                    "source": unit.source,
                    "page": unit.page,
                },
                "legal_citations": unit.legal_citations
            }
            all_graph_chunks.append(chunk_dict)

    log.info(f"Extracted {len(all_graph_chunks)} total chunks for Graph Construction.")

    # 2. Build the Graph
    builder = GraphBuilder(output_path=C.GRAPH_PICKLE_PATH)
    
    # Process all chunks (this will show the tqdm progress bar!)
    builder.process_chunks(all_graph_chunks)
    
    log.info("✅ Full PDF Ingestion to Knowledge Graph Complete!")

if __name__ == "__main__":
    ingest_pdfs_to_graph()