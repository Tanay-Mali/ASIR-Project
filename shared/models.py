from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# =============================================================================
# CORE QUERY OBJECTS
# =============================================================================

@dataclass
class RawQuery:
    query_id: str
    session_id: str
    raw_text: str
    timestamp: str
    character_count: int
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ENTITY MODEL (Single Source of Truth)
# =============================================================================

@dataclass
class NamedEntity:
    text: str
    label: str
    start_char: int = 0
    end_char: int = 0


# =============================================================================
# PREPROCESSOR OUTPUT
# =============================================================================

@dataclass
class PreprocessorOutput:
    # ─────────────────────────────
    # Identity
    # ─────────────────────────────
    query_id: str
    session_id: str
    raw_text: str
    timestamp: Optional[str]
    cleaned_text: str

    # ─────────────────────────────
    # Structural
    # ─────────────────────────────
    token_count: int
    sentence_count: int
    is_question: bool
    is_multi_part: bool

    # ─────────────────────────────
    # Question understanding
    # ─────────────────────────────
    question_word: Optional[str]
    question_word_signal: float

    # ─────────────────────────────
    # Comparative / relational
    # ─────────────────────────────
    comparative_markers_found: List[str]
    comparative_marker_signal: Dict[str, float]

    relational_markers_found: List[str]
    relational_marker_signal: Dict[str, float]

    # ─────────────────────────────
    # Entities
    # ─────────────────────────────
    named_entities: List[Any]
    entity_type_signal: Dict[str, float]

    # ─────────────────────────────
    # Verb features
    # ─────────────────────────────
    root_verb: Optional[str]
    action_verbs_found: List[str]
    verb_type_signal: Dict[str, float]

    # ─────────────────────────────
    # Length features
    # ─────────────────────────────
    length_category: str
    length_signal: Dict[str, float]

    # ─────────────────────────────
    # No retrieval
    # ─────────────────────────────
    no_retrieval_triggered: bool
    no_retrieval_reason: str

    # ─────────────────────────────
    # Decomposition
    # ─────────────────────────────
    needs_decomposition: bool
    sub_queries: List[str]

    # ─────────────────────────────
    # Layer 1
    # ─────────────────────────────
    layer1_predicted_type: str
    layer1_confidence: float
    layer1_all_scores: Dict[str, float]
    layer1_sufficient: bool

    # ─────────────────────────────
    # Timing
    # ─────────────────────────────
    preprocessing_time_ms: float

    
# =============================================================================
# ROUTER / STRATEGY CALL (Consolidated)
# =============================================================================

@dataclass
class StrategyCall:
    strategy: str              # e.g., "vector", "bm25", "graph", "hierarchical"
    query_text: str
    query_type: str = "relational"   # factual | relational | causal | comparative
    top_k: int = 5
    weight: float = 1.0


# =============================================================================
# FEATURE VECTOR (Classifier Training)
# =============================================================================

@dataclass
class FeatureVector:
    query_id: str

    # Linguistic features
    token_count: int
    sentence_count: int
    is_question: int

    # Structural features
    is_multi_part: int
    has_comparative: int
    comparative_strength: float
    has_relational: int
    relational_strength: float

    # Entity features
    entity_count: int
    has_multiple_entities: int
    dominant_entity_type: int

    # Query understanding
    question_word_code: int
    verb_type_code: int

    # Layer-1 routing scores
    layer1_factual: float
    layer1_relational: float
    layer1_causal: float
    layer1_comparative: float
    layer1_exploratory: float

    # Embeddings
    embedding_dims: List[float] = field(default_factory=list)

    # Label (for training only)
    true_label: Optional[str] = None


# =============================================================================
# CLASSIFIER OUTPUT (Passes intent to the Router)
# =============================================================================

@dataclass
class ClassifierOutput:
    query_id: str
    session_id: str
    raw_text: str
    cleaned_text: str
    timestamp: str
    query_type: str
    confidence: float
    classified_by: str
    
    # Optional/Default fields
    all_scores: Dict[str, float] = field(default_factory=dict)
    llm_reasoning: str = ""
    llm_raw_response: str = ""
    xgboost_confidence: Optional[float] = None
    xgboost_feature_vector: Optional[Dict] = None
    needs_decomposition: bool = False
    sub_queries: List[str] = field(default_factory=list)
    sub_query_classifications: List[Any] = field(default_factory=list)
    preprocessing_time_ms: float = 0.0
    classification_time_ms: float = 0.0
    total_time_ms: float = 0.0
    phase: str = "phase1"


# =============================================================================
# INGESTION CHUNK (New - Base representation for indexing)
# =============================================================================

@dataclass
class Chunk:
    """Base chunk representation used during file ingestion and indexing."""
    chunk_id: str
    text: str
    document_id: str
    
    # ── NEW METADATA FIELDS FOR LEGAL STRUCTURE ──
    act: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    
    # Optional catch-all for other metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# RETRIEVED CHUNK (Core Retrieval Contract)
# =============================================================================

@dataclass
class RetrievedChunk:
    # Required fields
    chunk_id: str
    content: str
    source: str
    strategy: str   # vector | keyword | graph | hybrid
    
    # ── NEW METADATA FIELDS FOR LEGAL STRUCTURE ──
    act: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None

    # Retrieval + ranking
    page_number: int = 0
    score: float = 0.0
    final_score: Optional[float] = None
    rank: int = -1
    confidence: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Graph explainability
    relation_paths: List[Dict[str, Any]] = field(default_factory=list)
    legal_citation: List[str] = field(default_factory=list)
    graph_node: Optional[str] = None

    # Debug / evaluation
    retrieval_time: float = 0.0


# =============================================================================
# ROUTER OUTPUTS (Passes the final chunks to the Generator)
# =============================================================================

@dataclass
class RouterDecision:
    # ── Required Fields (No defaults) ──
    query_id: str
    session_id: str
    raw_text: str
    query_type: str
    confidence: float
    routing_mode: str
    confidence_band: str
    merge_strategy: str

    # ── Optional Fields (With defaults) ──
    primary_strategies: List[Any] = field(default_factory=list)
    fallback_strategies: List[Any] = field(default_factory=list)
    weights_used: Dict[str, float] = field(default_factory=dict)
    is_decomposed: bool = False
    routing_time_ms: float = 0.0
    timestamp: str = ""


@dataclass
class MergedResults:
    # ── Required Fields ──
    query_id: str
    
    # ── Optional Fields (With defaults) ──
    chunks: List[Any] = field(default_factory=list) 
    strategies_used: List[str] = field(default_factory=list)
    total_before_merge: int = 0
    total_after_merge: int = 0
    merge_strategy: str = "none"
    retrieval_time_ms: float = 0.0
    fallback_used: bool = False
    fallback_reason: str = ""
