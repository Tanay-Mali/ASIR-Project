"""
graph_builder/digital_rights_entity_dictionary.py
==================================================
Canonical dictionary for Digital Rights and Laws.
Categorizes entities to enable structured graph traversal and reasoning.
"""

# ─────────────────────────────────────────────────────────────────────
# Entity Definitions
# ─────────────────────────────────────────────────────────────────────

DIGITAL_RIGHTS_ENTITIES = {
    # --- Legislation & Frameworks ---
    "gdpr": {
        "type": "legislation",
        "aliases": ["general data protection regulation", "eu gdpr", "regulation 2016/679", "european privacy law"]
    },
    "dpdp act 2023": {
        "type": "legislation",
        "aliases": ["digital personal data protection act", "dpdpa", "india data act", "dpdp", "dpdp 2023"]
    },
    "it act 2000": {
        "type": "legislation",
        "aliases": ["information technology act", "it act", "it rules", "section 66a"]
    },
    "digital services act": {
        "type": "legislation",
        "aliases": ["dsa", "eu digital services act"]
    },

    # --- Regulatory & Enforcement Bodies ---
    "edpb": {
        "type": "regulatory_body",
        "aliases": ["european data protection board", "article 29 working party", "edps"]
    },
    "cert-in": {
        "type": "regulatory_body",
        "aliases": ["indian computer emergency response team", "cert in", "national nodal agency"]
    },
    "data protection board of india": {
        "type": "regulatory_body",
        "aliases": ["dpb", "the board", "indian data regulator"]
    },

    # --- Individual Rights ---
    "right to erasure": {
        "type": "digital_right",
        "aliases": ["right to be forgotten", "rtbf", "deletion right", "article 17 gdpr", "section 12 dpdp"]
    },
    "right to portability": {
        "type": "digital_right",
        "aliases": ["data portability", "right to move data", "article 20 gdpr", "portability right"]
    },
    "right to access": {
        "type": "digital_right",
        "aliases": ["subject access request", "sar", "right to be informed", "article 15 gdpr"]
    },
    "right to withdraw consent": {
        "type": "digital_right",
        "aliases": ["consent withdrawal", "right to opt out", "withdrawal of consent"]
    },

    # --- Principles & Compliance ---
    "data minimization": {
        "type": "principle",
        "aliases": ["collection limitation", "minimal data collection", "article 5 gdpr"]
    },
    "purpose limitation": {
        "type": "principle",
        "aliases": ["specified purpose", "use limitation"]
    },
    "dpia": {
        "type": "mechanism",
        "aliases": ["data protection impact assessment", "privacy impact assessment", "risk assessment"]
    },
    "consent notice": {
        "type": "mechanism",
        "aliases": ["privacy notice", "privacy policy", "section 5 notice"]
    },

    # --- Actors & Concepts ---
    "data fiduciary": {
        "type": "actor",
        "aliases": ["data controller", "fiduciary", "significant data fiduciary"]
    },
    "data principal": {
        "type": "actor",
        "aliases": ["data subject", "the principal"]
    },
    "personal data": {
        "type": "concept",
        "aliases": ["pii", "personally identifiable information", "sensitive personal data"]
    },
    "data breach": {
        "type": "event",
        "aliases": ["security breach", "leak", "personal data breach"]
    }
}

# ─────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────

def build_alias_lookup() -> dict[str, str]:
    """
    Flattens dictionary into a map of normalized_alias -> canonical_name.
    Ensures consistent normalization with the EntityExtractor.
    """
    if not DIGITAL_RIGHTS_ENTITIES:
        return {}
        
    lookup = {}
    
    for canonical, info in DIGITAL_RIGHTS_ENTITIES.items():
        # 1. Normalize Canonical Name
        norm_canonical = " ".join(canonical.lower().strip().split())
        if norm_canonical not in lookup:
            lookup[norm_canonical] = canonical
        
        # 2. Normalize Aliases
        for alias in info.get("aliases", []):
            norm_alias = " ".join(alias.lower().strip().split())
            
            # Collision Protection: Don't let an alias overwrite 
            # an existing mapping (prioritize original entries)
            if norm_alias not in lookup:
                lookup[norm_alias] = canonical
            
    return lookup

def get_canonical_types() -> dict[str, str]:
    """
    Reverse map for O(1) type lookups during extraction.
    """
    return {k: v["type"] for k, v in DIGITAL_RIGHTS_ENTITIES.items()}
