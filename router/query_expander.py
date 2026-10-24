# router/query_expander.py

import re

# Map user-friendly/broad terms to the specific legal terminology used in your dataset
LEGAL_SYNONYMS = {
    "digital rights": "data protection privacy rights of data principal",
    "data controller": "data fiduciary",
    "data processor": "data processor",
    "companies": "body corporate data fiduciary",
    "user data": "personal data sensitive personal data",
    "user consent": "data principal consent",
    "punishment": "penalty offences compensation",
    "digital privacy": "data protection reasonable security practices",
    "breach": "personal data breach cyber incident",
    "user rights": "rights of data principal right to information",
}

def expand_query(query_text: str) -> str:
    """
    Expands a query string with legal synonyms to boost BM25 keyword matching.
    Only matching synonyms are appended to the end of the original query.
    """
    expanded_terms = set()
    query_lower = query_text.lower()
    
    # Check for substring matches (using word boundaries to avoid partial matches)
    for key, synonyms in LEGAL_SYNONYMS.items():
        if re.search(r'\b' + re.escape(key) + r'\b', query_lower):
            expanded_terms.add(synonyms)
            
    # If no synonyms match, return the exact original query
    if not expanded_terms:
        return query_text
        
    # Append the expanded terms to the original query
    expansion_string = " ".join(expanded_terms)
    expanded_query = f"{query_text} {expansion_string}"
    
    return expanded_query