"""
Configuration file for RAG system tuning.

Adjust these parameters to optimize search relevance for your specific use case.
"""

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

# Chunk size in characters
CHUNK_SIZE = 1000  # Target size for each chunk
CHUNK_OVERLAP = 200  # Overlap between chunks for context continuity
MIN_CHUNK_SIZE = 200  # Minimum chunk size to avoid tiny fragments
MAX_CHUNK_SIZE = 2000  # Maximum chunk size to avoid overly long chunks

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================

# Available models (uncomment the one you want to use):
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Best quality, slower
# EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Good balance, faster
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Fastest, lower quality

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

# Default number of results to return
DEFAULT_NUM_RESULTS = 5

# Maximum number of results to return
MAX_NUM_RESULTS = 20

# Relevance threshold (0.0 to 1.0) - only show results above this threshold
MIN_RELEVANCE_THRESHOLD = 0.3

# ============================================================================
# QUERY PROCESSING CONFIGURATION
# ============================================================================

# Stop words to filter out during query processing
STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "this",
    "that",
    "these",
    "those",
}

# Minimum query length after stop word removal
MIN_QUERY_LENGTH = 2

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# ChromaDB collection name
COLLECTION_NAME = "md_docs"

# Database path
DB_PATH = "./chroma_db"

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

# Terminal width for formatting
TERMINAL_WIDTH = 80

# Show chunk metadata in results
SHOW_CHUNK_METADATA = True

# Show relevance scores
SHOW_RELEVANCE_SCORES = True

# ============================================================================
# ADVANCED TUNING OPTIONS
# ============================================================================

# Enable semantic chunking (respects markdown structure)
ENABLE_SEMANTIC_CHUNKING = True

# Enable query preprocessing
ENABLE_QUERY_PREPROCESSING = True

# Enable result filtering by metadata
ENABLE_METADATA_FILTERING = True

# Chunking strategies (when semantic chunking is enabled)
CHUNKING_STRATEGIES = {
    "respect_headers": True,  # Split at markdown headers
    "respect_sentences": True,  # Try to end chunks at sentence boundaries
    "respect_paragraphs": True,  # Try to respect paragraph boundaries
}

# Search strategies
SEARCH_STRATEGIES = {
    "use_metadata_filtering": True,  # Allow filtering by filename/headers
    "use_relevance_threshold": True,  # Only show results above threshold
    "use_query_expansion": False,  # Future: expand queries with synonyms
}
