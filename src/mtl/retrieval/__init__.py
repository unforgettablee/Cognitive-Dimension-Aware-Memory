"""MTL Retrieval — 4-stage cognitive memory retrieval pipeline.

1. Cognitive profile extraction from query
2. Dual-score semantic retrieval (task + cognitive embeddings)
3. Dimension-aware cognitive rerank (LLM)
4. Synergy-aware selection
"""

from mtl.retrieval.retriever import CognitiveRetriever

__all__ = ["CognitiveRetriever"]
