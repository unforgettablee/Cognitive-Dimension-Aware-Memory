"""Memory Transfer Learning (MTL) Pipeline.

Cross-domain memory transfer for coding agents with 4x4 cognitive matrix retrieval.

Pipeline:
    1. Extract: Run agents on source tasks, extract cognitive + traditional memories
    2. Build Pool: Aggregate memories, index embeddings
    3. Retrieve: Dual-score retrieval -> cognitive rerank -> synergy selection
    4. Run: Inject memories into agent prompts
    5. Evaluate: Pass/fail judgment from verifier results

Usage:
    from mtl import CognitiveRetriever, MemoryExtractor, ExperimentPipeline
    from mtl.retrieval import CognitiveRetriever
"""

__version__ = "1.0.0"
__all__ = [
    "CognitiveRetriever",
    "MemoryExtractor",
    "ExperimentPipeline",
    "get_retrieval_config",
]

from mtl.retrieval.retriever import CognitiveRetriever
from mtl.extraction.extractor import MemoryExtractor
from mtl.pipeline.experiment import ExperimentPipeline
