"""Unified Cognitive Memory Retriever.

Four-stage retrieval pipeline:
  1. Cognitive profile extraction from query
  2. Dual-score semantic retrieval (task + cognitive embeddings)
  3. Dimension-aware cognitive rerank (LLM scores each memory on its own dimension)
  4. Synergy-aware selection (greedy diversity maximization)

Usage:
    from mtl.retrieval import CognitiveRetriever
    retriever = CognitiveRetriever("memories/swebench-verified")
    results = retriever.retrieve("Fix the KeyError in data processing pipeline", top_k=3)
"""
import pickle
import threading
from pathlib import Path
import numpy as np

from mtl.retrieval.cognitive_rerank import extract_cognitive_query, cognitive_rerank
from mtl.retrieval.synergy import synergy_aware_selection, compute_selection_quality

_embed_model = None
_embed_lock = threading.Lock()


def _get_embed_model():
    """Lazily load SentenceTransformer (thread-safe).

    Uses local_files_only=True to avoid HuggingFace network calls.
    """
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                from sentence_transformers import SentenceTransformer
                _embed_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    local_files_only=True,
                )
    return _embed_model


class CognitiveRetriever:
    """Unified retrieval with dimension-aware cognitive rerank + synergy."""

    def __init__(
        self,
        memory_dir: str,
        alpha_semantic: float = 0.35,
        alpha_cognitive: float = 0.65,
        alpha_dual_task: float = 0.70,
        top_n_candidates: int = 20,
        top_k: int = 3,
        use_cognitive_rerank: bool = True,
        use_llm_synergy: bool = True,
        score_threshold_floor: float = 0.45,
        score_threshold_std: float = 0.5,
        min_memories: int = 1,
    ):
        """
        Args:
            memory_dir: Path to directory containing *.pkl memory files
            alpha_semantic: Weight for semantic embedding (vs LLM rerank). Default 0.35.
            alpha_cognitive: Weight for cognitive relevance (vs semantic). Default 0.65.
            alpha_dual_task: Task vs cognitive embedding weight. 0.70 = 70% concrete + 30% abstract.
            top_n_candidates: Candidates before re-ranking
            top_k: Final memories to return
            use_cognitive_rerank: Enable LLM cognitive rerank (Step 3)
            use_llm_synergy: Enable LLM conflict/synergy verification (Step 4)
            score_threshold_floor: Absolute minimum combined_score
            score_threshold_std: Multiplier for dynamic threshold (higher = stricter)
            min_memories: Minimum memories to keep regardless of scores
        """
        self.memory_dir = Path(memory_dir)
        self.alpha_semantic = alpha_semantic
        self.alpha_cognitive = alpha_cognitive
        self.alpha_dual_task = alpha_dual_task
        self.top_n = top_n_candidates
        self.top_k = top_k
        self.use_cognitive_rerank = use_cognitive_rerank
        self.use_llm_synergy = use_llm_synergy
        self.score_threshold_floor = score_threshold_floor
        self.score_threshold_std = score_threshold_std
        self.min_memories = min_memories

        self.memories: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._cognitive_embeddings: np.ndarray | None = None
        self._stats: dict = {}
        self._task_to_insights: dict[str, list[dict]] = {}
        self._task_to_concrete: dict[str, list[dict]] = {}

        self._load_all()

    # -----------------------------------------------------------
    # Loading
    # -----------------------------------------------------------
    def _load_all(self):
        """Load all memory pkl files, separate insight layer, precompute embeddings."""
        pkl_files = sorted(self.memory_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {self.memory_dir}")

        print(f"[retriever] Loading {len(pkl_files)} memory files from {self.memory_dir}")

        skipped_count = 0
        all_entries: list[dict] = []
        for pkl_path in pkl_files:
            try:
                with open(pkl_path, "rb") as f:
                    entries = pickle.load(f)
                for entry in entries:
                    if "memory" not in entry:
                        alt_keys = ["insight", "workflow"]
                        alt_content = None
                        for ak in alt_keys:
                            if ak in entry:
                                alt_content = entry.pop(ak)
                                break
                        if alt_content is None:
                            alt_content = dict(entry)
                        entry["memory"] = alt_content
                        skipped_count += 1

                    if "key_embedding" not in entry:
                        for alt_emb_key in ("generalized_query_embedding", "embedding"):
                            if alt_emb_key in entry:
                                entry["key_embedding"] = entry.pop(alt_emb_key)
                                break
                    if "level" not in entry:
                        entry["level"] = "unknown"
                    if "type" not in entry:
                        entry["type"] = "unknown"
                    entry["_source_file"] = pkl_path.name
                    all_entries.append(entry)
            except Exception as e:
                print(f"  [retriever] WARNING: failed to load {pkl_path.name}: {e}")

        if skipped_count > 0:
            print(f"  [retriever] Normalized {skipped_count} legacy entries (missing 'memory' key).")

        # Separate insight-level from concrete
        insight_entries: list[dict] = []
        concrete_entries: list[dict] = []
        for entry in all_entries:
            if entry.get("level") == "insight":
                insight_entries.append(entry)
            else:
                concrete_entries.append(entry)

        for entry in insight_entries:
            task = entry.get("task_name", "")
            if task:
                self._task_to_insights.setdefault(task, []).append(entry)

        for entry in concrete_entries:
            task = entry.get("task_name", "")
            if task:
                self._task_to_concrete.setdefault(task, []).append(entry)

        self.memories = concrete_entries

        # Build key embedding array
        embeddings_list = []
        for mem in self.memories:
            emb = mem.get("key_embedding", [])
            embeddings_list.append(emb if emb else [0.0] * 384)
        if embeddings_list:
            self._embeddings = np.array(embeddings_list, dtype=np.float32)

        # Build cognitive embedding array
        cog_list = []
        cog_available = 0
        for mem in self.memories:
            cog_emb = mem.get("cognitive_embedding")
            if cog_emb:
                cog_list.append(cog_emb)
                cog_available += 1
            else:
                cog_list.append([0.0] * 384)
        if cog_list:
            self._cognitive_embeddings = np.array(cog_list, dtype=np.float32)
        print(f"  [retriever]   Cognitive embeddings available: {cog_available}/{len(self.memories)}")

        self._stats = {
            "total_memories": len(all_entries),
            "retrieval_pool": len(concrete_entries),
            "insight_attached": len(insight_entries),
            "tasks_with_insights": len(self._task_to_insights),
            "total_files": len(pkl_files),
            "dimensions": list(set(m.get("type", "") for m in self.memories)),
            "levels": list(set(m.get("level", "") for m in self.memories)),
        }
        print(f"  [retriever] Loaded {self._stats['total_memories']} total memories")
        print(f"  [retriever]   Retrieval pool (concrete): {self._stats['retrieval_pool']}")
        print(f"  [retriever]   Insight layer (attached):  {self._stats['insight_attached']}")
        print(f"  [retriever]   Tasks with insights:       {self._stats['tasks_with_insights']}")
        print(f"  [retriever] Dimensions: {self._stats['dimensions']}")
        print(f"  [retriever] Levels: {self._stats['levels']}")

    # -----------------------------------------------------------
    # Reload (for sequential/incremental memory pool updates)
    # -----------------------------------------------------------
    def reload(self):
        """Reload all memory files from disk.

        Use this when new memories have been added to the pool since
        the retriever was created (e.g., in sequential execution where
        each completed task contributes its memories to the pool).
        """
        self.memories = []
        self._embeddings = None
        self._cognitive_embeddings = None
        self._stats = {}
        self._task_to_insights = {}
        self._task_to_concrete = {}
        self._load_all()

    # -----------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------
    def retrieve(self, task_text: str, top_k: int | None = None) -> list[dict]:
        """Main retrieval entry point.

        Args:
            task_text: The new task description
            top_k: Override default number of memories to return

        Returns:
            List of memory dicts with scores and metadata attached
        """
        k = top_k or self.top_k
        if not self.memories:
            print("[retriever] No memories loaded, returning empty.")
            return []

        print(f"\n[retriever] Query: {task_text[:100]}...")

        # Step 1: Extract cognitive profile from query
        print("  [1/4] Extracting cognitive query profile...")
        query_profile = extract_cognitive_query(task_text)

        cognitive_query_parts = []
        cs = query_profile.get("causal_signature", {})
        if cs:
            cognitive_query_parts.append(cs.get("causal_signature", ""))
            cognitive_query_parts.append(cs.get("error_category", ""))
            cognitive_query_parts.append(cs.get("cause_category", ""))
            cognitive_query_parts.append(cs.get("intervention_type", ""))
            chain = cs.get("causal_chain", [])
            if chain:
                cognitive_query_parts.append(" -> ".join(chain))
        for dim_name in ["contrastive_needs", "strategic_needs", "environment_needs"]:
            val = query_profile.get(dim_name, "")
            if val:
                cognitive_query_parts.append(val)
        cognitive_query_text = " ".join(c for c in cognitive_query_parts if c and c != "other")

        query_task_emb = _get_embed_model().encode(task_text).tolist()
        query_cog_emb = _get_embed_model().encode(cognitive_query_text).tolist() if cognitive_query_text else None

        # Step 2: Dual-score semantic retrieval
        print(f"  [2/4] Dual-score semantic retrieval (top {self.top_n})...")
        candidates = self._semantic_retrieve(query_task_emb, query_cog_emb, self.top_n)

        # Step 3: Dimension-aware cognitive rerank
        if self.use_cognitive_rerank:
            print(f"  [3/4] Dimension-aware cognitive rerank (batch LLM)...")
            candidates = cognitive_rerank(
                query_profile, candidates,
                alpha_semantic=self.alpha_semantic,
                alpha_cognitive=self.alpha_cognitive,
            )
            for cand in candidates:
                semantic = cand.get("semantic_score", 0.0)
                cognitive = cand.get("cognitive_score", 0.0)
                cand["combined_score"] = (
                    self.alpha_semantic * semantic +
                    self.alpha_cognitive * cognitive
                )
                cand["score_breakdown"] = {
                    "semantic": round(semantic, 4),
                    "cognitive": round(cognitive, 4),
                    "combined": round(cand["combined_score"], 4),
                    "dual_task": round(cand.get("task_score", 0), 4),
                    "dual_cog": round(cand.get("cog_score", 0), 4),
                }
        else:
            print(f"  [3/4] Cognitive rerank disabled, using semantic scores only.")
            for cand in candidates:
                semantic = cand.get("semantic_score", 0.0)
                cand["combined_score"] = semantic
                cand["score_breakdown"] = {
                    "semantic": round(semantic, 4),
                    "cognitive": 0.0,
                    "combined": round(semantic, 4),
                    "dual_task": round(cand.get("task_score", 0), 4),
                    "dual_cog": round(cand.get("cog_score", 0), 4),
                }

        candidates.sort(key=lambda c: c.get("combined_score", 0), reverse=True)

        # Step 4: Synergy-aware final selection
        print(f"  [4/4] Synergy-aware selection (top {k}, llm_synergy={self.use_llm_synergy})...")
        selected = synergy_aware_selection(candidates, k, use_llm=self.use_llm_synergy)

        quality = compute_selection_quality(selected)
        for mem in selected:
            mem["selection_quality"] = quality

        # Dynamic threshold filtering
        dynamic_threshold = self._compute_dynamic_threshold(candidates)
        qualified = [m for m in selected
                     if m.get("combined_score", 0) >= dynamic_threshold]
        n_discarded = len(selected) - len(qualified)

        if len(qualified) < self.min_memories:
            fallback = sorted(selected,
                            key=lambda m: m.get("combined_score", 0),
                            reverse=True)[:self.min_memories]
            qualified = fallback
            n_discarded = len(selected) - len(qualified)

        if n_discarded > 0:
            print(f"  [retriever] Dynamic threshold {dynamic_threshold:.3f} "
                  f"(floor={self.score_threshold_floor}): "
                  f"discarded {n_discarded}, kept {len(qualified)}")

        selected = qualified

        for mem in selected:
            mem["_dynamic_threshold"] = round(dynamic_threshold, 4)

        # Step 5: Attach sibling insight memories
        for mem in selected:
            task = mem.get("task_name", "")
            linked = self._task_to_insights.get(task, [])
            mem["_linked_insights"] = linked
            if linked:
                dims = [ins.get("type", "?") for ins in linked]
                print(f"    -> attached {len(linked)} insight(s) [{', '.join(dims)}] "
                      f"from task={task}")

        print(f"  [retriever] Selected {len(selected)} memories "
              f"(threshold={dynamic_threshold:.3f})")
        for i, s in enumerate(selected):
            n_linked = len(s.get("_linked_insights", []))
            print(f"    [{i+1}] [{s.get('level','?')}/{s.get('type','?')}] "
                  f"combined={s.get('combined_score',0):.3f} "
                  f"(task={s.get('task_score', 0):.3f} cog={s.get('cog_score', 0):.3f}) "
                  f"task={s.get('task_name','?')} "
                  f"(+{n_linked} insights)")

        return selected

    # -----------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------
    def _compute_dynamic_threshold(self, candidates: list[dict]) -> float:
        """Compute dynamic score threshold from candidate distribution."""
        scores = [c.get("combined_score", 0) for c in candidates
                  if c.get("combined_score", 0) > 0]
        if not scores:
            return self.score_threshold_floor
        if len(scores) < 3:
            return max(self.score_threshold_floor, min(scores))

        import statistics
        median = statistics.median(scores)
        stdev = statistics.stdev(scores)
        relative = median - self.score_threshold_std * stdev
        return max(self.score_threshold_floor, relative)

    def _semantic_retrieve(
        self,
        query_task_emb: list[float],
        query_cog_emb: list[float] | None,
        n: int,
    ) -> list[dict]:
        """Dual-score semantic retrieval with task + cognitive embeddings."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return self.memories[:n]

        # Channel 1: Task-level similarity
        task_vec = np.array(query_task_emb, dtype=np.float32)
        t_norm = np.linalg.norm(task_vec)
        if t_norm > 0:
            task_vec = task_vec / t_norm

        _key_norms = np.linalg.norm(self._embeddings, axis=1)
        _key_norms = np.where(_key_norms == 0, 1.0, _key_norms)
        _key_normed = self._embeddings / _key_norms[:, np.newaxis]

        task_sims = np.dot(_key_normed, task_vec)
        task_sims = np.nan_to_num(task_sims, nan=0.0)

        # Channel 2: Cognitive-level similarity
        if query_cog_emb is not None and self._cognitive_embeddings is not None:
            cog_vec = np.array(query_cog_emb, dtype=np.float32)
            c_norm = np.linalg.norm(cog_vec)
            if c_norm > 0:
                cog_vec = cog_vec / c_norm

            _cog_norms = np.linalg.norm(self._cognitive_embeddings, axis=1)
            _cog_norms = np.where(_cog_norms == 0, 1.0, _cog_norms)
            _cog_normed = self._cognitive_embeddings / _cog_norms[:, np.newaxis]

            cog_sims = np.dot(_cog_normed, cog_vec)
            cog_sims = np.nan_to_num(cog_sims, nan=0.0)
        else:
            cog_sims = np.zeros(len(task_sims), dtype=np.float32)

        # Weighted combination
        alpha = self.alpha_dual_task
        final_sims = alpha * task_sims + (1.0 - alpha) * cog_sims

        top_indices = np.argsort(final_sims)[::-1][:n]

        candidates = []
        for idx in top_indices:
            mem = dict(self.memories[idx])
            mem["semantic_score"] = float(final_sims[idx])
            mem["task_score"] = float(task_sims[idx])
            mem["cog_score"] = float(cog_sims[idx])
            mem["_index"] = int(idx)
            candidates.append(mem)

        return candidates

    # -----------------------------------------------------------
    # Statistics & introspection
    # -----------------------------------------------------------
    @property
    def stats(self) -> dict:
        return self._stats

    def memory_type_breakdown(self) -> dict:
        """Count memories by type and level."""
        breakdown = {}
        for mem in self.memories:
            key = f"{mem.get('level','?')}/{mem.get('type','?')}"
            breakdown[key] = breakdown.get(key, 0) + 1
        return breakdown

    def describe(self) -> str:
        """Human-readable summary of the memory store."""
        return "\n".join([
            f"Memory Store: {self.memory_dir}",
            f"  Total memories: {self._stats['total_memories']}",
            f"  Retrieval pool: {self._stats['retrieval_pool']}",
            f"  Dimensions: {self._stats['dimensions']}",
            f"  Levels: {self._stats['levels']}",
            f"  Alpha: semantic={self.alpha_semantic} cognitive={self.alpha_cognitive} dual_task={self.alpha_dual_task}",
            f"  Cognitive rerank: {self.use_cognitive_rerank} | LLM synergy: {self.use_llm_synergy}",
            f"  Threshold: floor={self.score_threshold_floor} std_mult={self.score_threshold_std} min_mem={self.min_memories}",
        ])
