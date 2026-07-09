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

from mtl.retrieval.cognitive_rerank import (
    extract_cognitive_query,
    cognitive_rerank,
    _build_dimension_brief,
)
from mtl.retrieval.synergy import synergy_aware_selection, compute_selection_quality
from mtl.llm import get_client, get_model

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

    # Mapping from cognitive dimension names to their pkl file names.
    _DIMENSION_TO_FILE = {
        "causal": "causal_memory.pkl",
        "contrastive": "contrastive_memory.pkl",
        "strategic": "strategic_memory.pkl",
        "environment": "environment_memory.pkl",
    }

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
        excluded_dimensions: list | None = None,
        excluded_levels: list | None = None,
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
            excluded_dimensions: Cognitive dimensions to EXCLUDE from retrieval pool.
                                 For dimension leave-one-out ablation (e7--e10).
            excluded_levels: Abstraction levels to EXCLUDE from retrieval pool.
                             For abstraction tier leave-one-out ablation (e11--e13).
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
        self.excluded_dimensions = [d.strip().lower() for d in (excluded_dimensions or [])]
        self.excluded_levels = [l.strip().lower() for l in (excluded_levels or [])]

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
        """Load all memory pkl files, separate insight layer, precompute embeddings.

        Excluded dimensions (e7--e10 ablation) are skipped at the file level:
        their pkl files are never loaded.  Excluded levels (e11--e13 ablation)
        are filtered after loading.
        """
        # Compute which pkl files to skip based on excluded dimensions.
        _excluded_pkl_files: set[str] = set()
        if self.excluded_dimensions:
            for dim in self.excluded_dimensions:
                pkl_name = self._DIMENSION_TO_FILE.get(dim)
                if pkl_name:
                    _excluded_pkl_files.add(pkl_name)
            if _excluded_pkl_files:
                print(f"[retriever] Excluded dimensions: {self.excluded_dimensions}"
                      f"  (skipping: {sorted(_excluded_pkl_files)})")

        pkl_files = sorted(
            p for p in self.memory_dir.glob("*.pkl")
            if p.name not in _excluded_pkl_files
        )
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

        # Filter by excluded abstraction levels (e11--e13 ablation).
        if self.excluded_levels:
            n_before = len(all_entries)
            all_entries = [
                e for e in all_entries
                if e.get("level", "").lower() not in self.excluded_levels
            ]
            n_after = len(all_entries)
            print(f"  [retriever] Excluded levels: {self.excluded_levels}"
                  f"  (filtered {n_before - n_after} entries, {n_after} remain)")

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
    # Random retrieval (E15 ablation)
    # -----------------------------------------------------------
    def retrieve_random(self, task_text: str, top_n: int = 20, top_k: int = 3) -> list[dict]:
        """Semantic retrieval + random selection from top-N candidates.

        Used by E15 (random memory ablation): runs the same dual-score semantic
        retrieval as the normal pipeline (Steps 1--2), then uniformly samples K
        memories from the top-N candidates instead of applying cognitive rerank
        and synergy selection.
        """
        import random as _random_mod
        rng = _random_mod.Random(42)  # Fixed seed for reproducibility

        if not self.memories:
            print("[retriever-random] No memories loaded, returning empty.")
            return []

        print(f"\n[retriever-random] Query: {task_text[:100]}...")

        # Step 1: Extract cognitive profile
        print("  [R1] Extracting cognitive query profile...")
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
        cognitive_query_text = " ".join(
            c for c in cognitive_query_parts if c and c != "other"
        )

        query_task_emb = _get_embed_model().encode(task_text).tolist()
        query_cog_emb = (
            _get_embed_model().encode(cognitive_query_text).tolist()
            if cognitive_query_text else None
        )

        # Step 2: Dual-score semantic retrieval
        print(f"  [R2] Dual-score semantic retrieval (top {top_n})...")
        candidates = self._semantic_retrieve(query_task_emb, query_cog_emb, top_n)

        # Step 3: Random selection
        print(f"  [R3] Randomly selecting {top_k} from {len(candidates)} candidates...")
        n_pick = min(top_k, len(candidates))
        selected = rng.sample(candidates, n_pick) if n_pick > 0 else []

        # Attach insights
        if self.attach_insights:
            for mem in selected:
                task = mem.get("task_name", "")
                linked = self._task_to_insights.get(task, [])
                mem["_linked_insights"] = linked
                if linked:
                    dims = [ins.get("type", "?") for ins in linked]
                    print(f"    -> attached {len(linked)} insight(s) "
                          f"[{', '.join(dims)}] from task={task}")

        print(f"  [retriever-random] Selected {len(selected)} memories "
              f"(random from top-{len(candidates)} semantic candidates)")
        for i, s in enumerate(selected):
            n_linked = len(s.get("_linked_insights", []))
            print(f"    [{i+1}] [{s.get('level','?')}/{s.get('type','?')}] "
                  f"semantic={s.get('semantic_score',0):.3f} "
                  f"task={s.get('task_name','?')} (+{n_linked} insights)")

        return selected

    # -----------------------------------------------------------
    # LLM direct top-K (E16 ablation)
    # -----------------------------------------------------------
    def retrieve_llm_direct(self, task_text: str, top_n: int = 20, top_k: int = 3) -> list[dict]:
        """Semantic retrieval + LLM directly selects top-K without dimension structure.

        Used by E16 (LLM direct top-K ablation): runs the same dual-score semantic
        retrieval as the normal pipeline (Steps 1--2), then passes all N candidates
        to the LLM WITHOUT cognitive dimension labels.
        """
        import json as _json

        if not self.memories:
            print("[retriever-llm-direct] No memories loaded, returning empty.")
            return []

        print(f"\n[retriever-llm-direct] Query: {task_text[:100]}...")

        # Step 1: Extract cognitive profile
        print("  [D1] Extracting cognitive query profile...")
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
        cognitive_query_text = " ".join(
            c for c in cognitive_query_parts if c and c != "other"
        )

        query_task_emb = _get_embed_model().encode(task_text).tolist()
        query_cog_emb = (
            _get_embed_model().encode(cognitive_query_text).tolist()
            if cognitive_query_text else None
        )

        # Step 2: Dual-score semantic retrieval
        print(f"  [D2] Dual-score semantic retrieval (top {top_n})...")
        candidates = self._semantic_retrieve(query_task_emb, query_cog_emb, top_n)

        if not candidates:
            print("  [retriever-llm-direct] No candidates, returning empty.")
            return []

        # Step 3: LLM direct selection without dimension structure
        print(f"  [D3] LLM direct selection (flat, no dimension labels) from "
              f"{len(candidates)} candidates...")

        def _strip_dimension_label(brief: str) -> str:
            lines = brief.split("\n")
            if lines and lines[0].startswith("[type="):
                return "\n".join(lines[1:])
            return brief

        candidate_briefs = []
        for i, c in enumerate(candidates):
            full_brief = _build_dimension_brief(c)
            flat_brief = _strip_dimension_label(full_brief)
            candidate_briefs.append(f"### Candidate {i}\n{flat_brief}")

        candidates_text = "\n\n".join(candidate_briefs)

        prompt = f"""You are selecting relevant past experiences (memories) for a coding task.

## Task Query:
{task_text[:2000]}

## Candidate Memories (select the {top_k} most relevant):
{candidates_text}

## Instructions:
From the {len(candidates)} candidates above, select the {top_k} that are MOST relevant
to the task query.  Consider:
- Does the memory describe a similar bug pattern, error mechanism, or fix strategy?
- Would the knowledge in this memory help the agent solve this specific task?
- Is the memory about the same repository/ecosystem or a transferable pattern?

Output a JSON object (no markdown):
{{"selected_indices": [idx1, idx2, idx3], "reasons": ["reason for idx1", "reason for idx2", "reason for idx3"]}}
where indices are 0-based candidate numbers."""

        try:
            response = get_client().chat.completions.create(
                model=get_model(),
                messages=[
                    {"role": "system", "content": "You are a memory relevance judge. Output only the requested JSON."},
                    {"role": "user", "content": prompt},
                ],
                timeout=180.0,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            result = _json.loads(raw)
            selected_indices = result.get("selected_indices", [])
        except Exception as e:
            print(f"  [retriever-llm-direct] LLM selection failed: {e}, "
                  f"falling back to top-{top_k} by semantic score")
            selected_indices = list(range(min(top_k, len(candidates))))

        # Gather selected memories
        selected = []
        for idx in selected_indices:
            if 0 <= idx < len(candidates):
                c = candidates[idx]
                c["cognitive_score"] = 0.5
                c["cognitive_dimension"] = "llm_direct"
                c["cognitive_reason"] = "LLM direct selection (no dimension structure)"
                selected.append(c)

        # Fallback fill
        if len(selected) < top_k:
            for c in candidates:
                if c not in selected:
                    selected.append(c)
                    c["cognitive_score"] = 0.3
                    c["cognitive_dimension"] = "llm_direct_fallback"
                    c["cognitive_reason"] = "Fallback fill (LLM selected too few)"
                    if len(selected) >= top_k:
                        break

        # Attach insights
        if self.attach_insights:
            for mem in selected:
                task = mem.get("task_name", "")
                linked = self._task_to_insights.get(task, [])
                mem["_linked_insights"] = linked
                if linked:
                    dims = [ins.get("type", "?") for ins in linked]
                    print(f"    -> attached {len(linked)} insight(s) "
                          f"[{', '.join(dims)}] from task={task}")

        print(f"  [retriever-llm-direct] Selected {len(selected)} memories "
              f"(LLM picked {len(set(selected_indices) & set(range(len(candidates))))} "
              f"from {len(candidates)} candidates)")
        for i, s in enumerate(selected):
            n_linked = len(s.get("_linked_insights", []))
            print(f"    [{i+1}] [{s.get('level','?')}/{s.get('type','?')}] "
                  f"semantic={s.get('semantic_score',0):.3f} "
                  f"task={s.get('task_name','?')} (+{n_linked} insights)")

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
