"""Unified Cognitive Memory Retriever.

Combines four innovations into a single retrieval pipeline:

1. Dual-Score Semantic Retrieval -- query task text matches memory key_embedding
   (three-section hybrid) for concrete similarity, while query cognitive profile
   matches memory cognitive_embedding (pure abstract) for cross-repo pattern
   matching. Weighted combination prevents abstract signals from being drowned
   by concrete text.

2. Dimension-Aware Cognitive Rerank -- matches memories by their native
   cognitive dimension. Causal memories are scored on causal-structure
   isomorphism, contrastive on anti-pattern relevance, strategic on
   methodology transferability, environment on repo/tool applicability.

3. Memory Synergy -- the top-K set is selected to maximize joint information
   value, rewarding complementary pairs and penalizing redundant pairs.

4. Insight Layering -- insight-level memories are excluded from the retrieval
   pool and instead attached as _linked_insights to selected concrete memories.

Pipeline:
  Query -> cognitive profile extraction ->
    dual-score semantic retrieval (task_sim + cog_sim, weighted merge) ->
    dimension-aware cognitive rerank ->
    synergy-aware selection -> attach sibling insights -> top-K

Usage:
  retriever = CognitiveRetriever("memories/swebench-verified")
  results = retriever.retrieve("Fix the KeyError in data processing pipeline", top_k=3)
  for r in results:
      print(r["level"], r["type"], r["combined_score"])
      for ins in r.get("_linked_insights", []):
          print("  linked insight:", ins.get("type"), ins.get("memory", {}).get("principle_name"))
"""
import os
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
    """Lazily load the SentenceTransformer model (thread-safe).

    Uses local_files_only=True to avoid network calls to HuggingFace,
    which would fail in environments without internet access (e.g., China).
    The model must already be cached locally.
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


# Patterns to strip from query task text when use_anchor_in_query_embedding=False.
# These capture repository-specific concrete anchors (file paths, shell commands,
# test invocations) that would create false-positive embedding matches across
# same-repo memories regardless of actual bug relevance.
import re as _re

_CONCRETE_ANCHOR_PATTERNS: list[tuple[str, str]] = [
    # File paths: django/db/models/query.py, /testbed/django/..., etc.
    (r'\b[\w.-]+(?:/[\w.-]+)+\.py\b', ' '),
    # Python module paths in docstrings/code: django.db.models.query
    (r'\b[\w]+(?:\.[\w]+){2,}\b', ' '),
    # Shell commands: cd /testbed && python tests/runtests.py ...
    (r'(?:cd |python[23]? |pip |git |docker |conda |grep |find |cat |ls |sed |awk |nl |xargs |bash )'
     r'\S+(?:\s+\S+){0,15}', ' '),
    # Test runner invocations
    (r'python (?:-m )?(?:tests\.runtests|pytest|manage\.py test)\S*(?:\s+\S+){0,10}', ' '),
    # Git SHAs / commit hashes
    (r'\b[0-9a-f]{7,40}\b', ' '),
    # URL references to specific GitHub lines
    (r'https?://github\.com/\S+', ' '),
]
_CONCRETE_REPLACEMENTS = [(_re.compile(p), r) for p, r in _CONCRETE_ANCHOR_PATTERNS]


def _strip_concrete_terms_for_query(task_text: str) -> str:
    """Remove repository-specific concrete anchors from query task text.

    When ``use_anchor_in_query_embedding=False``, the query embedding should
    inhabit the same vocabulary space as the anchor-free memory key_embeddings.
    This function strips file paths, shell commands, test invocations, and
    other concrete terms that carry no abstract cognitive signal.
    """
    text = task_text
    for pattern, replacement in _CONCRETE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    # Collapse whitespace
    text = _re.sub(r'\s+', ' ', text)
    return text.strip()


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
        use_synergy_selection: bool = True,
        score_threshold_floor: float = 0.45,
        score_threshold_std: float = 0.5,
        min_memories: int = 1,
        attach_insights: bool = True,
        retrieval_source: str = "cognitive",
        excluded_dimensions: list | None = None,
        excluded_levels: list | None = None,
        use_anchor_in_query_embedding: bool = True,
    ):
        """
        Args:
            memory_dir: Path to directory containing *.pkl memory files
            alpha_semantic: Weight for semantic embedding similarity (vs LLM rerank score).
                            Default 0.35 biases toward LLM cognitive judgment to avoid
                            surface-similarity negative transfer.
            alpha_cognitive: Weight for dimension-aware cognitive relevance (vs semantic score).
                             Default 0.65 prioritizes LLM structural matching over embedding.
            alpha_dual_task: Within semantic retrieval, weight for task-level similarity
                             (key_embedding match) vs cognitive-level similarity
                             (cognitive_embedding match). 0.70 means 70% concrete + 30% abstract.
            top_n_candidates: How many candidates to retrieve before re-ranking
            top_k: Final number of memories to return
            use_cognitive_rerank: Whether to use LLM cognitive rerank (Step 3).
                                  When enabled, LLM evaluates each memory on its own cognitive
                                  dimension (causal/contrastive/strategic/environment), catching
                                  negative transfer cases that embedding similarity misses.
            use_llm_synergy: Whether to use LLM for conflict/synergy verification in
                             synergy-aware selection (Step 4). Adds ~1-3 extra LLM calls
                             but detects genuine memory conflicts and synergies.
            use_synergy_selection: Whether to use the synergy-aware selection algorithm
                                   at all (Step 4). When False, memories are selected by
                                   simple top-K combined_score without any redundancy
                                   penalty, complementarity bonus, or greedy selection.
                                   Default True. Set False for E4 (no-synergy) ablation.
            score_threshold_floor: Absolute minimum combined_score. Memories below
                                   this are always discarded regardless of distribution.
            score_threshold_std: Multiplier for standard deviation in relative
                                 threshold computation. Higher = stricter filtering.
                                 relative_threshold = median - std_mult * stdev
            min_memories: Minimum number of memories to keep (prevents empty result
                          when all scores are borderline).
            attach_insights: Whether to attach sibling insight memories to selected
                             concrete memories (Step 5). When False, only concrete
                             memories are returned without the abstract "why" layer.
                             Default True.
            retrieval_source: Which memory files to load from the pool.
                              "cognitive" (default): load 4x4 cognitive dimension
                                  files only (causal/contrastive/strategic/environment).
                              "traditional": load MTL-format files only
                                  (workflow/local/trajectory/summary/insight).
            excluded_dimensions: Cognitive dimensions to EXCLUDE from retrieval pool.
                                 Entries whose type matches are skipped during loading.
                                 For dimension leave-one-out ablation (e7--e10).
            excluded_levels: Abstraction levels to EXCLUDE from retrieval pool.
                             Entries whose level matches are filtered after loading.
                             For abstraction tier leave-one-out ablation (e11--e13).
            use_anchor_in_query_embedding: If False, strip concrete file paths,
                function names, and repository-specific vocabulary from the
                query task text BEFORE computing query_task_emb (channel 1 of
                dual-score retrieval).  Must match the corresponding
                ``use_anchor_in_embedding`` setting used during memory
                extraction so that query and memory vectors inhabit the same
                vocabulary space.  Default True.
        """
        self.memory_dir = Path(memory_dir)
        self.alpha_semantic = alpha_semantic
        self.alpha_cognitive = alpha_cognitive
        self.alpha_dual_task = alpha_dual_task
        self.top_n = top_n_candidates
        self.top_k = top_k
        self.use_cognitive_rerank = use_cognitive_rerank
        self.use_llm_synergy = use_llm_synergy
        self.use_synergy_selection = use_synergy_selection
        self.score_threshold_floor = score_threshold_floor
        self.score_threshold_std = score_threshold_std
        self.min_memories = min_memories
        self.attach_insights = attach_insights
        self.retrieval_source = retrieval_source
        self.excluded_dimensions = [d.strip().lower() for d in (excluded_dimensions or [])]
        self.excluded_levels = [l.strip().lower() for l in (excluded_levels or [])]
        self.use_anchor_in_query_embedding = use_anchor_in_query_embedding

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
        """Load all memory pkl files, separate insight layer, and precompute embeddings.

        When retrieval_source="cognitive" (default), only 4x4 cognitive dimension
        files (causal/contrastive/strategic/environment) are loaded.

        When retrieval_source="traditional", only MTL-format files
        (workflow/local/trajectory/summary/insight) are loaded.

        Insight-level memories are excluded from the retrieval pool because their
        highly abstract content cannot compete with concrete memories in embedding
        similarity. Instead, they are stored in _task_to_insights and attached as
        _linked_insights to their sibling concrete memories when those are selected.
        """
        # File name patterns for each retrieval source
        COGNITIVE_PATTERNS = {
            'causal_memory.pkl', 'contrastive_memory.pkl',
            'strategic_memory.pkl', 'environment_memory.pkl',
        }
        TRADITIONAL_PREFIXES = (
            'workflow_memory', 'local_memory', 'trajectory_memory',
            'summary_memory', 'insight_memory',
        )

        # Build the set of pkl files to exclude based on cognitive dimensions.
        # For dimension leave-one-out ablation (e7--e10), this prevents the
        # corresponding memory files from being loaded at all.
        _excluded_pkl_files: set[str] = set()
        if self.excluded_dimensions:
            for dim in self.excluded_dimensions:
                pkl_name = self._DIMENSION_TO_FILE.get(dim)
                if pkl_name:
                    _excluded_pkl_files.add(pkl_name)
            if _excluded_pkl_files:
                print(f"[retriever] Excluded dimensions: {self.excluded_dimensions}"
                      f"  (skipping: {sorted(_excluded_pkl_files)})")

        def _matches_source(filename: str) -> bool:
            if filename in _excluded_pkl_files:
                return False
            if self.retrieval_source == "cognitive":
                return filename in COGNITIVE_PATTERNS
            elif self.retrieval_source == "traditional":
                return filename.startswith(TRADITIONAL_PREFIXES)
            else:
                return True  # "all" or unknown -> load everything

        pkl_files = sorted(
            p for p in self.memory_dir.glob("*.pkl")
            if _matches_source(p.name)
        )
        if not pkl_files:
            # No files matched the requested source filter.  Try falling
            # back to whatever .pkl files ARE present so the experiment
            # doesn't crash just because the pool was populated by a
            # different extraction configuration.
            fallback_files = sorted(self.memory_dir.glob("*.pkl"))
            if fallback_files:
                print(f"[retriever] WARNING: no '{self.retrieval_source}' .pkl files "
                      f"found, falling back to all available pkl files "
                      f"({[p.name for p in fallback_files]})")
                pkl_files = fallback_files
                self.retrieval_source = "all"  # disable source filter for this session
            else:
                raise FileNotFoundError(
                    f"No .pkl files found in {self.memory_dir}"
                )

        print(f"[retriever] Loading {len(pkl_files)} memory files "
              f"(source={self.retrieval_source}) from {self.memory_dir}")

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
                        for alt_emb_key in ("generalized_query_embedding",
                                             "embedding"):
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
            print(f"  [retriever] Normalized {skipped_count} legacy entries "
                  f"(missing 'memory' wrapper key).")

        # Filter by excluded abstraction levels (e11--e13 ablation).
        # Applied BEFORE insight/concrete separation so that excluded
        # levels vanish from both the retrieval pool and the insight index.
        if self.excluded_levels:
            n_before = len(all_entries)
            all_entries = [
                e for e in all_entries
                if e.get("level", "").lower() not in self.excluded_levels
            ]
            n_after = len(all_entries)
            print(f"  [retriever] Excluded levels: {self.excluded_levels}"
                  f"  (filtered {n_before - n_after} entries, {n_after} remain)")

        # Separate insight-level memories from the retrieval pool.
        # Insight memories are abstract principles, anti-patterns, and
        # methodologies -- not directly actionable, cannot compete with
        # concrete memories in embedding similarity. They are stored in
        # a task-keyed index and will be attached as _linked_insights
        # when a sibling concrete memory is selected.
        insight_entries: list[dict] = []
        concrete_entries: list[dict] = []
        for entry in all_entries:
            if entry.get("level") == "insight":
                insight_entries.append(entry)
            else:
                concrete_entries.append(entry)

        # Build task+type -> insights index (for dimension-matched attachment)
        for entry in insight_entries:
            task = entry.get("task_name", "")
            dim = entry.get("type", "")
            if task:
                # Key by (task, type) so each concrete memory gets only
                # insights from the same cognitive dimension
                key = (task, dim)
                self._task_to_insights.setdefault(key, []).append(entry)

        # Build task -> concrete index (for optional extended linkage)
        for entry in concrete_entries:
            task = entry.get("task_name", "")
            if task:
                self._task_to_concrete.setdefault(task, []).append(entry)

        # Only concrete memories participate in retrieval
        self.memories = concrete_entries

        embeddings_list = []
        for mem in self.memories:
            emb = mem.get("key_embedding", [])
            if emb:
                embeddings_list.append(emb)
            else:
                embeddings_list.append([0.0] * 384)

        if embeddings_list:
            self._embeddings = np.array(embeddings_list, dtype=np.float32)

        # Build cognitive embedding array for dual-score retrieval.
        # These are pure-cognitive vectors (no task text, no anchors) that
        # capture the abstract cognitive pattern of each memory. Matching
        # query cognitive_profile against these vectors enables cross-repo
        # pattern retrieval (e.g., finding a causal principle from astropy
        # when the query is about a Django bug with isomorphic cause structure).
        cog_list = []
        cog_available = 0
        for mem in self.memories:
            cog_emb = mem.get("cognitive_embedding")
            if cog_emb:
                cog_list.append(cog_emb)
                cog_available += 1
            else:
                # For entries without cognitive_embedding (legacy or
                # cognitive_text was empty), fill with zero vector.
                # These memories will rely solely on task-level similarity.
                cog_list.append([0.0] * 384)
        if cog_list:
            self._cognitive_embeddings = np.array(cog_list, dtype=np.float32)
        print(f"  [retriever]   Cognitive embeddings available: {cog_available}/{len(self.memories)}")

        self._stats = {
            "total_memories": len(all_entries),
            "retrieval_pool": len(concrete_entries),
            "insight_attached": len(insight_entries),
            "tasks_with_insights": len(set(k[0] for k in self._task_to_insights.keys())),
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
    def _needs_llm_cognitive_query(self) -> bool:
        """Return True if an LLM call to extract cognitive query profile is needed.

        The cognitive query serves two purposes:
          1. Building cognitive_query_text for the cognitive embedding channel
             (channel 2 of dual-score retrieval).  Only needed when
             alpha_dual_task < 1.0 (i.e. cognitive channel has >0 weight).
          2. Building query_profile for dimension-aware cognitive rerank.
             Only needed when use_cognitive_rerank=True.

        When both are false (pure embedding, MTL traditional, random), the LLM
        call is a pure waste and can cause spurious failures.
        """
        # Cognitive rerank needs the profile
        if self.use_cognitive_rerank:
            return True
        # Cognitive embedding channel needs the query text when it has weight
        if self.alpha_dual_task < 1.0:
            return True
        return False

    def _extract_cognitive_profile_safe(self, task_text: str) -> dict:
        """Extract cognitive query profile, with fallback on failure.

        Returns a dict with (possibly empty) cognitive profile.  When the
        LLM call is not needed or fails, returns an empty/default profile
        so retrieval can continue with task-text-only embedding similarity.
        """
        if not self._needs_llm_cognitive_query():
            print("  [1/4] Cognitive query SKIPPED (embedding-only mode)")
            return {}

        print("  [1/4] Extracting cognitive query profile...")
        try:
            return extract_cognitive_query(task_text)
        except Exception as e:
            print(f"  [1/4] WARNING: cognitive query extraction failed: {e}")
            print(f"  [1/4] Falling back to task-text-only retrieval")
            return {}

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

        # Step 1: Extract cognitive profile from query (skipped for pure-embedding mode)
        query_profile = self._extract_cognitive_profile_safe(task_text)

        # Build cognitive query text from LLM-extracted profile.
        # This is the abstract cognitive signature of the query, used for
        # cross-repo pattern matching (channel 2 of dual-score retrieval).
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

        # Build TWO separate query embeddings for dual-score retrieval:
        #   Channel 1 (task): raw task text -> matches memory.key_embedding
        #   Channel 2 (cognitive): pure cognitive text -> matches memory.cognitive_embedding
        # This separation prevents the concrete task text from drowning out the
        # cognitive signal in the embedding vector, enabling cross-repo pattern matching.
        #
        # When use_anchor_in_query_embedding=False (e21+), strip concrete file
        # paths, shell commands, and test invocations from the query task text
        # so that query and memory key_embedding vectors inhabit the same
        # vocabulary space.
        _task_text_for_emb = (
            _strip_concrete_terms_for_query(task_text)
            if not self.use_anchor_in_query_embedding
            else task_text
        )
        query_task_emb = _get_embed_model().encode(_task_text_for_emb).tolist()
        query_cog_emb = _get_embed_model().encode(cognitive_query_text).tolist() if cognitive_query_text else None

        # Step 2: Dual-score semantic retrieval
        print(f"  [2/4] Dual-score semantic retrieval (top {self.top_n})...")
        candidates = self._semantic_retrieve(query_task_emb, query_cog_emb, self.top_n)

        # Step 3: Dimension-aware cognitive rerank (optional)
        if self.use_cognitive_rerank:
            print(f"  [3/4] Dimension-aware cognitive rerank (batch LLM)...")
            candidates = cognitive_rerank(
                query_profile, candidates,
                alpha_semantic=self.alpha_semantic,
                alpha_cognitive=self.alpha_cognitive,
            )
            # Recompute combined scores to ensure consistency
            # (cognitive_rerank also computes these, but we re-apply here
            #  with the identical weights to guard against stale values)
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

        # Step 4: Synergy-aware final selection (or simple top-K if disabled)
        if self.use_synergy_selection:
            print(f"  [4/4] Synergy-aware selection (top {k}, llm_synergy={self.use_llm_synergy})...")
            selected = synergy_aware_selection(candidates, k, use_llm=self.use_llm_synergy)
        else:
            print(f"  [4/4] Synergy-aware selection DISABLED, using simple top-{k} by score...")
            selected = candidates[:k]
            for mem in selected:
                mem["synergy_metadata"] = {"selected_by": "simple_topk"}

        quality = compute_selection_quality(selected)
        for mem in selected:
            mem["selection_quality"] = quality

        # Step 4.5: Dynamic threshold filtering.
        # Compute threshold from the full candidate distribution (top-N, reliable
        # statistics), then apply to the synergy-selected memories. This catches
        # borderline memories that passed embedding similarity but whose score
        # falls below the batch-adaptive cutoff.
        dynamic_threshold = self._compute_dynamic_threshold(candidates)
        qualified = [m for m in selected
                     if m.get("combined_score", 0) >= dynamic_threshold]
        n_discarded = len(selected) - len(qualified)

        # Fallback: ensure at least min_memories memories are returned.
        # When all scores are borderline, we keep the best one(s) rather than
        # returning nothing.
        if len(qualified) < self.min_memories:
            fallback = sorted(selected,
                              key=lambda m: m.get("combined_score", 0),
                              reverse=True)[:self.min_memories]
            qualified = fallback
            n_discarded = len(selected) - len(qualified)

        if n_discarded > 0:
            print(f"  [retriever] Dynamic threshold {dynamic_threshold:.3f} "
                  f"(floor={self.score_threshold_floor}, "
                  f"candidates median used for stats): "
                  f"discarded {n_discarded}, kept {len(qualified)}")

        selected = qualified

        # Store threshold metadata on each returned memory for logging/debugging
        for mem in selected:
            mem["_dynamic_threshold"] = round(dynamic_threshold, 4)

        # Step 5: Attach sibling insight memories to each selected concrete memory.
        # Insight-level memories (abstract principles, anti-patterns, methodologies)
        # are not directly retrievable, but when a concrete memory from the same
        # source task is selected, its sibling insights provide the "why" behind
        # the "what" -- turning a specific fix into a transferable lesson.
        # NOTE: Controlled by attach_insights flag (E26 ablation disables this).
        if self.attach_insights:
            for mem in selected:
                task = mem.get("task_name", "")
                dim = mem.get("type", "")
                key = (task, dim)
                linked = self._task_to_insights.get(key, [])
                mem["_linked_insights"] = linked
                if linked:
                    ins_dims = [ins.get("type", "?") for ins in linked]
                    print(f"    -> attached {len(linked)} insight(s) [{', '.join(ins_dims)}] "
                          f"from task={task}  (dimension-matched: {dim})")
        else:
            print(f"  [retriever] Insight attachment DISABLED (attach_insights=False)")

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

        This tests whether COGENT's retrieval pipeline is significantly better
        than random selection from a semantically relevant candidate set.
        """
        import random as _random_mod
        rng = _random_mod.Random(42)  # Fixed seed for reproducibility

        if not self.memories:
            print("[retriever-random] No memories loaded, returning empty.")
            return []

        print(f"\n[retriever-random] Query: {task_text[:100]}...")

        # Step 1: Extract cognitive profile (skipped for pure-embedding mode)
        print("  [R1] Extracting cognitive query profile...")
        query_profile = self._extract_cognitive_profile_safe(task_text)

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

        _task_text_for_emb = (
            _strip_concrete_terms_for_query(task_text)
            if not self.use_anchor_in_query_embedding
            else task_text
        )
        query_task_emb = _get_embed_model().encode(_task_text_for_emb).tolist()
        query_cog_emb = (
            _get_embed_model().encode(cognitive_query_text).tolist()
            if cognitive_query_text else None
        )

        # Step 2: Dual-score semantic retrieval (same as normal pipeline)
        print(f"  [R2] Dual-score semantic retrieval (top {top_n})...")
        candidates = self._semantic_retrieve(query_task_emb, query_cog_emb, top_n)

        # Step 3: Random selection from top-N
        print(f"  [R3] Randomly selecting {top_k} from {len(candidates)} candidates...")
        n_pick = min(top_k, len(candidates))
        selected = rng.sample(candidates, n_pick) if n_pick > 0 else []

        # Attach insights (respect the attach_insights flag, dimension-matched)
        if self.attach_insights:
            for mem in selected:
                task = mem.get("task_name", "")
                dim = mem.get("type", "")
                key = (task, dim)
                linked = self._task_to_insights.get(key, [])
                mem["_linked_insights"] = linked
                if linked:
                    ins_dims = [ins.get("type", "?") for ins in linked]
                    print(f"    -> attached {len(linked)} insight(s) [{', '.join(ins_dims)}] "
                          f"from task={task}  (dimension-matched: {dim})")

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
        to the LLM WITHOUT cognitive dimension labels.  The LLM is asked simply:
        "which of these memories are most relevant to the query?"  It does NOT
        see dimension tags, type labels, or facet structure.

        This tests whether the structured dimension-aware pipeline outperforms a
        flat LLM-as-selector approach on the same semantic candidate set.
        """
        import json as _json

        if not self.memories:
            print("[retriever-llm-direct] No memories loaded, returning empty.")
            return []

        print(f"\n[retriever-llm-direct] Query: {task_text[:100]}...")

        # Step 1: Extract cognitive profile (skipped for pure-embedding mode)
        print("  [D1] Extracting cognitive query profile...")
        query_profile = self._extract_cognitive_profile_safe(task_text)

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

        _task_text_for_emb = (
            _strip_concrete_terms_for_query(task_text)
            if not self.use_anchor_in_query_embedding
            else task_text
        )
        query_task_emb = _get_embed_model().encode(_task_text_for_emb).tolist()
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

        # Step 3: LLM directly selects best K without dimension structure.
        # Strip the [type=...] prefix so the LLM cannot use cognitive dimension labels.
        print(f"  [D3] LLM direct selection (flat, no dimension labels) from "
              f"{len(candidates)} candidates...")

        def _strip_dimension_label(brief: str) -> str:
            """Remove the [type=... level=... task=...] prefix line."""
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
                c["cognitive_score"] = 0.5  # neutral score
                c["cognitive_dimension"] = "llm_direct"
                c["cognitive_reason"] = "LLM direct selection (no dimension structure)"
                selected.append(c)

        # Fallback: fill from semantic top if fewer than top_k selected
        if len(selected) < top_k:
            for c in candidates:
                if c not in selected:
                    selected.append(c)
                    c["cognitive_score"] = 0.3
                    c["cognitive_dimension"] = "llm_direct_fallback"
                    c["cognitive_reason"] = "Fallback fill (LLM selected too few)"
                    if len(selected) >= top_k:
                        break

        # Attach insights (dimension-matched)
        if self.attach_insights:
            for mem in selected:
                task = mem.get("task_name", "")
                dim = mem.get("type", "")
                key = (task, dim)
                linked = self._task_to_insights.get(key, [])
                mem["_linked_insights"] = linked
                if linked:
                    ins_dims = [ins.get("type", "?") for ins in linked]
                    print(f"    -> attached {len(linked)} insight(s) "
                          f"[{', '.join(ins_dims)}] from task={task}  (dimension-matched: {dim})")

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
        """Compute a dynamic score threshold from the candidate distribution.

        Uses the full top-N candidate pool (not just the post-synergy K) for
        reliable statistics. The threshold is:

            max(absolute_floor, median - std_mult * stdev)

        This adapts to the quality of each retrieval batch:
        - High-quality batch (tight, high scores)  -> higher threshold
        - Low-quality batch (wide spread, low scores) -> floor acts as safety net

        Falls back to absolute floor when fewer than 3 candidates have non-zero
        scores (insufficient data for meaningful statistics).
        """
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
        """Dual-score semantic retrieval.

        Computes two independent cosine similarity scores for each memory:

        - task_score: query raw task text vs memory.key_embedding (three-section hybrid).
          Captures concrete similarity: same repo, same error pattern, same file paths.

        - cog_score: query cognitive profile vs memory.cognitive_embedding (pure abstract).
          Captures cognitive pattern similarity: same causal structure, same anti-pattern,
          regardless of repository or programming language.

        Combined with alpha_dual_task weighting: final = alpha * task + (1-alpha) * cog.
        Memories missing cognitive_embedding rely solely on task_score.

        This dual-channel approach ensures that cross-repo memories with isomorphic
        cognitive patterns can enter the top-N candidate set even when their concrete
        task text shares no vocabulary with the query.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return self.memories[:n]

        # --- Task-level similarities (channel 1) ---
        task_vec = np.array(query_task_emb, dtype=np.float32)
        t_norm = np.linalg.norm(task_vec)
        if t_norm > 0:
            task_vec = task_vec / t_norm

        _key_norms = np.linalg.norm(self._embeddings, axis=1)
        _key_norms = np.where(_key_norms == 0, 1.0, _key_norms)
        _key_normed = self._embeddings / _key_norms[:, np.newaxis]

        task_sims = np.dot(_key_normed, task_vec)
        task_sims = np.nan_to_num(task_sims, nan=0.0)

        # --- Cognitive-level similarities (channel 2) ---
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

        # --- Weighted combination ---
        alpha = self.alpha_dual_task
        final_sims = alpha * task_sims + (1.0 - alpha) * cog_sims

        top_indices = np.argsort(final_sims)[::-1][:n]

        candidates = []
        for idx in top_indices:
            mem = dict(self.memories[idx])
            # Store dual-score breakdown for transparency
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
        lines = [
            f"Memory Store: {self.memory_dir}",
            f"  Total memories: {self._stats['total_memories']}",
            f"  Retrieval pool: {self._stats['retrieval_pool']}",
            f"  Dimensions: {self._stats['dimensions']}",
            f"  Levels: {self._stats['levels']}",
            f"  Alpha: semantic={self.alpha_semantic} cognitive={self.alpha_cognitive} dual_task={self.alpha_dual_task}",
            f"  Cognitive rerank: {self.use_cognitive_rerank} | Synergy selection: {self.use_synergy_selection} | LLM synergy: {self.use_llm_synergy}",
            f"  Threshold: floor={self.score_threshold_floor} std_mult={self.score_threshold_std} min_mem={self.min_memories}",
        ]
        return "\n".join(lines)
