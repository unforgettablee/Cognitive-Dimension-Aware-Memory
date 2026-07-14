"""Memory Synergy & Conflict: detect how memories interact when used together.

Traditional retrieval picks top-K memories independently -- but memories are not
independent atoms. Two memories can:

- SYNERGIZE: cover complementary aspects of the same problem (e.g., one provides
  the causal diagnosis while another provides the repo-specific fix pattern).
- CONFLICT: recommend contradictory actions (e.g., one says "modify setup.py"
  while another says "don't touch setup.py -- the bug is in __init__.py").
- REDUNDANT: say essentially the same thing, wasting the precious top-K budget.

This module provides:
1. Heuristic synergy/conflict detection (fast, based on level+dimension patterns)
2. LLM-based conflict verification (for high-stakes pairs)
3. Greedy diversity-aware selection that maximizes the joint information value
   of the final K-memory set rather than individual relevance scores.
"""
import json
import os
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI

_deepseek_client: OpenAI | None = None
_client_lock = threading.Lock()


def _get_deepseek_client() -> OpenAI:
    """Lazily create and cache the DeepSeek client (thread-safe)."""
    global _deepseek_client
    if _deepseek_client is None:
        with _client_lock:
            if _deepseek_client is None:
                _deepseek_client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY"),
                    base_url=os.getenv("BASE_URL", "https://api.deepseek.com"),
                )
    return _deepseek_client


CONFLICT_CHECK_PROMPT = """\
You are a memory conflict detector. Two memories retrieved for the same coding
task may CONTRADICT each other -- recommending incompatible actions, diagnosing
different root causes that can't both be correct, or making claims about the
same code component that are mutually exclusive.

Read the two memories below and determine if they conflict.

Output a JSON object (no markdown) with this structure:
{
  "conflict": true/false,
  "conflict_type": "direct_negation|different_diagnosis|incompatible_actions|none",
  "explanation": "One sentence explaining the conflict or why there is none",
  "severity": "high|medium|low (high = using both would actively harm the task)",
  "resolution": "If conflicting, which one should be preferred and why"
}"""


SYNERGY_CHECK_PROMPT = """\
You are a memory synergy detector. Two memories retrieved for the same coding
task may COMPLEMENT each other -- one might explain WHY something fails (causal
diagnosis) while another explains HOW to navigate the repo to fix it (environment
knowledge). Together they provide more value than either alone.

Read the two memories below and determine if they synergize.

Output a JSON object (no markdown) with this structure:
{
  "synergy": true/false,
  "synergy_type": "dimension_complement|abstraction_bridge|example_principle_pair|none",
  "explanation": "One sentence explaining how these memories complement each other",
  "combined_value": "high|medium|low -- how much more useful they are together vs separately"
}"""


# ---------------------------------------------------------------
# Fast heuristic checks (no LLM call)
# ---------------------------------------------------------------

def _is_redundant(mem_a: dict, mem_b: dict) -> tuple[bool, float]:
    """Check if two memories are likely redundant based on structure.

    Returns (is_redundant, redundancy_score 0-1).
    """
    score = 0.0

    # Same level + same dimension = likely redundant
    if mem_a.get("level") == mem_b.get("level") and mem_a.get("type") == mem_b.get("type"):
        score += 0.5

    # Same task source = very likely redundant
    if mem_a.get("task_name") == mem_b.get("task_name"):
        score += 0.3

    # Check embedding similarity if available
    emb_a = mem_a.get("key_embedding", [])
    emb_b = mem_b.get("key_embedding", [])
    if emb_a and emb_b:
        dot = sum(x * y for x, y in zip(emb_a, emb_b))
        norm_a = sum(x * x for x in emb_a) ** 0.5
        norm_b = sum(x * x for x in emb_b) ** 0.5
        if norm_a > 0 and norm_b > 0:
            cos_sim = dot / (norm_a * norm_b)
            if cos_sim > 0.92:
                score += 0.4
            elif cos_sim > 0.85:
                score += 0.2

    return score >= 0.5, min(score, 1.0)


def _is_complementary(mem_a: dict, mem_b: dict) -> tuple[bool, float]:
    """Fast heuristic check for complementary memory pairs.

    Returns (is_complementary, complementarity_score 0-1).
    """
    score = 0.0

    level_a = mem_a.get("level", "")
    level_b = mem_b.get("level", "")
    dim_a = mem_a.get("type", "")
    dim_b = mem_b.get("type", "")

    # Same task + different dimensions = complementary perspectives
    if mem_a.get("task_name") == mem_b.get("task_name") and dim_a != dim_b:
        score += 0.6

    # One is causal, other is environment = classic complement
    if {dim_a, dim_b} == {"causal", "environment"}:
        score += 0.3

    # One is causal, other is strategic = understanding + methodology
    if {dim_a, dim_b} == {"causal", "strategic"}:
        score += 0.25

    # Different abstraction levels of same dimension = depth
    if dim_a == dim_b and level_a != level_b:
        score += 0.2

    # Same task with different levels = multi-granularity view
    if mem_a.get("task_name") == mem_b.get("task_name") and level_a != level_b:
        score += 0.2

    return score >= 0.4, min(score, 1.0)


# ---------------------------------------------------------------
# LLM-based verification (for important pairs)
# ---------------------------------------------------------------

def _build_memory_preview(mem: dict) -> str:
    """Build a compact text preview of a memory for LLM evaluation.

    Includes both abstract cognitive content AND concrete anchors (file paths,
    function names, error patterns) so the LLM can evaluate real similarity.
    """
    parts = [
        f"Level: {mem.get('level', '?')}",
        f"Dimension: {mem.get('type', '?')}",
        f"Task: {mem.get('task_name', '?')}",
    ]
    memory_data = mem.get("memory", {})
    if isinstance(memory_data, dict):
        for key in ["summary", "causal_graph_summary", "causal_principle",
                     "principle_statement", "core_idea", "meta_strategy",
                     "success_failure_boundary", "transition_insight"]:
            if key in memory_data and memory_data[key]:
                val = str(memory_data[key])[:300]
                parts.append(f"{key}: {val}")
                break

    # Include concrete anchors for actionable similarity evaluation
    anchors = mem.get("concrete_anchors", {}) or memory_data.get("concrete_anchors", {}) or memory_data.get("concrete_origin", {}) or {}
    if anchors:
        anchor_items = []
        for k, v in anchors.items():
            if isinstance(v, list):
                anchor_items.append(f"{k}: {', '.join(str(x)[:100] for x in v[:3])}")
            elif isinstance(v, str) and v:
                anchor_items.append(f"{k}: {str(v)[:200]}")
        if anchor_items:
            parts.append("[ANCHORS] " + " | ".join(anchor_items[:5]))

    return "\n".join(parts)


def detect_conflict_llm(mem_a: dict, mem_b: dict) -> dict:
    """Use LLM to check for genuine conflict between two memories."""
    preview_a = _build_memory_preview(mem_a)
    preview_b = _build_memory_preview(mem_b)
    try:
        response = _get_deepseek_client().chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": CONFLICT_CHECK_PROMPT},
                {"role": "user", "content": (
                    f"### Memory A:\n{preview_a}\n\n### Memory B:\n{preview_b}"
                )},
            ],
            timeout=60.0,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
        return json.loads(raw)
    except Exception:
        return {"conflict": False, "conflict_type": "none",
                "explanation": "LLM check failed", "severity": "low", "resolution": ""}


def detect_synergy_llm(mem_a: dict, mem_b: dict) -> dict:
    """Use LLM to check for genuine synergy between two memories."""
    preview_a = _build_memory_preview(mem_a)
    preview_b = _build_memory_preview(mem_b)
    try:
        response = _get_deepseek_client().chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": SYNERGY_CHECK_PROMPT},
                {"role": "user", "content": (
                    f"### Memory A:\n{preview_a}\n\n### Memory B:\n{preview_b}"
                )},
            ],
            timeout=60.0,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
        return json.loads(raw)
    except Exception:
        return {"synergy": False, "synergy_type": "none",
                "explanation": "LLM check failed", "combined_value": "medium"}


# ---------------------------------------------------------------
# Pairwise interaction evaluation
# ---------------------------------------------------------------

def evaluate_pair(mem_a: dict, mem_b: dict, use_llm: bool = False) -> dict:
    """Evaluate the interaction between two memory entries.

    Returns a dict with:
      - redundant: bool
      - redundant_score: float 0-1
      - complementary: bool
      - complementary_score: float 0-1
      - conflict: bool (only set if use_llm=True)
      - conflict_info: dict (only if use_llm=True)
      - synergy: bool (only set if use_llm=True)
      - synergy_info: dict (only if use_llm=True)
      - interaction_score: float (-1 to +1, negative = conflict, positive = synergy)
    """
    result = {}

    # Fast heuristic checks (always run)
    is_red, red_score = _is_redundant(mem_a, mem_b)
    is_comp, comp_score = _is_complementary(mem_a, mem_b)
    result["redundant"] = is_red
    result["redundant_score"] = red_score
    result["complementary"] = is_comp
    result["complementary_score"] = comp_score

    # Default interaction score from heuristics
    interaction = 0.0
    if is_comp:
        interaction += comp_score * 0.8
    if is_red:
        interaction -= red_score * 0.5

    # LLM verification for important pairs
    if use_llm:
        # Check conflict if memories recommend different things about same topic
        if is_red and comp_score < 0.3:
            conflict_info = detect_conflict_llm(mem_a, mem_b)
            result["conflict"] = conflict_info.get("conflict", False)
            result["conflict_info"] = conflict_info
            if result["conflict"]:
                severity = conflict_info.get("severity", "low")
                penalty = {"high": 0.9, "medium": 0.6, "low": 0.3}[severity]
                interaction -= penalty

        # Check synergy if pair looks complementary
        if is_comp and not is_red:
            synergy_info = detect_synergy_llm(mem_a, mem_b)
            result["synergy"] = synergy_info.get("synergy", False)
            result["synergy_info"] = synergy_info
            if result["synergy"]:
                combined = synergy_info.get("combined_value", "medium")
                bonus = {"high": 0.4, "medium": 0.25, "low": 0.1}[combined]
                interaction += bonus

    result["interaction_score"] = max(-1.0, min(1.0, interaction))
    return result


# ---------------------------------------------------------------
# Synergy-aware selection
# ---------------------------------------------------------------

def synergy_aware_selection(
    candidates: list[dict],
    top_k: int = 3,
    use_llm: bool = False,
    redundancy_penalty: float = 0.3,
    complementarity_bonus: float = 0.2,
) -> list[dict]:
    """Select top-K memories maximizing joint information value.

    Algorithm:
      1. Pick the highest-scoring candidate first
      2. For each subsequent pick, adjust scores by:
         - Penalizing redundancy with already-selected memories
         - Boosting complementarity with already-selected memories
      3. Repeat until K memories are selected

    Args:
        candidates: List of memory dicts, each must have at least 'combined_score'
        top_k: Number of memories to select
        use_llm: If True, use LLM for conflict/synergy verification
        redundancy_penalty: How strongly to penalize redundancy
        complementarity_bonus: How strongly to reward complementarity

    Returns:
        Selected memories in order, each with synergy_metadata attached.
    """
    if len(candidates) <= top_k:
        for c in candidates:
            c["synergy_metadata"] = {"selected_by": "sole_candidate"}
        return candidates[:top_k]

    selected = []
    remaining = list(candidates)

    for selection_round in range(top_k):
        if not remaining:
            break

        if not selected:
            # First pick: highest individual score
            best = remaining.pop(0)
            best["synergy_metadata"] = {"selected_by": "highest_score", "round": selection_round}
            selected.append(best)
            continue

        # Evaluate each remaining candidate against the selected set
        best_candidate = None
        best_adjusted_score = -float("inf")

        for cand in remaining:
            total_interaction = 0.0
            interaction_details = []

            for sel in selected:
                pair_result = evaluate_pair(cand, sel, use_llm=use_llm)

                # Penalize redundancy
                if pair_result["redundant"]:
                    total_interaction -= redundancy_penalty * pair_result["redundant_score"]

                # Reward complementarity
                if pair_result["complementary"]:
                    total_interaction += complementarity_bonus * pair_result["complementary_score"]

                # Apply conflict penalty if detected
                if pair_result.get("conflict"):
                    severity = pair_result.get("conflict_info", {}).get("severity", "low")
                    penalty = {"high": 0.5, "medium": 0.3, "low": 0.1}[severity]
                    total_interaction -= penalty

                # Apply synergy bonus if detected
                if pair_result.get("synergy"):
                    combined = pair_result.get("synergy_info", {}).get("combined_value", "medium")
                    bonus = {"high": 0.3, "medium": 0.15, "low": 0.05}[combined]
                    total_interaction += bonus

                interaction_details.append(pair_result)

            base_score = cand.get("combined_score", cand.get("semantic_score", 0.5))
            adjusted_score = base_score + total_interaction

            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_candidate = cand
                best_candidate["_adjusted_score"] = adjusted_score
                best_candidate["_interaction_details"] = interaction_details

        if best_candidate:
            remaining.remove(best_candidate)
            best_candidate["synergy_metadata"] = {
                "selected_by": "synergy_aware",
                "round": selection_round,
                "base_score": best_candidate.get("combined_score", best_candidate.get("semantic_score", 0.5)),
                "adjusted_score": best_adjusted_score,
                "interaction_details": best_candidate.pop("_interaction_details", []),
            }
            selected.append(best_candidate)

    return selected


def compute_selection_quality(selected: list[dict]) -> dict:
    """Compute quality metrics for a selected memory set."""
    if len(selected) < 2:
        return {"diversity": 1.0, "coverage": 1.0, "conflict_free": True}

    # Dimension coverage: how many different dimensions are represented
    dims = {m.get("type", "") for m in selected}
    levels = {m.get("level", "") for m in selected}

    # Redundancy check
    total_redundancy = 0.0
    total_complementarity = 0.0
    pairs = 0
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            is_red, red_score = _is_redundant(selected[i], selected[j])
            is_comp, comp_score = _is_complementary(selected[i], selected[j])
            total_redundancy += red_score
            total_complementarity += comp_score
            pairs += 1

    avg_redundancy = total_redundancy / pairs if pairs else 0
    avg_complementarity = total_complementarity / pairs if pairs else 0

    return {
        "dimension_coverage": len(dims),
        "dimensions": list(dims),
        "level_coverage": len(levels),
        "levels": list(levels),
        "avg_redundancy": round(avg_redundancy, 3),
        "avg_complementarity": round(avg_complementarity, 3),
        "num_pairs": pairs,
        "diversity_score": round(1.0 - avg_redundancy + avg_complementarity, 3),
        "conflict_free": True,  # Will be updated if LLM checks find conflicts
    }
