"""Cognitive Rerank: dimension-aware memory relevance scoring via LLM.

Each memory has a type (causal / contrastive / strategic / environment) that
captures a different cognitive perspective. When retrieving memories for a new
task, each memory must be scored on its OWN dimension's relevance to the query.

Key principle: a contrastive memory is evaluated on whether its anti-patterns
are relevant, not on whether its causal chain matches the query.

LLM settings (API key, base URL, model) are read from mtl.llm.
Configure via: mtl.llm.configure(api_key=..., base_url=..., model=...)
or environment variables: MTL_LLM_API_KEY, MTL_LLM_BASE_URL, MTL_LLM_MODEL.
"""
import json
from mtl.llm import get_client, get_model


# ---------------------------------------------------------------
# Prompt: Extract cognitive profile from query (1 LLM call)
# ---------------------------------------------------------------
# The prompt is composed from dimension-specific parts so that excluded
# dimensions (e.g. for e8-e11 ablation experiments) can be selectively
# removed before the LLM call.

_ALL_QUERY_DIMENSIONS = ["causal", "contrastive", "strategic", "environment"]

_QUERY_PROMPT_PREFIX = """\
You are analyzing a coding task to determine what KIND of prior experience would
be most helpful. Given a task description, assess which cognitive dimensions are
needed and extract the relevant structural information.

Output a JSON object (no markdown) with this structure:
{
  "task_type": "debug|feature|refactor|config|test|other","""

_QUERY_PROMPT_CAUSAL = """
  "need_causal": true/false,
  "causal_signature": {
    "error_category": "type_error|import_error|logic_error|runtime_error|test_failure|build_error|performance|other",
    "cause_category": "missing_dependency|version_mismatch|incorrect_assumption|side_effect|race_condition|off_by_one|api_misuse|config_error|other",
    "intervention_type": "add_dependency|modify_logic|reorder_operations|add_validation|change_config|refactor_structure|other",
    "causal_chain": ["abstract step 1", "abstract step 2", "..."],
    "causal_signature": "One sentence in domain-independent causal language"
  },"""

_QUERY_PROMPT_CONTRASTIVE = """
  "need_contrastive": true/false,
  "contrastive_needs": "What kind of failure patterns or anti-patterns would be relevant? Leave empty if not needed.","""

_QUERY_PROMPT_STRATEGIC = """
  "need_strategic": true/false,
  "strategic_needs": "What kind of methodology or decision strategy would help? Leave empty if not needed.","""

_QUERY_PROMPT_ENVIRONMENT = """
  "need_environment": true/false,
  "environment_needs": "What repo/tool knowledge would help? Leave empty if not needed.","""

_QUERY_PROMPT_SUFFIX = """
  "query_summary": "One sentence describing the essence of this task"
}"""

# Mapping for dimension-specific prompt parts (used by _build_query_prompt).
_DIM_PROMPT_PARTS: dict[str, str] = {
    "causal": _QUERY_PROMPT_CAUSAL,
    "contrastive": _QUERY_PROMPT_CONTRASTIVE,
    "strategic": _QUERY_PROMPT_STRATEGIC,
    "environment": _QUERY_PROMPT_ENVIRONMENT,
}

# Full prompt (all dimensions active) — kept for backward compatibility.
COGNITIVE_QUERY_PROMPT = (
    _QUERY_PROMPT_PREFIX
    + _QUERY_PROMPT_CAUSAL
    + _QUERY_PROMPT_CONTRASTIVE
    + _QUERY_PROMPT_STRATEGIC
    + _QUERY_PROMPT_ENVIRONMENT
    + _QUERY_PROMPT_SUFFIX
)


def _build_query_prompt(excluded_dimensions: list[str] | None) -> str:
    """Build a cognitive query prompt with only the active dimensions.

    When ``excluded_dimensions`` is empty or None, returns the full prompt
    (identical to ``COGNITIVE_QUERY_PROMPT``).  Otherwise, the excluded
    dimension blocks are omitted so the LLM is never asked to produce them.
    """
    excluded = [d.strip().lower() for d in (excluded_dimensions or [])]
    if not excluded:
        return COGNITIVE_QUERY_PROMPT

    active = [d for d in _ALL_QUERY_DIMENSIONS if d not in excluded]
    parts = [_QUERY_PROMPT_PREFIX]
    for dim in active:
        parts.append(_DIM_PROMPT_PARTS[dim])
    parts.append(_QUERY_PROMPT_SUFFIX)
    return "".join(parts)


def _post_process_query_profile(profile: dict, excluded_dimensions: list[str] | None) -> dict:
    """Zero out excluded dimensions in the query profile.

    Defense in depth: even with a filtered prompt the LLM may occasionally
    include excluded dimensions.  This post-processing guarantees that
    ``need_<dim>`` is False and the corresponding fields are cleared.
    """
    excluded = [d.strip().lower() for d in (excluded_dimensions or [])]
    if not excluded:
        return profile

    for dim in excluded:
        profile[f"need_{dim}"] = False
        if dim == "causal":
            profile["causal_signature"] = {}
        else:
            profile[f"{dim}_needs"] = ""

    return profile


def extract_cognitive_query(task_text: str,
                            excluded_dimensions: list[str] | None = None) -> dict:
    """Extract cognitive profile and causal signature from a query. 1 LLM call.

    Args:
        task_text: The task description / instruction text.
        excluded_dimensions: Cognitive dimensions to exclude from the query
            profile (e.g. ``["causal"]`` for e8).  The LLM prompt is filtered
            so these dimensions are not requested, and the result is
            post-processed as a defense-in-depth guarantee.
    """
    excluded = [d.strip().lower() for d in (excluded_dimensions or [])]
    prompt = _build_query_prompt(excluded)

    try:
        response = get_client().chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"### Task:\n{task_text}"},
            ],
            timeout=120.0,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        profile = json.loads(raw)
        return _post_process_query_profile(profile, excluded)
    except Exception:
        print("  [cognitive] Failed to extract cognitive query, using fallback")
        profile = {
            "task_type": "other",
            "need_causal": True, "need_contrastive": True,
            "need_strategic": True, "need_environment": True,
            "causal_signature": {
                "error_category": "other", "cause_category": "other",
                "intervention_type": "other",
                "causal_chain": [task_text[:200]],
                "causal_signature": task_text[:200],
            },
            "contrastive_needs": "", "strategic_needs": "", "environment_needs": "",
            "query_summary": task_text[:200],
            "_fallback": True,
        }
        return _post_process_query_profile(profile, excluded)


# ---------------------------------------------------------------
# Memory cognitive brief (zero LLM calls -- from stored data)
# ---------------------------------------------------------------
def _build_concrete_anchor_brief(anchors: dict | None) -> str:
    """Build a compact text representation of concrete anchors for LLM evaluation."""
    if not anchors:
        return ""
    parts = []
    for key in ["files_involved", "files_modified", "key_files_paths",
                "critical_files", "key_discovery_files", "key_decision_files"]:
        vals = anchors.get(key, [])
        if vals:
            parts.append(f"Files: {', '.join(str(v) for v in vals[:5])}")
            break
    for key in ["key_functions", "error_signature", "error_pattern",
                "fix_pattern", "test_command", "build_command",
                "test_commands", "repo_root_structure"]:
        val = anchors.get(key, "")
        if val:
            if isinstance(val, list):
                parts.append(f"{key}: {'; '.join(str(v) for v in val[:3])}")
            else:
                parts.append(f"{key}: {str(val)[:200]}")
    for key in ["actual_failed_command", "actual_successful_command",
                "diagnostic_commands_used", "tool_sequence",
                "error_recovery_commands"]:
        val = anchors.get(key, "")
        if val:
            if isinstance(val, list):
                parts.append(f"{key}: {'; '.join(str(v) for v in val[:3])}")
            else:
                parts.append(f"{key}: {str(val)[:200]}")
    return " | ".join(parts) if parts else ""


def _build_dimension_brief(memory_entry: dict) -> str:
    """Build a compact brief from stored memory, focused on its native dimension."""
    mem = memory_entry.get("memory", {})
    level = memory_entry.get("level", "")
    dim = memory_entry.get("type", "")
    parts = [f"[type={dim} level={level} task={memory_entry.get('task_name','?')}]"]

    anchors = memory_entry.get("concrete_anchors", {}) or {}
    if not anchors:
        anchors = mem.get("concrete_anchors", {}) or {}
        if not anchors:
            anchors = mem.get("concrete_origin", {}) or {}
    concrete_text = _build_concrete_anchor_brief(anchors)

    if dim == "causal":
        if level == "trajectory":
            chains = mem.get("causal_chains", [])
            for c in chains[:3]:
                parts.append(
                    f"Causal chain: {c.get('trigger_command','')} -> "
                    f"{c.get('conclusion_drawn','')[:120]} -> {c.get('next_action','')[:120]}"
                )
            parts.append(f"Summary: {mem.get('summary','')}")
        elif level == "workflow":
            for cp in mem.get("critical_path", [])[:3]:
                parts.append(f"Critical: {cp.get('step','')[:120]} (depends on: {cp.get('dependency','')})")
            parts.append(f"Causal graph: {mem.get('causal_graph_summary','')}")
        elif level == "summary":
            rc = mem.get("root_cause", {})
            res = mem.get("resolution", {})
            parts.append(
                f"Root cause: {rc.get('what_was_wrong','')} because {rc.get('why_it_was_wrong','')}. "
                f"Fix: {res.get('approach','')}. Bug class: {mem.get('bug_class','')}. "
                f"Principle: {mem.get('causal_principle','')}"
            )
        elif level == "insight":
            parts.append(f"Principle: {mem.get('principle_name','')}: {mem.get('principle_statement','')}")
            parts.append(f"Applies: {'; '.join(mem.get('when_applies',[]))}")

    elif dim == "contrastive":
        if level == "trajectory":
            for f in mem.get("failed_commands", [])[:2]:
                parts.append(f"Failed: {f.get('command_pattern','')} - {f.get('what_went_wrong','')[:100]}")
            for s in mem.get("successful_commands", [])[:2]:
                parts.append(f"Worked: {s.get('command_pattern','')} - {s.get('what_worked','')[:100]}")
            parts.append(f"Lesson: {mem.get('transition_insight','')}")
        elif level == "workflow":
            cw = mem.get("chosen_workflow", {})
            parts.append(f"Chosen: {cw.get('description','')[:120]}. Strength: {cw.get('strength','')[:100]}")
            for alt in mem.get("alternative_workflows", [])[:2]:
                parts.append(f"Alternative: {alt.get('description','')[:100]} outcome={alt.get('likely_outcome','')}")
            for rule in mem.get("workflow_decision_rules", [])[:2]:
                parts.append(f"Rule: {rule}")
        elif level == "summary":
            parts.append(f"Outcome: {mem.get('outcome','')}. Boundary: {mem.get('success_failure_boundary','')}")
            if_pass = mem.get("if_pass", {}) or {}
            parts.append(f"Success factors: {if_pass.get('key_success_factors',[])}")
        elif level == "insight":
            for ap in mem.get("anti_patterns", [])[:2]:
                parts.append(
                    f"Anti-pattern '{ap.get('name','')}': {ap.get('description','')[:120]}. "
                    f"Fix: {ap.get('escape_strategy','')[:100]}"
                )
            parts.append(f"Correct pattern: {mem.get('positive_pattern','')}")

    elif dim == "strategic":
        if level == "trajectory":
            for p in mem.get("phases", [])[:3]:
                parts.append(f"Phase [{p.get('phase','')}]: {p.get('purpose','')[:100]}")
            for pp in mem.get("pivoting_points", [])[:2]:
                parts.append(f"Pivot: {pp.get('strategy_change','')[:120]}")
            parts.append(f"Efficiency: {mem.get('efficiency_assessment','')}")
        elif level == "workflow":
            parts.append(f"Rationale: {mem.get('workflow_rationale','')[:150]}")
            parts.append(f"Info gathering: {mem.get('information_gathering_strategy','')[:100]}")
            parts.append(f"Risk mgmt: {mem.get('risk_management','')[:100]}")
            for step in mem.get("template_workflow", [])[:3]:
                parts.append(f"Template: {step}")
        elif level == "summary":
            parts.append(f"Meta-strategy: {mem.get('meta_strategy','')}")
            for cm in mem.get("cognitive_moves", [])[:3]:
                parts.append(f"Cognitive move: {cm}")
            parts.append(f"Error recovery: {mem.get('error_recovery_strategy','')}")
        elif level == "insight":
            parts.append(f"Methodology: {mem.get('methodology_name','')}: {mem.get('core_idea','')}")
            for dr in mem.get("decision_rules", [])[:3]:
                parts.append(f"Decision rule: {dr}")
            parts.append(f"Use when: {mem.get('when_to_use','')}")

    elif dim == "environment":
        if level == "trajectory":
            for d in mem.get("discovery_commands", [])[:2]:
                parts.append(f"Discovered: {d.get('what_it_revealed','')[:100]}")
            for p in mem.get("useful_patterns", [])[:2]:
                parts.append(f"Pattern: {p}")
            for w in mem.get("wasted_commands", [])[:2]:
                parts.append(f"Wasted: {w}")
            parts.append(f"Init: {mem.get('repo_initialization','')}")
        elif level == "workflow":
            for st in mem.get("repo_specific_steps", [])[:2]:
                parts.append(f"Repo step: {st.get('step','')} because {st.get('why_specific','')[:80]}")
            for rule in mem.get("repo_adaptation_rules", [])[:2]:
                parts.append(f"Adaptation: {rule}")
            parts.append(f"Tool chain: {mem.get('tool_chain','')}")
        elif level == "summary":
            profile = mem.get("repo_profile", {})
            parts.append(f"Repo: lang={profile.get('language','')} size={profile.get('size_hint','')}")
            for fg in mem.get("footguns", [])[:2]:
                parts.append(f"Footgun: {fg}")
            for kf in mem.get("key_files", [])[:2]:
                parts.append(f"Key file: {kf.get('path','')} - {kf.get('why_important','')[:80]}")
        elif level == "insight":
            for ep in mem.get("environment_patterns", [])[:2]:
                parts.append(f"Env pattern: {ep.get('pattern','')}. Generalizes: {ep.get('generalization','')}")
            for adv in mem.get("tool_agnostic_advice", [])[:2]:
                parts.append(f"Advice: {adv}")
            parts.append(f"Lang transfer: {mem.get('language_transfer','')}")

    if concrete_text:
        parts.append(f"[CONCRETE] {concrete_text}")

    return "\n".join(parts)


# ---------------------------------------------------------------
# Prompt: Batch dimension-aware cognitive relevance scoring
# ---------------------------------------------------------------
BATCH_COGNITIVE_RERANK_PROMPT = """\
You are a COGNITIVE MEMORY EVALUATOR. Given a coding task query and a set of
retrieved memories, rate each memory on how relevant its specific cognitive
content is to the query.

IMPORTANT: Each memory has a TYPE that determines HOW it should be evaluated:

- type=causal: Rate whether the CAUSE-EFFECT LOGIC in this memory matches the
  query. Look for structural isomorphism in the causal chain -- same pattern of
  "A causes B which causes failure", same diagnostic reasoning, same fix mechanism.
  Do NOT match on keywords or language names.

- type=contrastive: Rate whether the ANTI-PATTERNS, FAILURE MODES, or
  SUCCESS/FAILURE BOUNDARIES in this memory are relevant to the query. Would
  knowing "what NOT to do" or "what distinguishes success from failure" help
  with this task?

- type=strategic: Rate whether the METHODOLOGY, DECISION STRATEGY, or WORKFLOW
  PATTERN in this memory transfers to the query. Would the same problem-solving
  approach, information-gathering strategy, or error-recovery tactic apply?

- type=environment: Rate whether the REPO KNOWLEDGE, TOOL PATTERNS, or
  ENVIRONMENT-SPECIFIC INSIGHTS in this memory would help with the query. Is the
  repo structure, tool chain, or footgun knowledge applicable?

DIMENSION PRIORITY: The query profile specifies which cognitive dimensions are
NEEDED for this task (need_causal / need_contrastive / need_strategic /
need_environment). Memories in NEEDED dimensions should receive higher scores
when their content is relevant. Memories in NON-NEEDED dimensions should be
down-weighted unless they reveal exceptionally relevant insight.

Rate each memory 0-1 on its OWN dimension's relevance:
- 0.8-1.0: Highly relevant; the cognitive insight directly applies
- 0.5-0.7: Moderately relevant; partially applicable
- 0.2-0.4: Weakly relevant; superficial overlap
- 0.0-0.1: Not relevant

Output a JSON object (no markdown):
{
  "comparisons": [
    {
      "memory_index": 0,
      "dimension": "causal",
      "relevance_score": 0.75,
      "reason": "One sentence explaining why this memory is or isn't relevant on its dimension"
    }
  ]
}"""


def _batch_cognitive_compare(query_profile: dict, memory_briefs: list[str]) -> list[dict]:
    """Batch LLM comparison: query profile vs N memory briefs."""
    if not memory_briefs:
        return []

    needed_dims = []
    for dim in ["causal", "contrastive", "strategic", "environment"]:
        if query_profile.get(f"need_{dim}"):
            needed_dims.append(dim)
    q_parts = [
        f"Task type: {query_profile.get('task_type', '?')}",
        f"Summary: {query_profile.get('query_summary', '')}",
        f"DIMENSION PRIORITY: The following dimensions are NEEDED for this task: {', '.join(needed_dims) if needed_dims else 'all'}. Memories from needed dimensions should be scored higher; memories from non-needed dimensions should be scored lower.",
    ]
    cs = query_profile.get("causal_signature", {})
    if cs:
        q_parts.extend([
            f"Causal -- error_category: {cs.get('error_category','')}",
            f"Causal -- cause_category: {cs.get('cause_category','')}",
            f"Causal -- intervention_type: {cs.get('intervention_type','')}",
            f"Causal -- chain: {' -> '.join(cs.get('causal_chain',[]))}",
            f"Causal -- signature: {cs.get('causal_signature','')}",
        ])
    if query_profile.get("contrastive_needs"):
        q_parts.append(f"Contrastive needs: {query_profile['contrastive_needs']}")
    if query_profile.get("strategic_needs"):
        q_parts.append(f"Strategic needs: {query_profile['strategic_needs']}")
    if query_profile.get("environment_needs"):
        q_parts.append(f"Environment needs: {query_profile['environment_needs']}")
    query_desc = "\n".join(q_parts)

    memory_desc = "\n\n".join(
        f"### Memory {i}\n{brief}" for i, brief in enumerate(memory_briefs)
    )

    try:
        response = get_client().chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": BATCH_COGNITIVE_RERANK_PROMPT},
                {"role": "user", "content": (
                    f"### Query Profile:\n{query_desc}\n\n"
                    f"### Memories to Evaluate:\n{memory_desc}"
                )},
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
        result = json.loads(raw)
        return result.get("comparisons", [])
    except Exception as e:
        print(f"  [cognitive] Batch LLM comparison failed: {e}")
        return []


def _fallback_dimension_match(query_profile: dict, memory_briefs: list[str]) -> list[dict]:
    """Fallback: category-based scoring per dimension."""
    results = []
    cs = query_profile.get("causal_signature", {})
    q_error = cs.get("error_category", "")
    q_cause = cs.get("cause_category", "")
    q_fix = cs.get("intervention_type", "")

    for i, brief in enumerate(memory_briefs):
        brief_lower = brief.lower()
        score = 0.0

        hits = 0
        for q_val in [q_error, q_cause, q_fix]:
            if q_val and q_val != "other" and q_val.replace("_", " ") in brief_lower:
                hits += 1
        if hits >= 2:
            score = 0.6
        elif hits == 1:
            score = 0.35

        for need_field in ["contrastive_needs", "strategic_needs", "environment_needs"]:
            need_text = query_profile.get(need_field, "")
            if need_text and any(w in brief_lower for w in need_text.lower().split()[:5]):
                score = max(score, 0.4)

        results.append({
            "memory_index": i,
            "dimension": "unknown",
            "relevance_score": score,
            "reason": "Fallback: keyword-based (LLM unavailable)",
        })
    return results


# ---------------------------------------------------------------
# Cognitive rerank
# ---------------------------------------------------------------
def cognitive_rerank(
    query_profile: dict,
    candidates: list[dict],
    alpha_semantic: float = 0.35,
    alpha_cognitive: float = 0.65,
) -> list[dict]:
    """Re-rank candidates using dimension-aware cognitive relevance.

    Each memory is scored on its own dimension:
    - causal memories -> causal structure relevance
    - contrastive memories -> anti-pattern / boundary relevance
    - strategic memories -> methodology transfer relevance
    - environment memories -> repo/tool knowledge relevance

    Args:
        query_profile: P(q) dict with need_causal, need_contrastive, etc.
        candidates: Candidates from semantic retrieval with semantic_score.
        alpha_semantic: Weight for semantic embedding score (default 0.35).
        alpha_cognitive: Weight for LLM cognitive relevance score (default 0.65).
            combined_score = alpha_semantic * semantic + alpha_cognitive * cognitive
    """
    if not candidates:
        return candidates

    memory_briefs = [_build_dimension_brief(c) for c in candidates]
    comparisons = _batch_cognitive_compare(query_profile, memory_briefs)

    if not comparisons:
        print("  [cognitive] Using fallback dimension matching")
        comparisons = _fallback_dimension_match(query_profile, memory_briefs)

    comp_lookup = {}
    for comp in comparisons:
        idx = comp.get("memory_index", -1)
        if 0 <= idx < len(candidates):
            comp_lookup[idx] = comp

    for i, cand in enumerate(candidates):
        comp = comp_lookup.get(i, {})
        cand["cognitive_score"] = comp.get("relevance_score", 0.3)
        cand["cognitive_dimension"] = comp.get("dimension", cand.get("type", "?"))
        cand["cognitive_reason"] = comp.get("reason", "")

        dim = cand.get("type", "")
        dim_priority = query_profile.get(f"need_{dim}", True)
        if not dim_priority:
            cand["cognitive_score"] *= 0.65
            cand["_dim_penalty"] = True

        semantic = cand.get("semantic_score", 0.5)
        cand["combined_score"] = alpha_semantic * semantic + alpha_cognitive * cand["cognitive_score"]

    candidates.sort(key=lambda c: c.get("combined_score", 0), reverse=True)
    return candidates


# Backward compatibility aliases
def extract_causal_query(task_text: str,
                         excluded_dimensions: list[str] | None = None) -> dict:
    """Alias: extract cognitive profile (includes causal signature)."""
    return extract_cognitive_query(task_text, excluded_dimensions=excluded_dimensions)


def causal_rerank(query_causal: dict, candidates: list[dict]) -> list[dict]:
    """Alias: full cognitive rerank (handles all dimensions)."""
    return cognitive_rerank(query_causal, candidates)
