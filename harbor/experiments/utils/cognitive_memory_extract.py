"""Extract cognitive-dimensional memories from agent trajectories.

Optimized v2 flow (5 LLM calls per task, down from 20):
  1. Trajectory summarization (1 call) — compresses full trajectory into structured JSON
  2. Combined level extraction (4 calls) — each level extracts all 4 cognitive
     dimensions in a single call using the summary as input
  3. Traditional memories (workflow, local, summary, insight) are DERIVED from
     cognitive output without extra LLM calls

Output format is identical to v1 for downstream compatibility.
"""
import os
import json
import pickle
import threading
from filelock import FileLock, Timeout
from openai import OpenAI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.prompts.cognitive_memory import TRAJECTORY_SUMMARY_PROMPT, COMBINED_MATRIX
from sentence_transformers import SentenceTransformer

_deepseek_client: OpenAI | None = None
_client_lock = threading.Lock()


def _get_deepseek_client() -> OpenAI:
    """Lazily create and cache the DeepSeek client (thread-safe)."""
    global _deepseek_client
    if _deepseek_client is None:
        with _client_lock:
            if _deepseek_client is None:
                _deepseek_client = OpenAI(
                    api_key=os.getenv("API_KEY"),
                    base_url=os.getenv("BASE_URL", "https://api.deepseek.com"),
                )
    return _deepseek_client

embed_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
MEMORY_LOCK_TIMEOUT = 180


def _embed(text: str) -> list[float]:
    return embed_model.encode(text).tolist()


def _parse_json(raw: str) -> dict | None:
    """Robust JSON extraction from LLM output (handles markdown wrapping)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON block within markdown
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Try outermost braces
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(raw[first:last + 1])
            except json.JSONDecodeError:
                pass
        print(f"    [cognitive] JSON parse failed, raw preview: {raw[:200]}")
        return None


def _call_llm(system_prompt: str, user_content: str, max_retries: int = 2) -> dict | None:
    """Call DeepSeek with prompt + user content, return parsed JSON with retries."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    last_raw = ""
    for attempt in range(max_retries + 1):
        if attempt > 0:
            messages.append({"role": "user", "content": (
                "Your previous output was not valid JSON. "
                "Output ONLY the JSON object, no markdown wrapping, no extra text."
            )})
        try:
            response = _get_deepseek_client().chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                timeout=300.0,
            )
            raw = response.choices[0].message.content or ""
            last_raw = raw
            result = _parse_json(raw)
            if result is not None:
                return result
        except Exception as e:
            print(f"    [cognitive] LLM call failed (attempt {attempt + 1}): {e}")
    print(f"    [cognitive] Failed after {max_retries + 1} attempts. Last raw: {last_raw[:300]}")
    return None


# ---------------------------------------------------------------------------
# Phase 1: Trajectory Summarization
# ---------------------------------------------------------------------------

def _summarize_trajectory(trajectory: list[dict]) -> dict | None:
    """Compress full trajectory into a structured summary (1 LLM call).

    The summary preserves all information needed for cognitive extraction
    while being ~80-90% shorter than the raw trajectory.
    """
    traj_text = str(trajectory[1:])  # Skip system message
    result = _call_llm(TRAJECTORY_SUMMARY_PROMPT,
                       f"### Full Trajectory:\n{traj_text}")
    if result is None:
        return None
    # Validate required fields
    if "task_description" not in result:
        print("    [cognitive] Summary missing task_description, using raw")
        return None
    return result


# ---------------------------------------------------------------------------
# Phase 2: Combined Level Extraction
# ---------------------------------------------------------------------------

# All 4 cognitive dimensions in canonical order
_ALL_DIMENSIONS = ["causal", "contrastive", "strategic", "environment"]


def _build_filtered_prompt(original_prompt: str, excluded_dimensions: list[str]) -> str:
    """Remove excluded dimensions from the combined extraction prompt.

    Modifies the prompt header (e.g. "4 keys" -> "3 keys") and strips the
    ``--- dim: description ---`` sections for every excluded dimension.
    """
    active = [d for d in _ALL_DIMENSIONS if d not in excluded_dimensions]
    if len(active) == len(_ALL_DIMENSIONS):
        return original_prompt  # nothing to filter

    # 1. Fix the header line: "Output JSON with 4 keys: ..."
    import re
    dim_list = ", ".join(f'"{d}"' for d in active)
    prompt = re.sub(
        r'Output JSON with \d+ keys: "[^"]+"(?:, "[^"]+")*',
        f'Output JSON with {len(active)} keys: {dim_list}',
        original_prompt,
    )

    # 2. Remove `--- excluded_dim: description ---` sections
    for dim in excluded_dimensions:
        # Match from "--- dim:" to just before the next "--- " or end of string
        # Pattern: optional blank line + --- dim_name: ... followed by content until next ---
        prompt = re.sub(
            rf'\n*--- {re.escape(dim)}:.*?(?=\n--- |\n*\Z)',
            '',
            prompt,
            flags=re.DOTALL,
        )

    return prompt


def _extract_level(level: str, prompt: str, summary: dict,
                   excluded_dimensions: list[str] | None = None) -> dict[str, dict]:
    """Run one combined extraction for a single abstraction level.

    Returns: {dimension_name: data_dict} for applicable dimensions only.
    """
    excluded = [d.strip().lower() for d in (excluded_dimensions or [])]
    filtered_prompt = _build_filtered_prompt(prompt, excluded)
    active_dims = [d for d in _ALL_DIMENSIONS if d not in excluded]

    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    result = _call_llm(filtered_prompt,
                       f"### Trajectory Summary:\n{summary_text}")

    if result is None:
        print(f"    [cognitive] {level}: LLM returned no valid JSON")
        return {}

    # Only process dimensions that were NOT excluded
    dimensions = {}
    for dim in active_dims:
        dim_data = result.get(dim)
        if dim_data is None:
            # Some LLMs might return flat keys — try to find matching keys
            for key, val in result.items():
                if key.lower().startswith(dim.lower()[:4]):
                    dim_data = val
                    break
        if dim_data is None:
            # If the LLM returned the data flat (not nested), try treating
            # the whole result as the dimension data for the first dimension
            # that matches. This handles edge cases.
            continue

        if isinstance(dim_data, dict):
            if dim_data.get("applicable") is False:
                reason = dim_data.get("reason", "no reason given")
                print(f"    [cognitive] {level}/{dim}: not applicable — {reason}")
            else:
                dimensions[dim] = dim_data
        else:
            print(f"    [cognitive] {level}/{dim}: unexpected type {type(dim_data)}")

    return dimensions


# ---------------------------------------------------------------------------
# Phase 3: Cognitive Abstract & Embedding (unchanged from v1)
# ---------------------------------------------------------------------------

def _build_cognitive_abstract(level: str, dimension: str, data: dict) -> str:
    """Extract the cognitive abstract from each cell's output (for retrieval embedding).

    Returns a compact summary of the cognitive insight in domain-independent language.
    """
    if level == "trajectory":
        if dimension == "causal":
            return data.get("summary", "")
        if dimension == "contrastive":
            return data.get("transition_insight", "")
        if dimension == "strategic":
            return data.get("efficiency_assessment", "")
        if dimension == "environment":
            return data.get("repo_initialization", "")

    if level == "workflow":
        if dimension == "causal":
            return data.get("causal_graph_summary", "")
        if dimension == "contrastive":
            return " ".join(data.get("workflow_decision_rules", []))
        if dimension == "strategic":
            return (data.get("workflow_rationale", "") + " " +
                    " ".join(data.get("ordering_constraints", [])))
        if dimension == "environment":
            return " ".join(data.get("repo_adaptation_rules", []))

    if level == "summary":
        if dimension == "causal":
            return data.get("causal_principle", "")
        if dimension == "contrastive":
            return data.get("success_failure_boundary", "")
        if dimension == "strategic":
            return (data.get("meta_strategy", "") + " " +
                    " ".join(data.get("cognitive_moves", [])))
        if dimension == "environment":
            profile = data.get("repo_profile", {})
            return str(profile) + " " + " ".join(data.get("footguns", []))

    if level == "insight":
        if dimension == "causal":
            return data.get("principle_name", "") + " " + data.get("principle_statement", "")
        if dimension == "contrastive":
            ap_names = [ap.get("name", "") for ap in data.get("anti_patterns", [])]
            return data.get("positive_pattern", "") + " " + " ".join(ap_names)
        if dimension == "strategic":
            return data.get("methodology_name", "") + " " + data.get("core_idea", "")
        if dimension == "environment":
            return (" ".join(data.get("tool_agnostic_advice", [])) + " " +
                    data.get("language_transfer", ""))

    return ""


def _build_embedding_text(level: str, dimension: str, data: dict,
                          task_text: str = "") -> str:
    """Build HYBRID embedding text: original task + cognitive abstract + anchors.

    Puts BOTH concrete task description AND cognitive abstract into the same
    embedding vector so queries can match on task-level or cognitive-level similarity.
    """
    cognitive_abstract = _build_cognitive_abstract(level, dimension, data)

    anchors = data.get("concrete_anchors", {}) or data.get("concrete_origin", {}) or {}
    anchor_parts = []
    for key in ["files_involved", "files_modified", "key_files_paths"]:
        vals = anchors.get(key, [])
        if vals:
            anchor_parts.append("Files: " + ", ".join(str(v) for v in vals[:5]))
            break
    for key in ["key_functions"]:
        vals = anchors.get(key, [])
        if vals:
            anchor_parts.append("Functions: " + ", ".join(str(v) for v in vals[:5]))
    for key in ["error_pattern", "error_signature", "fix_pattern"]:
        val = anchors.get(key, "")
        if val:
            anchor_parts.append(str(key) + ": " + str(val)[:200])
    for key in ["test_command", "build_command"]:
        val = anchors.get(key, "")
        if val:
            anchor_parts.append(str(key) + ": " + str(val)[:200])
    for key in ["source_file", "source_example", "anti_pattern_file",
                "positive_pattern_code", "methodology_in_action"]:
        val = anchors.get(key, "")
        if val:
            anchor_parts.append(str(key) + ": " + str(val)[:200])
    anchor_text = " ".join(anchor_parts)

    parts = []
    if task_text:
        parts.append(task_text[:800])
    if cognitive_abstract:
        parts.append("[COG] " + cognitive_abstract)
    if anchor_text:
        parts.append("[ANC] " + anchor_text)

    return " ".join(parts) if parts else cognitive_abstract


# ---------------------------------------------------------------------------
# Phase 4: Save cognitive memory entries (unchanged format from v1)
# ---------------------------------------------------------------------------

def _save_pkl(memory_path: str, new_entry: dict):
    lock_path = memory_path + ".lock"
    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(new_entry)
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
    except Timeout:
        print(f"    [cognitive] Lock timeout for {memory_path}")


# ---------------------------------------------------------------------------
# Phase 5: Derive traditional memories from cognitive output (0 LLM calls)
# ---------------------------------------------------------------------------

def _derive_workflow_memory(cognitive_data: dict, summary: dict,
                            judgement: bool, task: str) -> dict:
    """Derive workflow memory from trajectory-level + summary cognitive data."""
    traj_data = cognitive_data.get("trajectory", {})
    wf_data = cognitive_data.get("workflow", {})

    # Build goal from strategic analysis
    traj_strategic = traj_data.get("strategic", {})
    phases = traj_strategic.get("phases", [])
    if phases:
        goal = "Solve task by: " + "; ".join(
            f"{p.get('phase', '')}: {p.get('purpose', '')}" for p in phases[:3]
        )
    else:
        wf_strategic = wf_data.get("strategic", {})
        goal = wf_strategic.get("workflow_rationale", "Code editing workflow")

    # Build workflow commands from trajectory summary
    commands = summary.get("command_sequence", [])
    workflow = []
    for cmd in commands[:15]:
        cmd_text = cmd.get("command", "")
        if cmd_text:
            workflow.append(cmd_text)

    if not workflow:
        # Fallback: from trajectory causal chains
        traj_causal = traj_data.get("causal", {})
        for chain in traj_causal.get("causal_chains", []):
            cmd = chain.get("trigger_command", "")
            if cmd:
                workflow.append(cmd)

    return {"goal": goal, "workflow": workflow}


def _derive_local_memory(cognitive_data: dict, summary: dict,
                         judgement: bool, task: str) -> dict:
    """Derive local (traj) memory from summary-level cognitive data."""
    sum_data = cognitive_data.get("summary", {})
    sum_causal = sum_data.get("causal", {})
    sum_contrastive = sum_data.get("contrastive", {})

    # when_to_use: from bug class + causal principle
    bug_class = sum_causal.get("bug_class", "")
    causal_principle = sum_causal.get("causal_principle", "")
    when_to_use = f"When encountering {bug_class} bugs. {causal_principle}".strip()

    # experience: from root cause + resolution
    root_cause = sum_causal.get("root_cause", {})
    resolution = sum_causal.get("resolution", {})
    if isinstance(root_cause, dict):
        experience_parts = [
            root_cause.get("what_was_wrong", ""),
            root_cause.get("why_it_was_wrong", ""),
        ]
        if isinstance(resolution, dict):
            experience_parts.append(resolution.get("approach", ""))
            experience_parts.append(resolution.get("why_it_works", ""))
        experience = " ".join(p for p in experience_parts if p)
    else:
        experience = str(root_cause)

    # generalized_query: abstract the task
    generalized_query = f"Fix a {bug_class} issue in a repository where {causal_principle}"

    # tags: from repo context + bug class
    repo_context = summary.get("repo_context", "")
    files_modified = summary.get("files_modified", [])
    tags = [bug_class] if bug_class else []
    for f in files_modified[:3]:
        # Extract module name from path
        parts = f.replace("\\", "/").split("/")
        for p in parts:
            if p.endswith(".py") and p != "__init__.py":
                tags.append(p.replace(".py", ""))

    return {
        "when_to_use": when_to_use.strip() or task[:200],
        "task_query": task,
        "generalized_query": generalized_query.strip(),
        "experience": experience.strip() or f"Task {'passed' if judgement else 'failed'}.",
        "tags": tags[:5] if tags else ["code", "fix"],
    }


def _derive_summary_memory(cognitive_data: dict, summary: dict,
                           judgement: bool, task: str) -> dict:
    """Derive summary memory from trajectory summary + cognitive summary level."""
    sum_data = cognitive_data.get("summary", {})
    sum_causal = sum_data.get("causal", {})
    sum_contrastive = sum_data.get("contrastive", {})

    task_summary = summary.get("task_description", task)

    # Build experience summary from cognitive data
    parts = []
    root_cause = sum_causal.get("root_cause", {})
    if isinstance(root_cause, dict):
        parts.append(f"Issue: {root_cause.get('what_was_wrong', '')}")
    resolution = sum_causal.get("resolution", {})
    if isinstance(resolution, dict):
        parts.append(f"Fix: {resolution.get('approach', '')}")
        parts.append(f"Verification: {resolution.get('verification', '')}")

    if judgement:
        if_pass = sum_contrastive.get("if_pass", {})
        if isinstance(if_pass, dict):
            factors = [str(f) for f in if_pass.get("key_success_factors", [])]
            if factors:
                parts.append("Key success factors: " + "; ".join(factors))
    else:
        if_fail = sum_contrastive.get("if_fail", {})
        if isinstance(if_fail, dict):
            parts.append(f"Failure: {if_fail.get('root_cause_of_failure', '')}")

    boundary = sum_contrastive.get("success_failure_boundary", "")
    if boundary:
        parts.append(f"Critical boundary: {boundary}")

    experience_summary = " | ".join(p for p in parts if p)

    return {
        "task_summary": task_summary,
        "experience_summary": experience_summary or task_summary,
    }


def _derive_insight_memory(cognitive_data: dict, summary: dict,
                           judgement: bool, task: str) -> dict:
    """Derive insight memory from insight-level cognitive data."""
    ins_data = cognitive_data.get("insight", {})
    ins_causal = ins_data.get("causal", {})
    ins_strategic = ins_data.get("strategic", {})
    ins_contrastive = ins_data.get("contrastive", {})

    # Prefer causal principle name, fall back to methodology name
    title = ins_causal.get("principle_name", "")
    if not title:
        title = ins_strategic.get("methodology_name", "")
    if not title:
        title = ins_contrastive.get("positive_pattern", "")
    if not title:
        title = f"Lessons from {'successful' if judgement else 'failed'} task"

    # Description: first sentence of the principle/methodology
    description = ins_causal.get("principle_statement", "")
    if not description:
        description = ins_strategic.get("core_idea", "")
    if description:
        description = description.split(".")[0].strip() + "."
    else:
        description = title

    # Content: combine key insights
    content_parts = []
    if ins_causal.get("principle_statement"):
        content_parts.append(ins_causal["principle_statement"])
    anti_patterns = ins_contrastive.get("anti_patterns", [])
    for ap in anti_patterns[:2]:
        if isinstance(ap, dict):
            content_parts.append(f"Avoid: {ap.get('name', '')} — {ap.get('escape_strategy', '')}")
        else:
            content_parts.append(str(ap))
    decision_rules = ins_strategic.get("decision_rules", [])
    for dr in decision_rules[:2]:
        if isinstance(dr, dict):
            # LLM may return decision rules as structured dicts
            # e.g. {"id":"DR1","condition":"...","action":"..."}
            parts = []
            if dr.get("condition"):
                parts.append(f"If {dr['condition']}")
            if dr.get("action"):
                parts.append(f"then {dr['action']}")
            if parts:
                content_parts.append("; ".join(parts))
            else:
                content_parts.append(str(dr))
        else:
            content_parts.append(str(dr))
    content = " ".join(content_parts) if content_parts else title

    return {"title": title, "description": description, "content": content}


def _save_derived_memory(memory_path: str, entry: dict):
    """Save a derived traditional memory entry to a pkl file."""
    lock_path = memory_path + ".lock"
    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(entry)
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
    except Timeout:
        print(f"    [derive] Lock timeout for {memory_path}")


def _derive_traditional_memories(cognitive_data: dict, summary: dict,
                                 judgement: bool, log_dir: str,
                                 task_name: str, task: str,
                                 commands: list, benchmark: str):
    """Derive all 4 traditional memory types from cognitive output (0 LLM calls).

    The cognitive 4x4 matrix already contains richer versions of what the
    traditional extraction produces. We extract and reformat the relevant
    portions without additional LLM calls.
    """
    parent_dir = log_dir.rsplit("/", 1)[0]

    # 1. Workflow memory
    try:
        wf_mem = _derive_workflow_memory(cognitive_data, summary, judgement, task)
        wf_entry = {
            "benchmark": benchmark,
            "task_name": task_name,
            "llm_judge": judgement,
            "task": task,
            "type": "workflow",
            "workflow": wf_mem,
            "key_embedding": _embed(wf_mem["goal"]),
        }
        _save_derived_memory(f"{parent_dir}/workflow_memory.pkl", wf_entry)
        print(f"    [derive] workflow_memory saved")
    except Exception as e:
        print(f"    [derive] workflow_memory FAILED: {e}")

    # 2. Local (traj) memory
    try:
        local_mem = _derive_local_memory(cognitive_data, summary, judgement, task)
        local_entry = {
            "when_to_use": local_mem["when_to_use"],
            "task_query": local_mem["task_query"],
            "generalized_query": local_mem["generalized_query"],
            "experience": local_mem["experience"],
            "tags": local_mem["tags"],
            "generalized_query_embedding": _embed(local_mem["generalized_query"]),
            "benchmark": benchmark,
            "task_name": task_name,
            "commands": commands,
        }
        local_path = f"{parent_dir}/local_memory.pkl"
        _save_derived_memory(local_path, local_entry)
        # Also save JSON copy (matching legacy format)
        try:
            # log_dir is already guaranteed to exist (created in Phase 0)
            all_local = []
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    all_local = pickle.load(f)
            json_entries = [{
                "when_to_use": x["when_to_use"],
                "task_query": x["task_query"],
                "generalized_query": x["generalized_query"],
                "experience": x["experience"],
                "tags": x["tags"],
                "benchmark": x.get("benchmark", ""),
                "task_name": x.get("task_name", ""),
            } for x in all_local[:-1]]
            with open(f"{log_dir}/local_memory.json", "w", encoding="utf-8") as f:
                json.dump({"memory": {k: v for k, v in local_entry.items()
                                      if k != "generalized_query_embedding"},
                           "all_memory": json_entries}, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"    [derive] local_memory.json save failed: {e}")
        print(f"    [derive] local_memory saved")
    except Exception as e:
        print(f"    [derive] local_memory FAILED: {e}")

    # 3. Summary memory
    try:
        sum_mem = _derive_summary_memory(cognitive_data, summary, judgement, task)
        sum_entry = {
            "task_summary": sum_mem["task_summary"],
            "experience_summary": sum_mem["experience_summary"],
            "embedding": _embed(sum_mem["task_summary"]),
            "benchmark": benchmark,
            "task_name": task_name,
            "commands": commands,
            "judgement": judgement,
            "task": task,
            "type": "summary",
        }
        _save_derived_memory(f"{parent_dir}/summary_memory_{benchmark}.pkl", sum_entry)
        print(f"    [derive] summary_memory saved")
    except Exception as e:
        print(f"    [derive] summary_memory FAILED: {e}")

    # 4. Insight memory
    try:
        ins_mem = _derive_insight_memory(cognitive_data, summary, judgement, task)
        ins_entry = {
            "key_embedding": _embed(ins_mem["title"]),
            "benchmark": benchmark,
            "type": "insight",
            "llm_judge": judgement,
            "task_name": task_name,
            "task": task,
            "insight": ins_mem,
        }
        _save_derived_memory(f"{parent_dir}/insight_memory.pkl", ins_entry)
        print(f"    [derive] insight_memory saved")
    except Exception as e:
        print(f"    [derive] insight_memory FAILED: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_cognitive_memories(judgement, trajectory, log_dir, task_name,
                               task, commands, benchmark,
                               derive_traditional: bool = True,
                               excluded_dimensions: list | None = None,
                               excluded_levels: list | None = None) -> bool:
    """Extract all cognitive + traditional memories (5 LLM calls total).

    Flow:
      0. Create task subdirectory (always — serves as a "task was processed" marker)
      1. Summarize trajectory (1 LLM call)
      2. Extract levels x dimensions via combined prompts (4 LLM calls normally,
         fewer when levels are excluded)
      3. Save cognitive pkl files (skipping excluded dimensions/levels)
      4. Derive and save traditional memories (0 LLM calls) -- only if
         derive_traditional=True (default).

    Args:
        derive_traditional: If False, skip Phase 4 (traditional memory derivation).
            Cognitive memories are always extracted (subject to excluded_* filters).
        excluded_dimensions: Cognitive dimensions to skip entirely (e.g. ["causal"]).
            Neither LLM extraction nor pkl saving will happen for these dimensions.
        excluded_levels: Abstraction levels to skip entirely (e.g. ["trajectory"]).
            The LLM call for these levels is skipped, and no memories are saved.

    Returns:
        True if at least Phase 2 (cognitive extraction) succeeded.  False means
        the trajectory summary failed and no cognitive memories were saved.
        (Phase 4 traditional derivation failures do NOT affect the return value —
        they are logged but considered non-critical.)
    """
    excluded_dim = [d.strip().lower() for d in (excluded_dimensions or [])]
    excluded_lvl = [l.strip().lower() for l in (excluded_levels or [])]

    if excluded_dim:
        print(f"    [cognitive] Excluded dimensions: {excluded_dim}")
    if excluded_lvl:
        print(f"    [cognitive] Excluded levels: {excluded_lvl}")

    parent_dir = log_dir.rsplit("/", 1)[0]

    # ---- Phase 0: Ensure task subdirectory exists ----
    # Create early so the directory serves as a "task was processed" marker
    # even if later phases fail.  Previously this was buried inside
    # _derive_local_memory (Phase 4), so any Phase 2/3/4 failure silently
    # left no trace that the task was ever processed.
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"    [cognitive] WARNING: could not create task dir {log_dir}: {e}")

    # ---- Phase 1: Trajectory Summarization ----
    print(f"    [cognitive] Phase 1/4: Summarizing trajectory...")
    summary = _summarize_trajectory(trajectory)
    if summary is None:
        # The LLM call failed — we cannot extract cognitive memories without a
        # summary.  Still attempt traditional derivation with empty data
        # (it will produce degraded but non-empty output from the raw task).
        print(f"    [cognitive] ERROR: trajectory summary failed — "
              f"skipping Phase 2 cognitive extraction")
        if derive_traditional:
            print(f"    [cognitive] Phase 4/4: Deriving traditional memories "
                  f"(from empty cognitive data — degraded)...")
            try:
                _derive_traditional_memories({}, summary or {}, judgement,
                                             log_dir, task_name, task,
                                             commands, benchmark)
            except Exception as e:
                print(f"    [cognitive] WARNING: traditional memory derivation "
                      f"failed for {task_name}: {e}")
                import traceback
                traceback.print_exc()
        return False  # signal: cognitive extraction failed

    # ---- Phase 2: Combined Level Extraction ----
    active_levels = [l for l in COMBINED_MATRIX if l not in excluded_lvl]
    n_levels = len(active_levels)
    print(f"    [cognitive] Phase 2/4: Extracting {n_levels} levels (combined)...")
    all_cognitive_data = {}  # {level: {dimension: data}}
    count = 0

    for level, prompt in COMBINED_MATRIX.items():
        if level in excluded_lvl:
            print(f"    [cognitive]   {level}... SKIPPED (excluded level)")
            continue

        print(f"    [cognitive]   {level}...")
        dimensions = _extract_level(level, prompt, summary,
                                    excluded_dimensions=excluded_dim)

        if dimensions:
            all_cognitive_data[level] = dimensions

        for dimension, data in dimensions.items():
            # Double-check: never save excluded dimensions (defense in depth)
            if dimension in excluded_dim:
                print(f"    [cognitive]   SKIP saving {level}/{dimension} (excluded)")
                continue

            embedding_text = _build_embedding_text(level, dimension, data,
                                                       task_text=task)
            cognitive_text = _build_cognitive_abstract(level, dimension, data)

            entry = {
                "benchmark": benchmark,
                "task_name": task_name,
                "task": task,
                "llm_judge": judgement,
                "type": dimension,
                "level": level,
                "memory": data,
                "key_embedding": _embed(embedding_text),
                "cognitive_embedding": _embed(cognitive_text) if cognitive_text else None,
                "concrete_anchors": data.get("concrete_anchors", {}) or data.get("concrete_origin", {}),
            }

            pkl_path = f"{parent_dir}/{dimension}_memory.pkl"
            _save_pkl(pkl_path, entry)
            count += 1

    total_dims = sum(len(d) for d in all_cognitive_data.values())
    print(f"    [cognitive] Extracted {count} cognitive memories "
          f"({total_dims} dimensions across {len(all_cognitive_data)} levels)")

    # ---- Phase 3: Derive Traditional Memories (optional) ----
    if derive_traditional:
        print(f"    [cognitive] Phase 3/4: Deriving traditional memories...")
        try:
            _derive_traditional_memories(all_cognitive_data, summary, judgement,
                                         log_dir, task_name, task, commands, benchmark)
        except Exception as e:
            print(f"    [cognitive] WARNING: traditional memory derivation failed for "
                  f"{task_name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"    [cognitive] Phase 3/4: Skipped (derive_traditional=False)")

    # Report which traditional memory types were saved (or skipped due to errors)
    n_traditional = 0
    traditional_types = []
    for ttype, fname in [("workflow", "workflow_memory.pkl"),
                          ("local", "local_memory.pkl"),
                          ("summary", f"summary_memory_{benchmark}.pkl"),
                          ("insight", "insight_memory.pkl")]:
        if os.path.exists(f"{parent_dir}/{fname}"):
            traditional_types.append(ttype)
            n_traditional += 1
    if derive_traditional and n_traditional < 4:
        missing = [t for t in ["workflow", "local", "summary", "insight"]
                   if t not in traditional_types]
        print(f"    [cognitive] WARNING: {len(missing)} traditional memory type(s) "
              f"not saved: {missing}")
    elif derive_traditional:
        print(f"    [cognitive] All 4 traditional memory types saved successfully")

    return True


# Backward compatibility alias
_build_cognitive_embedding_text = _build_cognitive_abstract
