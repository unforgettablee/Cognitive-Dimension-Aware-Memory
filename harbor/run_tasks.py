"""
Run harbor tasks sequentially, one per ``harbor jobs start`` call.

All pipelines are **sequential** — every task gets its own harbor invocation.
When ``use_memory: true`` in the pipeline config, the script also:

  1. Retrieves relevant memories from the pool and injects them into the task
     instruction before each round.
  2. Extracts cognitive + traditional memories from each completed task into
     the pool, so the next task can retrieve from it.

  Round 1: task[0], no memory          → run harbor → extract → pool += task_0
  Round 2: task[1], pool = {task_0}    → inject → run harbor → extract → pool += task_1
  Round 3: task[2], pool = {task_0,1}  → inject → run harbor → extract → pool += task_2
  ...

When ``use_memory: false``, tasks still run one at a time but without any
memory injection or extraction — a clean sequential baseline.

All trial directories are consolidated into a single run-timestamp folder::

    jobs/<name>/2026-06-21__12-30-00/
      astropy__astropy-12907__xxx/
      astropy__astropy-13033__xxx/
      ...

Environment variables:
    API_KEY     API key (also checked as DEEPSEEK_API_KEY)
    BASE_URL    API base URL (default: https://api.deepseek.com)
    MODEL       Model name (e.g. "deepseek-chat" or "deepseek/deepseek-chat")

Priority: CLI flag > env var > pipeline config > hard-coded default.

Usage (bash, from project root):
    export API_KEY="sk-xxx"
    export BASE_URL="https://api.deepseek.com"
    export MODEL="deepseek-chat"

    # Full system
    python harbor/run_tasks.py --pipeline e1_full --start 0 --end 500 --ak step_limit=150 --force-build

    # No-memory baseline
    python harbor/run_tasks.py --pipeline e24_no_memory --start 0 --end 500 --ak step_limit=150 --force-build

    # Turn off memory from CLI
    python harbor/run_tasks.py --pipeline e1_full --start 0 --end 500 --no-memory --force-build

    # Dry-run
    python harbor/run_tasks.py --pipeline e1_full --start 0 --end 2 --dry-run
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Pipeline config helpers
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "configs" / "experiments"


def _find_pipeline_config(name: str) -> Path | None:
    """Resolve a pipeline name or path to a YAML file."""
    path = Path(name)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return path
    for candidate in [
        EXPERIMENTS_DIR / f"{name}.yaml",
        EXPERIMENTS_DIR / f"{name}.yml",
        EXPERIMENTS_DIR / name,
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_pipeline_yaml(path: Path) -> dict:
    """Load a pipeline YAML config file."""
    if yaml is None:
        print("ERROR: PyYAML is required for --pipeline.  Install with: pip install pyyaml")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_model_name(raw: str) -> str:
    """Ensure model name has provider/ prefix; default to deepseek/."""
    if not raw:
        return ""
    if "/" in raw:
        return raw
    return f"deepseek/{raw}"


def _find_harbor_exe() -> str | None:
    """Locate the harbor CLI executable.

    Returns an absolute path to the venv's python3 interpreter.  We always
    run harbor as ``<python3> -m harbor.cli.main`` because the ``harbor``
    wrapper script has a shebang that can break under ``subprocess``.
    """
    if sys.platform == "win32":
        candidates = [
            ".venv-harbor/Scripts/python.exe",
            ".venv/Scripts/python.exe",
        ]
    else:
        candidates = [
            ".venv-harbor/bin/python3",
            ".venv/bin/python3",
        ]

    for candidate in candidates:
        p = Path(candidate).absolute()
        if p.exists():
            return str(p)

    return None


# ---------------------------------------------------------------------------
# Extraction helpers (used by memory-pipeline sequential mode)
# ---------------------------------------------------------------------------

def _find_python_exe() -> str | None:
    """Find a Python executable that has the extraction dependencies.

    Tries the current venv first (``.venv/bin/python3``), then falls back
    to ``sys.executable``.
    """
    candidates = []
    if sys.platform == "win32":
        candidates = [
            ".venv-harbor/Scripts/python.exe",
            ".venv/Scripts/python.exe",
        ]
    else:
        candidates = [
            ".venv-harbor/bin/python3",
            ".venv/bin/python3",
        ]

    for candidate in candidates:
        p = Path(candidate).absolute()
        if p.exists():
            return str(p)

    return sys.executable


def _get_judgement(trial_dir: Path) -> bool:
    """Read reward from harbor result.json (1.0 = pass, 0.0 = fail)."""
    result_path = trial_dir / "result.json"
    if result_path.exists():
        raw = result_path.read_text(encoding="utf-8").strip()
        if not raw or raw == "null":
            return False  # trial crashed before result could be written
        result = json.loads(raw)
        if not isinstance(result, dict):
            return False
        verifier = result.get("verifier_result") or {}
        reward = (verifier or {}).get("rewards", {}).get("reward", 0.0)
        return reward >= 1.0
    # Fallback: verifier/reward.txt
    reward_path = trial_dir / "verifier" / "reward.txt"
    if reward_path.exists():
        try:
            return float(reward_path.read_text().strip()) >= 1.0
        except ValueError:
            pass
    return False


def _find_trial_dir(job_dir: Path, task_name: str) -> Path | None:
    """Find the trial directory for a task within a job directory.

    Harbor creates trial dirs named ``<task_name>__<random_suffix>`` inside the
    job directory.  This finds the first match.
    """
    if not job_dir.exists():
        return None
    for d in job_dir.iterdir():
        if d.is_dir() and d.name.startswith(task_name + "__"):
            return d
    return None


def _extract_memories_for_task(
    task_name: str,
    jobs_dir: Path,
    pool_dir: Path,
    only_passed: bool = True,
    derive_traditional: bool = True,
    python_exe: str | None = None,
    start_index: int = 0,
    excluded_dimensions: list | None = None,
    excluded_levels: list | None = None,
    pipeline_config: dict | None = None,
) -> bool:
    """Extract memories from a completed task's trial directory into the pool.

    Calls ``harbor/extract_memories.py`` as a subprocess.  ``jobs_dir`` should
    be the parent directory containing trial dirs; ``start_index`` selects
    which task to extract (0-based, in alphabetical order).

    Returns True if extraction succeeded, False otherwise.
    """
    if python_exe is None:
        python_exe = _find_python_exe()

    # Verify the trial directory exists
    trial_dir = _find_trial_dir(jobs_dir, task_name)
    if trial_dir is None:
        print(f"  WARNING: No trial directory found for {task_name} in {jobs_dir}")
        return False

    # Always extract memories — even FAIL tasks contain partial trajectory
    # data that can be mined for anti-patterns, failure modes, and
    # environment knowledge.  The ``only_passed`` config still controls
    # whether FAIL-task memories are injected into *subsequent* tasks'
    # prompts (via the retrieval pool), but all tasks contribute to the
    # indexed pool so downstream experiments can filter.
    judgement = _get_judgement(trial_dir)
    status = "PASS" if judgement else "FAIL"

    if only_passed and not judgement:
        print(f"  Task {status}: only_passed=True would normally skip extraction, "
              f"but extracting anyway (partial trajectory still has value).")
        # NOTE: we still extract — the flag is informational only.

    extract_script = str(Path(__file__).resolve().parent / "extract_memories.py")
    cmd = [
        python_exe, extract_script,
        "--jobs-dir", str(jobs_dir),
        "--memory-dir", str(pool_dir),
        "--limit", "1",
        "--start", str(start_index),
    ]
    if not derive_traditional:
        cmd.append("--no-derive-traditional")
    if excluded_dimensions:
        cmd.extend(["--excluded-dimensions"] + list(excluded_dimensions))
    if excluded_levels:
        cmd.extend(["--excluded-levels"] + list(excluded_levels))

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    # Explicitly pass API config to subprocess (extraction scripts use both names)
    api_key = os.environ.get("API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    if api_key:
        env["API_KEY"] = api_key
        env["DEEPSEEK_API_KEY"] = api_key
    base_url = os.environ.get("BASE_URL", "")
    # Fallback: use pipeline config's llm section for base_url
    if not base_url and pipeline_config:
        llm_cfg = pipeline_config.get("llm", {}) or {}
        base_url = llm_cfg.get("base_url", "")
    if base_url:
        env["BASE_URL"] = base_url

    print(f"  Extracting memories via: {' '.join(cmd[:4])} ... --memory-dir {pool_dir} --start {start_index}")

    try:
        result = subprocess.run(
            cmd, check=False, timeout=600,
            capture_output=True, text=True,
            env=env,
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"  [extract] {line}")
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                print(f"  [extract:err] {line}")
        if result.returncode != 0:
            print(f"  WARNING: extract_memories.py returned exit code {result.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  WARNING: extract_memories.py timed out for {task_name}")
        return False
    except Exception as e:
        print(f"  ERROR running extract_memories.py: {e}")
        return False


def _count_pool_tasks(pool_dir: Path) -> int:
    """Count how many unique tasks have contributed to the memory pool."""
    if not pool_dir.exists():
        return 0
    task_dirs = [d for d in pool_dir.iterdir()
                 if d.is_dir() and not d.name.startswith(".")]
    return len(task_dirs)


def _format_value_as_text(val, label: str = "", indent: str = "") -> list[str]:
    """Format any value (str, list, dict, etc.) into one or more compact text lines.

    Returns a list of formatted lines.
    """
    lines = []
    if val is None or val == "" or val == [] or val == {}:
        return lines

    if isinstance(val, str):
        if val and val != "other":
            prefix = f"{label}: " if label else ""
            lines.append(f"{prefix}{val}")

    elif isinstance(val, list):
        if len(val) == 0:
            return lines
        prefix = f"{label}:" if label else ""
        if prefix:
            lines.append(prefix)
        for item in val:
            if isinstance(item, dict):
                parts = []
                for k, v in item.items():
                    if v and v != "other" and k not in ("applicable",):
                        parts.append(f"{k}: {str(v)[:150]}")
                if parts:
                    lines.append(f"{indent}- {', '.join(parts)}")
            elif isinstance(item, str) and item and item != "other":
                lines.append(f"{indent}- {item}")
            elif item is not None:
                lines.append(f"{indent}- {str(item)[:250]}")

    elif isinstance(val, dict):
        prefix = f"{label}:" if label else ""
        if prefix:
            lines.append(prefix)
        for k, v in val.items():
            if v and v != "other" and k not in ("applicable",):
                if isinstance(v, list):
                    sub_items = [str(s)[:120] for s in v[:5] if s]
                    if sub_items:
                        lines.append(f"{indent}- {k}: {', '.join(sub_items)}")
                elif isinstance(v, dict):
                    parts = [f"{sk}: {str(sv)[:120]}" for sk, sv in v.items() if sv]
                    if parts:
                        lines.append(f"{indent}- {k}: {'; '.join(parts)}")
                elif isinstance(v, str) and v:
                    lines.append(f"{indent}- {k}: {v[:200]}")
                else:
                    lines.append(f"{indent}- {k}: {str(v)[:150]}")

    else:
        s = str(val)
        if s and s != "other":
            prefix = f"{label}: " if label else ""
            lines.append(f"{prefix}{s[:300]}")

    return lines


def _format_concrete_anchors_display(anchors: dict) -> str:
    """Format concrete_anchors/concrete_origin into compact single-line summary.

    Reuses extraction logic from _build_concrete_anchor_brief in cognitive_rerank.py.
    """
    if not anchors:
        return ""
    parts = []

    # File paths
    for key in ["files_involved", "files_modified", "key_files_paths",
                "critical_files", "key_discovery_files", "key_decision_files",
                "key_directories"]:
        vals = anchors.get(key, [])
        if vals and isinstance(vals, list):
            parts.append(f"files={', '.join(str(v) for v in vals[:4])}")
            break

    # Key functions
    for key in ["key_functions"]:
        vals = anchors.get(key, [])
        if vals and isinstance(vals, list):
            parts.append(f"funcs={', '.join(str(v) for v in vals[:4])}")

    # Error/fix signatures
    for key in ["error_signature", "error_pattern", "fix_pattern"]:
        val = anchors.get(key, "")
        if val and isinstance(val, str) and val.strip():
            short_key = {"error_signature": "err_sig", "error_pattern": "err_pat",
                         "fix_pattern": "fix"}.get(key, key)
            parts.append(f"{short_key}={val[:200]}")

    # Test/build commands
    for key in ["test_command", "test_commands", "build_command"]:
        val = anchors.get(key, "")
        if val:
            if isinstance(val, list):
                v = "; ".join(str(x)[:120] for x in val[:2] if x)
            elif isinstance(val, str):
                v = val[:150]
            else:
                continue
            if v.strip():
                short_key = {"test_command": "test", "test_commands": "tests",
                             "build_command": "build"}.get(key, key)
                parts.append(f"{short_key}={v}")

    # Error recovery / diagnostic commands
    for key in ["error_recovery_commands", "diagnostic_commands_used",
                "actual_failed_command", "actual_successful_command"]:
        val = anchors.get(key, "")
        if val:
            if isinstance(val, list):
                v = "; ".join(str(x)[:100] for x in val[:2] if x)
            elif isinstance(val, str):
                v = val[:150]
            else:
                continue
            if v.strip():
                short_key = {"error_recovery_commands": "recovery", "diagnostic_commands_used": "diag",
                             "actual_failed_command": "fail_cmd", "actual_successful_command": "ok_cmd"}.get(key, key)
                parts.append(f"{short_key}={v}")

    # Evidence
    for key in ["success_evidence", "failure_evidence", "information_sources"]:
        vals = anchors.get(key, [])
        if vals and isinstance(vals, list):
            items = [str(v)[:100] for v in vals[:2] if v]
            if items:
                short_key = {"success_evidence": "success", "failure_evidence": "failure",
                             "information_sources": "sources"}.get(key, key)
                parts.append(f"{short_key}={'; '.join(items)}")

    # Key files (structured list)
    key_files = anchors.get("key_files", [])
    if key_files and isinstance(key_files, list):
        file_strs = []
        for kf in key_files[:3]:
            if isinstance(kf, dict):
                p = kf.get("path", "") or kf.get("why_important", "")
                file_strs.append(str(p)[:100])
            elif isinstance(kf, str):
                file_strs.append(kf[:100])
        if file_strs:
            parts.append(f"key_files={', '.join(file_strs)}")

    # Debug keys for insight-level concrete_origin
    for key in ["source_environment", "tool_specific_example", "tool_agnostic_example",
                "scale_example", "anti_example", "pivot_example",
                "source_file", "source_function", "detection_command",
                "decision_point_file", "key_test_output", "decision_trigger"]:
        val = anchors.get(key, "")
        if val and isinstance(val, str) and val.strip():
            short_key = key.replace("_", "")[:12]
            parts.append(f"{short_key}={val[:200]}")

    # Remaining string keys
    handled = frozenset({"files_involved", "files_modified", "key_files_paths",
                         "critical_files", "key_discovery_files", "key_decision_files",
                         "key_directories", "key_functions", "error_signature", "error_pattern",
                         "fix_pattern", "test_command", "test_commands", "build_command",
                         "error_recovery_commands", "diagnostic_commands_used",
                         "actual_failed_command", "actual_successful_command",
                         "success_evidence", "failure_evidence", "information_sources",
                         "key_files", "source_environment", "tool_specific_example",
                         "tool_agnostic_example", "scale_example", "anti_example",
                         "pivot_example", "source_file", "source_function",
                         "detection_command", "decision_point_file", "key_test_output",
                         "decision_trigger"})
    for k, v in anchors.items():
        if k in handled:
            continue
        if isinstance(v, str) and v.strip():
            parts.append(f"{k[:15]}={v[:150]}")
        elif isinstance(v, list):
            items = [str(x)[:100] for x in v[:3] if x]
            if items:
                parts.append(f"{k[:15]}={'; '.join(items)}")

    return " | ".join(parts) if parts else ""


# Ordered list of description-priority field names per (level, type).
# The first non-empty match becomes the description; all fields go into content.
_DESC_PRIORITY: dict[str, list[str]] = {
    # Trajectory level
    "trajectory/causal":      ["summary"],
    "trajectory/contrastive": ["transition_insight", "transition_point"],
    "trajectory/strategic":   ["efficiency_assessment"],
    "trajectory/environment": ["repo_initialization"],
    # Workflow level
    "workflow/causal":        ["causal_graph_summary"],
    "workflow/contrastive":   ["chosen_workflow"],
    "workflow/strategic":     ["workflow_rationale"],
    "workflow/environment":   ["tool_chain"],
    # Summary level
    "summary/causal":         ["causal_principle", "bug_class"],
    "summary/contrastive":    ["success_failure_boundary", "outcome"],
    "summary/strategic":      ["meta_strategy"],
    "summary/environment":    ["repo_profile"],
    # Insight level
    "insight/causal":         ["principle_statement", "principle_name"],
    "insight/contrastive":    ["positive_pattern", "generality"],
    "insight/strategic":      ["core_idea", "methodology_name"],
    "insight/environment":    ["language_transfer", "scale_considerations"],
}

# Field names to skip (handled separately or not useful in display)
_SKIP_FIELDS = frozenset({"applicable", "concrete_anchors", "concrete_origin"})


def _extract_memory_display(mem: dict) -> tuple[str, str, str]:
    """Extract (title, description, content) from a memory entry.

    Handles multiple memory formats:
      - Cognitive dimension memories (type in {causal, contrastive, strategic,
        environment}, content in ``mem["memory"]``).
      - Derived traditional memories (type in {insight, workflow, summary, local},
        content in ``mem["insight"]``, ``mem["workflow"]``, etc.).
      - Legacy flat memories (``title`` / ``description`` / ``content`` at top level).

    For cognitive memories, ALL fields from ``mem["memory"]`` are formatted
    into the content (including concrete_anchors/concrete_origin).

    Returns (title, description, content) — all strings, may be empty.
    """
    mem_type = mem.get("type", "")
    level = mem.get("level", "")
    mem_data = mem.get("memory", {})

    # --- Derived traditional: insight ---
    if mem_type == "insight":
        inner = mem.get("insight", None) or mem_data
        if isinstance(inner, dict):
            return (inner.get("title", ""), inner.get("description", ""),
                    inner.get("content", ""))

    # --- Derived traditional: workflow ---
    if mem_type == "workflow":
        wf = mem.get("workflow", None) or mem_data
        if isinstance(wf, dict):
            title = wf.get("goal", f"Workflow from {mem.get('task_name', '')}")
            cmds = wf.get("workflow", [])
            content = "\n".join(f"  - {c}" for c in cmds[:10]) if cmds else ""
            return (title, "", content)

    # --- Derived traditional: summary ---
    if mem_type == "summary":
        title = mem.get("task_summary", f"Summary from {mem.get('task_name', '')}")
        desc = mem.get("experience_summary", "")
        return (title, desc, "")

    # --- Derived traditional: local ---
    if mem_type == "local" or "when_to_use" in mem:
        title = mem.get("when_to_use", f"Local memory from {mem.get('task_name', '')}")
        desc = mem.get("generalized_query", "")
        content = mem.get("experience", "")
        return (title, desc, content)

    # --- Derived traditional: trajectory ---
    if mem_type == "trajectory":
        task_name = mem.get("task_name", "")
        judgement = mem.get("judgement", "")
        status = "PASS" if judgement else "FAIL"
        title = f"Trajectory from {task_name} ({status})"
        desc = ""
        cmds = mem_data.get("commands", []) or mem.get("commands", [])
        if cmds:
            content = "\n".join(
                f"  {i+1}. {c[:300]}" if isinstance(c, str) else
                f"  {c.get('seq', i+1)}. {c.get('command', str(c))[:300]}"
                for i, c in enumerate(cmds[:15])
            )
        else:
            task_text = mem.get("task", "") or mem_data.get("task", "")
            content = task_text[:500] if task_text else ""
        return (title, desc, content)

    # ================================================================
    # Cognitive dimension memories (content in mem["memory"])
    # ================================================================

    # ---- Title ----
    dim_label_map = {"causal": "Causal", "contrastive": "Contrastive",
                     "strategic": "Strategic", "environment": "Environment"}
    lvl_label_map = {"trajectory": "Trajectory", "workflow": "Workflow",
                     "summary": "Summary", "insight": "Insight"}
    dim_label = dim_label_map.get(mem_type, mem_type)
    lvl_label = lvl_label_map.get(level, level)
    task_name = mem.get("task_name", "")

    title_parts = [dim_label, lvl_label]
    if task_name:
        title_parts.append(f"from {task_name}")
    title = " / ".join(t for t in title_parts if t)

    # ---- Description: pick 1-2 most descriptive fields ----
    combo_key = f"{level}/{mem_type}"
    desc_priority = _DESC_PRIORITY.get(combo_key, [])
    desc_parts = []
    for field in desc_priority:
        val = mem_data.get(field, "")
        if val and val != "other":
            if isinstance(val, dict):
                # Extract first meaningful string from dict
                for dk, dv in val.items():
                    if dv and isinstance(dv, str) and dv != "other":
                        desc_parts.append(f"{field}: {str(dv)[:200]}")
                        break
            elif isinstance(val, str):
                # Sentences longer than 8 words: just take first sentence
                first_sentence = val.split(".")[0].strip()
                if len(first_sentence.split()) > 3:
                    desc_parts.append(first_sentence + ".")
                else:
                    desc_parts.append(val[:200])
                break  # one good description is enough
            elif isinstance(val, list):
                # Take first string item
                for item in val:
                    if isinstance(item, str) and item and item != "other":
                        desc_parts.append(str(item)[:200])
                        break
                    elif isinstance(item, dict):
                        for dk, dv in item.items():
                            if dv and isinstance(dv, str) and dv != "other":
                                desc_parts.append(str(dv)[:200])
                                break
                break
    description = " | ".join(p for p in desc_parts if p and p != "other")

    # ---- Content: format ALL fields ----
    content_lines = []

    # Build a formatted representation of each field in mem_data.
    # Skip: applicable, concrete_anchors, concrete_origin (handled separately).
    # Also skip fields already used in the description.
    for field_name, field_val in mem_data.items():
        if field_name in _SKIP_FIELDS:
            continue
        label = field_name.replace("_", " ").capitalize()
        formatted = _format_value_as_text(field_val, label=label)
        content_lines.extend(formatted)

    # ---- Concrete Anchors (appended inline) ----
    anchors = (mem_data.get("concrete_anchors") or
               mem_data.get("concrete_origin") or
               mem.get("concrete_anchors") or {})
    if anchors:
        anchor_text = _format_concrete_anchors_display(anchors)
        if anchor_text:
            content_lines.append(f"[Anchors] {anchor_text}")

    content = "\n".join(line for line in content_lines if line)

    # ---- Fallback ----
    if not content and not description:
        flat_title = mem.get("title", "")
        flat_desc = mem.get("description", "")
        flat_content = mem.get("content", "")
        if flat_title or flat_desc or flat_content:
            return (flat_title, flat_desc, flat_content)
        task_text = mem.get("task", "")
        if task_text:
            return (task_text[:120] + ("..." if len(task_text) > 120 else ""), "",
                    task_text[:500])

    return (title, description, content)


def _extract_insight_display(ins: dict) -> tuple[str, str]:
    """Extract (title, content) from a linked insight entry.

    Handles both:
      - Derived traditional insight memories: content in ``ins["insight"]``
        with keys ``title`` / ``description`` / ``content``.
      - Cognitive insight-level memories: content in ``ins["memory"]``
        with keys like ``principle_name`` / ``principle_statement`` / etc.
    """
    # --- Derived traditional insight ---
    inner = ins.get("insight", None)
    if isinstance(inner, dict):
        t = inner.get("title", "")
        c = inner.get("content", "") or inner.get("description", "")
        if t or c:
            return (t, c)

    # --- Cognitive insight (content in ins["memory"]) ---
    mem_data = ins.get("memory", {})
    dim_type = ins.get("type", "")

    # Title: principle_name > methodology_name > positive_pattern > fallback
    title = (mem_data.get("principle_name", "") or
             mem_data.get("methodology_name", "") or
             mem_data.get("positive_pattern", "") or
             f"{dim_type} insight from {ins.get('task_name', '')}")

    # Content: format ALL fields from mem_data (same approach as main display)
    content_parts = []
    for field_name, field_val in mem_data.items():
        if field_name in _SKIP_FIELDS:
            continue
        label = field_name.replace("_", " ").capitalize()
        formatted = _format_value_as_text(field_val, label=label)
        content_parts.extend(formatted)

    # Concrete origin (insight-level equivalent of concrete_anchors)
    origin = mem_data.get("concrete_origin", {})
    if origin:
        anchor_text = _format_concrete_anchors_display(origin)
        if anchor_text:
            content_parts.append(f"[Origin] {anchor_text}")

    content = "\n".join(p for p in content_parts if p and p != "other")
    return (title, content)


def _inject_memory_context(
    task_dir: Path,
    pool_dir: Path,
    pipeline_config: dict,
) -> str | None:
    """Run retrieval and inject memory context into the task's instruction.md.

    Reads the original instruction, retrieves relevant memories from the pool,
    prepends them, and writes the augmented instruction back.

    Returns the original instruction text so it can be restored later.
    """
    # Guard: if memory is explicitly disabled, do NOT inject anything.
    if not pipeline_config.get("use_memory", True):
        print(f"  Memory injection SKIPPED: use_memory=false in pipeline config")
        return None

    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        return None

    original = instruction_path.read_text(encoding="utf-8")

    # Lazy-import retriever (needs sentence-transformers; set HF_HUB_OFFLINE first)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    _harbor_dir = str(Path(__file__).resolve().parent)
    if _harbor_dir not in sys.path:
        sys.path.insert(0, _harbor_dir)

    try:
        from experiments.utils.retriever import CognitiveRetriever
    except ImportError as e:
        print(f"  WARNING: Cannot import retriever: {e}")
        return None

    # Parse retrieval config from pipeline
    retrieval_config = pipeline_config.get("retrieval_config", {})
    features = retrieval_config.get("features", {})
    weights = retrieval_config.get("weights", {})
    retrieval = retrieval_config.get("retrieval", {})
    threshold = retrieval_config.get("threshold", {})

    retriever = CognitiveRetriever(
        memory_dir=str(pool_dir),
        use_cognitive_rerank=features.get("use_cognitive_rerank", True),
        use_llm_synergy=features.get("use_llm_synergy", True),
        use_synergy_selection=features.get("use_synergy_selection", True),
        alpha_dual_task=weights.get("alpha_dual_task", 0.70),
        alpha_semantic=weights.get("alpha_semantic", 0.35),
        alpha_cognitive=weights.get("alpha_cognitive", 0.65),
        top_n_candidates=retrieval.get("top_n_candidates", 20),
        top_k=retrieval.get("top_k", 3),
        min_memories=retrieval.get("min_memories", 1),
        score_threshold_floor=threshold.get("score_threshold_floor", 0.45),
        score_threshold_std=threshold.get("score_threshold_std", 0.5),
        attach_insights=features.get("attach_insights", True),
        retrieval_source=features.get("retrieval_source", "cognitive"),
        excluded_dimensions=pipeline_config.get("excluded_dimensions", []),
        excluded_levels=pipeline_config.get("excluded_levels", []),
    )

    try:
        if pipeline_config.get("use_random_retrieval"):
            # E15: random retrieval from top-N semantic candidates
            results = retriever.retrieve_random(
                original,
                top_n=retrieval.get("top_n_candidates", 20),
                top_k=retrieval.get("top_k", 3),
            )
        elif pipeline_config.get("use_direct_llm_topk"):
            # E16: LLM direct top-K selection without dimension structure
            results = retriever.retrieve_llm_direct(
                original,
                top_n=retrieval.get("top_n_candidates", 20),
                top_k=retrieval.get("top_k", 3),
            )
        else:
            results = retriever.retrieve(original)
    except Exception as e:
        print(f"  WARNING: Retrieval failed: {e}")
        return None

    if not results:
        print(f"  Memory retrieval: no relevant memories found")
        return original  # No memories to inject, but return original for restore

    # Check if retrieved memories have any actual content (defense against
    # empty-memory syndrome where template is injected with blank fields).
    has_content = False
    for mem in results:
        title, desc, content = _extract_memory_display(mem)
        if title.strip() or desc.strip() or content.strip():
            has_content = True
            break
        for ins in mem.get("_linked_insights", []):
            ins_title, ins_content = _extract_insight_display(ins)
            if ins_title.strip() or ins_content.strip():
                has_content = True
                break
        if has_content:
            break

    if not has_content:
        print(f"  Memory retrieval: memories found but all have empty content — SKIPPING injection")
        return original

    # Format memory context with actual extracted content
    lines = [
        "## Memory Context (from previous tasks)",
        "",
        "The following experiences from similar tasks may be helpful:",
        "",
    ]
    for i, mem in enumerate(results, 1):
        title, description, content = _extract_memory_display(mem)
        score = mem.get("combined_score", mem.get("semantic_score", 0))

        lines.append(f"### Memory {i}: {title}  (score={score:.2f})")
        if content:
            lines.append(f"  {content}")
        lines.append("")

        # Attach linked insights if any
        insights = mem.get("_linked_insights", [])
        for ins in insights:
            ins_title, ins_content = _extract_insight_display(ins)
            if not ins_title.strip() and not ins_content.strip():
                continue  # skip empty insights
            lines.append(f"  > Insight: {ins_title}")
            if ins_content:
                lines.append(f"  > {ins_content}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(original)

    augmented = "\n".join(lines)
    instruction_path.write_text(augmented, encoding="utf-8")
    print(f"  Memory retrieval: injected {len(results)} memories into instruction")
    return original


def _build_eval_key(pipeline_config: dict, agent: str, model: str) -> str:
    """Build the harbor eval key: <agent>__<model>__<benchmark>."""
    benchmark = pipeline_config.get("tasks_dir", "swebench-verified")
    if "/" in benchmark:
        benchmark = benchmark.rsplit("/", 1)[-1]
    model_short = model.split("/", 1)[-1] if "/" in model else model
    return f"{agent}__{model_short}__{benchmark}"


def _write_result_json_incremental(
    run_job_dir: Path,
    eval_key: str,
    results: list[dict],
    started_at: str,
    finished_at: str | None,
    is_complete: bool,
):
    """Write (or update) the aggregated result.json after each task.

    Called after every round so the file is always up to date.  When
    ``is_complete=False`` the ``finished_at`` field is left null to
    indicate a run in progress.
    """
    import uuid

    passed_trial_ids: list[str] = []
    failed_trial_ids: list[str] = []
    errored_trial_ids: dict[str, list[str]] = {}
    total_tokens_in = 0
    total_tokens_cache = 0
    total_tokens_out = 0

    for r in results:
        # trial_id is the actual harbor-generated directory name
        # (e.g. "astropy__astropy-12907__BVe9axb"), which matches the
        # on-disk task folder exactly.
        trial_id = r.get("trial_id", r["task"])
        if r.get("errored"):
            err_type = r.get("error_type", "UnknownError")
            errored_trial_ids.setdefault(err_type, []).append(trial_id)
        elif r["passed"]:
            passed_trial_ids.append(trial_id)
        else:
            failed_trial_ids.append(trial_id)

        total_tokens_in += r.get("n_input_tokens", 0) or 0
        total_tokens_cache += r.get("n_cache_tokens", 0) or 0
        total_tokens_out += r.get("n_output_tokens", 0) or 0

    n_trials = len([r for r in results if not r.get("errored")])
    n_errors = sum(len(v) for v in errored_trial_ids.values())
    n_total = len(results)
    pass_count = len(passed_trial_ids)
    mean = pass_count / max(n_trials, 1)

    reward_stats: dict[str, list[str]] = {}
    if failed_trial_ids:
        reward_stats["0.0"] = failed_trial_ids
    if passed_trial_ids:
        reward_stats["1.0"] = passed_trial_ids

    eval_entry: dict = {
        "n_trials": n_trials,
        "n_errors": n_errors,
        "metrics": [{"mean": round(mean, 4)}],
        "reward_stats": {"reward": reward_stats} if reward_stats else {},
    }
    if errored_trial_ids:
        eval_entry["exception_stats"] = dict(errored_trial_ids)
    if total_tokens_in:
        eval_entry["n_input_tokens"] = total_tokens_in
        eval_entry["n_cache_tokens"] = total_tokens_cache
        eval_entry["n_output_tokens"] = total_tokens_out

    result: dict = {
        "id": str(uuid.uuid4()),
        "started_at": started_at,
        "finished_at": finished_at,
        "n_total_trials": n_total,
        "stats": {
            "n_trials": n_trials,
            "n_errors": n_errors,
            "evals": {eval_key: eval_entry},
        },
    }

    result_path = run_job_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=4, ensure_ascii=False))

    if is_complete:
        print(f"Aggregated result saved to: {result_path}")
    else:
        progress = f"{pass_count}/{n_trials}" if n_trials else "0/0"
        print(f"  [result.json] updated ({progress} passed, mean={mean:.3f})")


def _restore_instruction(task_dir: Path, original: str | None):
    """Restore the original instruction.md after harbor completes."""
    if original is None:
        return
    instruction_path = task_dir / "instruction.md"
    if instruction_path.exists():
        instruction_path.write_text(original, encoding="utf-8")


def _run_sequential_pipeline(
    selected: list[str],
    tasks_dir: str,
    agent: str,
    model: str,
    jobs_dir: str,
    enable_memory: bool = False,
    memory_path: str = "",
    agent_kwargs_list: list[dict] | None = None,
    pipeline_config: dict | None = None,
    args: argparse.Namespace | None = None,
    harbor_exe: str = "",
):
    """Run tasks one at a time, with optional in-memory accumulation.

    Always sequential — one ``harbor jobs start`` per task.  When
    ``enable_memory=True``, relevant memories are injected into the task
    instruction before each round and extracted after each completed task.
    All trial directories are consolidated under a single timestamp::

        jobs/<name>/2026-06-21__12-30-00/
          astropy__astropy-12907__xxx/
          astropy__astropy-13033__xxx/
          ...
    """
    from datetime import datetime
    import re

    if agent_kwargs_list is None:
        agent_kwargs_list = []
    if pipeline_config is None:
        pipeline_config = {}

    # ---- Guard: prevent nested runs inside an existing timestamp directory ----
    # This catches misconfigurations where --jobs-dir points inside a prior run.
    _ts_pattern = re.compile(r"\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}")
    _jobs_path = Path(jobs_dir).resolve()
    for _parent in _jobs_path.parents:
        if _ts_pattern.match(_parent.name):
            print(f"ERROR: jobs_dir '{jobs_dir}' appears to be nested inside a "
                  f"previous run timestamp directory ('{_parent.name}').")
            print(f"  This would cause harbor output directories to mix with "
                  f"task instance directories.")
            print(f"  Use the parent directory instead: {_parent.parent}")
            sys.exit(1)

    # ---- One run timestamp for ALL rounds ----
    run_ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    run_job_dir = Path(jobs_dir) / run_ts
    run_job_dir.mkdir(parents=True, exist_ok=True)
    # Sleep 1 second so the next harbor invocation gets a DIFFERENT timestamp.
    # Without this, harbor may reuse run_job_dir as its own job directory and
    # overwrite our result.json / lock.json.
    time.sleep(1.1)

    pool_dir = Path(memory_path) if memory_path else None
    if enable_memory and pool_dir:
        pool_dir.mkdir(parents=True, exist_ok=True)

    only_passed = pipeline_config.get("only_passed", True)
    derive_traditional = pipeline_config.get("derive_traditional_memory", True)
    excluded_dimensions = pipeline_config.get("excluded_dimensions") or []
    excluded_levels = pipeline_config.get("excluded_levels") or []
    python_exe = _find_python_exe() if enable_memory else None
    jobs_dir_path = Path(jobs_dir)
    tasks_dir_path = Path(tasks_dir)
    eval_key = _build_eval_key(pipeline_config, agent, model)

    print(f"  Run dir:      {run_job_dir}")
    if enable_memory:
        print(f"  Memory:       ENABLED  (pool: {memory_path})")
        print(f"  Extract py:   {python_exe}")
    else:
        print(f"  Memory:       DISABLED")

    passed_count = 0
    total_count = 0
    error_count = 0
    start_time = time.time()
    started_at_iso = datetime.now().isoformat()
    results: list[dict] = []

    def _build_harbor_cmd(task_name: str) -> list[str]:
        """Build a harbor jobs start command for a single task."""
        cmd = [harbor_exe, "-m", "harbor.cli.main", "jobs", "start"]

        cmd.extend([
            "-p", str(tasks_dir),
            "-a", agent,
            "-m", model,
            "--jobs-dir", jobs_dir,
            "-n", "1",
            "-i", task_name,
        ])

        if args.env_class:
            cmd.extend(["-e", args.env_class])

        for ak in agent_kwargs_list:
            cmd.extend(["--ak", ak])

        if args.ek:
            cmd.extend(["--ek", args.ek])

        if args.force_build:
            cmd.append("--force-build")

        return cmd

    for i, task_name in enumerate(selected):
        round_num = i + 1
        total_count += 1
        round_start = time.time()

        print(f"\n{'─' * 60}")
        print(f"[Round {round_num}/{len(selected)}] {task_name}")

        # ---- Memory pool status ----
        if enable_memory and pool_dir:
            pkl_count = len(list(pool_dir.rglob("*.pkl"))) if pool_dir.exists() else 0
            n_prior_tasks = _count_pool_tasks(pool_dir)
            has_pool = pkl_count > 0
            if has_pool:
                print(f"  Memory pool: {pkl_count} pkl files "
                      f"from {n_prior_tasks} previous tasks")
            else:
                print(f"  Memory pool: empty (first task, no memory)")
        else:
            has_pool = False

        # ---- Step 0: Retrieve & inject memory context (memory pipelines only) ----
        task_dir = tasks_dir_path / task_name
        original_instruction = None
        if has_pool:
            print(f"  [0/3] Retrieving relevant memories...")
            original_instruction = _inject_memory_context(
                task_dir, pool_dir, pipeline_config,
            )

        # ---- Step 1: Run harbor for this single task ----
        jobs_dir_path.mkdir(parents=True, exist_ok=True)
        before_dirs = set(d.name for d in jobs_dir_path.iterdir() if d.is_dir())

        cmd = _build_harbor_cmd(task_name)

        print(f"  [1/3] Running agent...")
        print(f"  Command: {' '.join(cmd[:5])} ... -i {task_name}")

        harbor_error_type = ""
        try:
            result = subprocess.run(cmd, check=False, timeout=3600)
            if result.returncode != 0:
                print(f"  WARNING: harbor returned exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: harbor timed out for {task_name}")
            harbor_error_type = "AgentTimeoutError"
        except Exception as exc:
            print(f"  WARNING: harbor subprocess failed: {exc}")
            harbor_error_type = type(exc).__name__

        # Restore original instruction.md
        _restore_instruction(task_dir, original_instruction)

        # Identify harbor's newly created timestamp directory
        after_dirs = set(d.name for d in jobs_dir_path.iterdir() if d.is_dir())
        new_dirs = after_dirs - before_dirs
        harbor_job_dir = jobs_dir_path / new_dirs.pop() if new_dirs else None

        # ---- Move trial from harbor's timestamp dir into our run_job_dir ----
        # Harbor may reuse run_job_dir as its own timestamp dir when the
        # second-precision timestamps collide.  Therefore we look in BOTH
        # harbor_job_dir (newly created) AND run_job_dir (possibly reused).
        trial_dir = None
        search_dirs = [run_job_dir]
        if harbor_job_dir and harbor_job_dir.resolve() != run_job_dir.resolve():
            search_dirs.insert(0, harbor_job_dir)

        for search_dir in search_dirs:
            found = _find_trial_dir(search_dir, task_name)
            if found:
                if search_dir.resolve() != run_job_dir.resolve():
                    dest = run_job_dir / found.name
                    # If destination already exists (from a previous re-run),
                    # remove it first.
                    if dest.exists():
                        shutil.rmtree(str(dest), ignore_errors=True)
                    shutil.move(str(found), str(dest))
                    trial_dir = dest
                else:
                    # Trial is already in run_job_dir (harbor reused our dir)
                    trial_dir = found
                break

        # Clean up harbor's timestamp dir (only if it's a separate dir)
        if harbor_job_dir and harbor_job_dir.resolve() != run_job_dir.resolve():
            shutil.rmtree(str(harbor_job_dir), ignore_errors=True)

        # ---- Step 2: Extract memories (memory pipelines only) ----
        extracted = False
        if not enable_memory:
            print(f"  [2/3] SKIP extraction: memory disabled")
        elif args.no_extract:
            print(f"  [2/3] SKIP extraction: --no-extract flag set")
        elif trial_dir is not None:
            existing_trials = [d for d in run_job_dir.iterdir()
                             if d.is_dir() and "__" in d.name]
            extraction_start = len(existing_trials) - 1

            print(f"  [2/3] Extracting memories...")
            extracted = _extract_memories_for_task(
                task_name, run_job_dir, pool_dir,
                only_passed=only_passed,
                derive_traditional=derive_traditional,
                python_exe=python_exe,
                start_index=extraction_start,
                excluded_dimensions=excluded_dimensions,
                pipeline_config=pipeline_config,
                excluded_levels=excluded_levels,
            )
        else:
            print(f"  [2/3] SKIP extraction: trial directory not found")

        # ---- Step 3: Record result ----
        passed = _get_judgement(trial_dir) if trial_dir else False
        errored = trial_dir is None
        error_type = harbor_error_type
        if passed:
            passed_count += 1
        if errored:
            error_count += 1

        # Gather trial_id (harbor-generated dir name like "django__django-10097__XyZ123")
        trial_id = trial_dir.name if trial_dir else task_name

        # Gather per-trial token counts from harbor's result.json.
        # Harbor stores tokens in the top-level "agent_result" key, not "stats".
        n_input_tokens = 0
        n_cache_tokens = 0
        n_output_tokens = 0
        if trial_dir:
            trial_result_path = trial_dir / "result.json"
            if trial_result_path.exists():
                try:
                    tr = json.loads(trial_result_path.read_text(encoding="utf-8"))
                    ar = tr.get("agent_result", {})
                    n_input_tokens = ar.get("n_input_tokens") or 0
                    n_cache_tokens = ar.get("n_cache_tokens") or 0
                    n_output_tokens = ar.get("n_output_tokens") or 0
                except (json.JSONDecodeError, OSError):
                    pass

        elapsed = time.time() - round_start
        status = "PASS" if passed else "FAIL"
        cum_pass_rate = passed_count / total_count * 100 if total_count > 0 else 0

        print(f"  [3/3] Result: {status} | "
              f"Round time: {elapsed:.0f}s | "
              f"Cumulative: {passed_count}/{total_count} ({cum_pass_rate:.1f}%) | "
              f"Extracted: {'YES' if extracted else 'NO'}")

        results.append({
            "round": round_num,
            "task": task_name,
            "trial_id": trial_id,
            "passed": passed,
            "errored": errored,
            "error_type": error_type,
            "memories_extracted": extracted,
            "pool_tasks_before": _count_pool_tasks(pool_dir) if pool_dir else 0,
            "elapsed_sec": round(elapsed, 1),
            "n_input_tokens": n_input_tokens,
            "n_cache_tokens": n_cache_tokens,
            "n_output_tokens": n_output_tokens,
        })

        # Write updated result.json after EVERY round so it stays in sync
        # with the on-disk task directories.
        _write_result_json_incremental(
            run_job_dir=run_job_dir,
            eval_key=eval_key,
            results=results,
            started_at=started_at_iso,
            finished_at=None,   # null = run in progress
            is_complete=False,
        )

    # ---- Final summary ----
    total_elapsed = time.time() - start_time
    final_pass_rate = passed_count / total_count * 100 if total_count > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"Sequential pipeline complete.")
    print(f"  Pipeline: {pipeline_config.get('name', 'unknown')}")
    print(f"  Run dir:    {run_job_dir}")
    print(f"  Tasks:      {total_count}")
    print(f"  Passed:     {passed_count}/{total_count} ({final_pass_rate:.1f}%)")
    print(f"  Total time: {total_elapsed / 60:.1f} min")
    if enable_memory and pool_dir:
        print(f"  Memory pool: {_count_pool_tasks(pool_dir)} tasks contributed")
    print(f"{'=' * 70}")

    finished_at_iso = datetime.now().isoformat()

    # Save summary inside the run directory
    summary_path = run_job_dir / "sequential_summary.json"
    summary_path.write_text(json.dumps({
        "name": pipeline_config.get("name", "unknown"),
        "run_ts": run_ts,
        "total_tasks": total_count,
        "passed": passed_count,
        "pass_rate": round(final_pass_rate, 1),
        "total_elapsed_sec": round(total_elapsed, 0),
        "selected_tasks": selected,
        "random_seed": pipeline_config.get("random_seed"),
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"Summary saved to: {summary_path}")

    # Write final result.json with finished_at timestamp set
    _write_result_json_incremental(
        run_job_dir=run_job_dir,
        eval_key=eval_key,
        results=results,
        started_at=started_at_iso,
        finished_at=finished_at_iso,
        is_complete=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run harbor tasks sequentially, one per harbor call.  "
                    "Memory pipelines also extract experience between rounds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Memory pipelines
  python harbor/run_tasks.py --pipeline e1_full --start 0 --end 500 --ak step_limit=150 --force-build

  # No-memory baseline
  python harbor/run_tasks.py --pipeline e24_no_memory --start 0 --end 500 --ak step_limit=150 --force-build
        """,
    )

    # ---- Pipeline ----
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Pipeline config name (e.g. 'e24_no_memory') or path to YAML.  "
             f"Searched in {EXPERIMENTS_DIR}",
    )

    # ---- Task selection ----
    parser.add_argument(
        "--tasks-dir", default="harbor-tasks/swebench-verified",
        help="Directory containing task subdirectories",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start task index (0-based, inclusive)",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="End task index (exclusive), default = all remaining",
    )

    # ---- Agent ----
    parser.add_argument(
        "--agent", "-a", default=None,
        help="Agent name (default from pipeline or 'mini-swe-agent')",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Model name (default from MODEL env, pipeline config, or deepseek/deepseek-chat)",
    )
    parser.add_argument(
        "--ak", "--agent-kwarg", dest="agent_kwargs", action="append", default=None,
        help="Agent kwarg in key=value format (e.g. --ak step_limit=100).  "
             "Can be used multiple times.  Merged with pipeline agent_kwargs.",
    )

    # ---- Environment ----
    parser.add_argument(
        "--env", "--environment-class", dest="env_class", default=None,
        help="Environment class: docker (default), daytona, e2b, modal, runloop",
    )
    parser.add_argument(
        "--ek", default=None,
        help="Extra key=value pairs for harbor (e.g. run_dir=...)",
    )

    # ---- Job orchestration ----
    parser.add_argument(
        "--jobs-dir", default=None,
        help="Jobs output directory (default from pipeline or 'jobs/YYYY-MM-DD')",
    )
    parser.add_argument(
        "--force-build", action="store_true",
        help="Force rebuild of Docker image",
    )

    # ---- Memory ----
    parser.add_argument(
        "--memory-path", default=None,
        help="Path to memory files (auto-set from pipeline pool_dir).  "
             "Use '--no-memory' to explicitly disable.",
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Disable memory retrieval (overrides pipeline pool_dir)",
    )
    parser.add_argument(
        "--no-extract", action="store_true",
        help="Skip memory extraction between rounds (memory pipelines only).  "
             "Useful when you only want retrieval from a pre-built pool.",
    )

    # ---- Misc ----
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print command without executing",
    )
    args = parser.parse_args()

    # =====================================================================
    # Load pipeline config
    # =====================================================================
    pipeline_config: dict = {}
    if args.pipeline:
        pipeline_path = _find_pipeline_config(args.pipeline)
        if pipeline_path is None:
            available = []
            if EXPERIMENTS_DIR.exists():
                available = sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.yaml"))
            print(f"ERROR: Pipeline config not found: {args.pipeline}")
            print(f"Searched in: {EXPERIMENTS_DIR}")
            if available:
                print(f"Available pipelines: {', '.join(available)}")
            sys.exit(1)
        pipeline_config = _load_pipeline_yaml(pipeline_path)

        print(f"Pipeline: {pipeline_config.get('name', pipeline_path.stem)}")
        desc = pipeline_config.get("description", "")
        if desc:
            print(f"  {desc}")

    # =====================================================================
    # Resolve: model
    # =====================================================================
    if args.model:
        model = _resolve_model_name(args.model)
        model_source = "CLI"
    else:
        env_model = os.environ.get("MODEL", "")
        if env_model:
            model = _resolve_model_name(env_model)
            model_source = "MODEL env"
        elif pipeline_config.get("model"):
            model = _resolve_model_name(pipeline_config["model"])
            model_source = "pipeline"
        else:
            model = "deepseek/deepseek-chat"
            model_source = "default"

    # =====================================================================
    # Resolve: agent
    # =====================================================================
    agent = args.agent or pipeline_config.get("agent") or "mini-swe-agent"

    # =====================================================================
    # Resolve: jobs-dir
    # =====================================================================
    if args.jobs_dir:
        jobs_dir = args.jobs_dir
        jobs_source = "CLI"
    elif pipeline_config.get("jobs_dir"):
        jobs_dir = pipeline_config["jobs_dir"]
        jobs_source = "pipeline"
    else:
        from datetime import date
        today = date.today().isoformat()
        jobs_dir = f"jobs/{today}"
        jobs_source = "default"

    # =====================================================================
    # Resolve: memory
    #
    # Priority (highest to lowest):
    #   1. CLI  --no-memory          → force off
    #   2. CLI  --memory-path PATH   → force on, use PATH
    #   3. Config  use_memory: true/false  → explicit YAML switch
    #   4. Config  pool_dir (legacy) → non-empty = on, empty = off
    #   5. Default                    → off
    # =====================================================================
    use_memory = False
    memory_path = args.memory_path

    if args.no_memory:
        use_memory = False
        memory_path = None
    elif memory_path:
        use_memory = True
    elif pipeline_config:
        config_use_memory = pipeline_config.get("use_memory")
        if config_use_memory is not None:
            # New explicit flag
            use_memory = bool(config_use_memory)
            if use_memory:
                memory_path = pipeline_config.get("pool_dir", "")
            else:
                memory_path = None
        else:
            # Legacy: infer from pool_dir
            pool_dir = pipeline_config.get("pool_dir", "")
            if pool_dir == "" and "pool_dir" in pipeline_config:
                use_memory = False
                memory_path = None
            elif pool_dir:
                use_memory = True
                memory_path = pool_dir
            else:
                use_memory = False
                memory_path = None

    # =====================================================================
    # Resolve: agent_kwargs
    # =====================================================================
    agent_kwargs_list = list(args.agent_kwargs or [])

    pipeline_ak = pipeline_config.get("agent_kwargs", {}) or {}
    cli_ak_keys = set()
    for ak in agent_kwargs_list:
        if "=" in ak:
            cli_ak_keys.add(ak.split("=", 1)[0])

    for key, value in pipeline_ak.items():
        if key not in cli_ak_keys:
            agent_kwargs_list.append(f"{key}={value}")

    # =====================================================================
    # Locate tasks
    # =====================================================================
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"ERROR: tasks-dir not found: {tasks_dir}")
        sys.exit(1)

    all_tasks = sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())
    print(f"Total tasks found: {len(all_tasks)}")

    # ---- Task selection ----
    # When pipeline config specifies num_tasks (random sampling), always
    # sample first, then apply --start/--end as a slice on the result.
    # This way "python run_tasks.py --pipeline e3 --start 0 --end 2" tests
    # the first 2 tasks of the actual experiment set (seed=42), not the
    # first 2 tasks of the full 500-task list.
    #
    # Only when num_tasks is null (E1/E2/E7 run all 500) does --start/--end
    # slice directly from the full task list.
    num_tasks = pipeline_config.get("num_tasks")
    random_seed = pipeline_config.get("random_seed", 42)

    if num_tasks is not None and isinstance(num_tasks, int) and num_tasks > 0:
        # Random sampling from pipeline config (reproducible via random_seed)
        import random as _random
        rng = _random.Random(random_seed)
        if num_tasks < len(all_tasks):
            sampled = sorted(rng.sample(all_tasks, num_tasks))
        else:
            sampled = all_tasks.copy()
        print(f"Randomly sampled {len(sampled)}/{len(all_tasks)} tasks (seed={random_seed})")
        # Apply --start/--end as slice on the sampled list
        selected = sampled[args.start:args.end]
        if args.start != 0 or args.end is not None:
            end_idx = args.end if args.end is not None else len(sampled)
            print(f"  -> subset via --start/--end: [{args.start}:{end_idx}] = {len(selected)} tasks")
    else:
        # Full run: slice from all tasks (for E1/E2/E7 or manual --start/--end)
        selected = all_tasks[args.start:args.end]
        end_idx = args.end if args.end is not None else len(all_tasks)
        print(f"Selected range: [{args.start}:{end_idx}] = {len(selected)} tasks")

    if selected:
        print(f"  First: {selected[0]}")
        print(f"  Last:  {selected[-1]}")
    else:
        print("  (empty range, nothing to run)")
        sys.exit(0)

    # =====================================================================
    # Print configuration summary
    # =====================================================================
    print()
    print("─" * 60)
    print(f"  Model:       {model}  (from {model_source})")
    print(f"  Agent:       {agent}")
    print(f"  Jobs dir:    {jobs_dir}  (from {jobs_source})")
    print(f"  Memory:      {'YES' if use_memory else 'NO'}"
          f"{'  (pool: ' + memory_path + ')' if use_memory and memory_path else ''}")
    if use_memory and memory_path:
        only_passed = pipeline_config.get("only_passed", True)
        derive_traditional = pipeline_config.get("derive_traditional_memory", True)
        print(f"  Only passed: {only_passed}")
        print(f"  Derive trad: {derive_traditional}")
        if args.no_extract:
            print(f"  Extraction:  DISABLED (--no-extract)")
    if agent_kwargs_list:
        print(f"  Agent kwargs: {agent_kwargs_list}")
    if args.force_build:
        print(f"  Force build: YES")
    if args.ek:
        print(f"  Extra env:   {args.ek}")
    print("─" * 60)
    print()

    # =====================================================================
    # Find harbor executable
    # =====================================================================
    harbor_exe = _find_harbor_exe()
    if harbor_exe is None:
        print("ERROR: No Python venv found.  Cannot run harbor.")
        print("  Tried: .venv-harbor/bin/python3, .venv/bin/python3")
        sys.exit(1)

    # =====================================================================
    # Execute — always sequential
    # =====================================================================
    if args.dry_run:
        if use_memory and memory_path:
            print(f"[dry-run] Would run {len(selected)} tasks sequentially "
                  f"with per-task memory extraction.")
        else:
            print(f"[dry-run] Would run {len(selected)} tasks sequentially "
                  f"(no memory).")
        return

    _run_sequential_pipeline(
        selected=selected,
        tasks_dir=str(tasks_dir),
        agent=agent,
        model=model,
        jobs_dir=jobs_dir,
        enable_memory=use_memory,
        memory_path=memory_path or "",
        agent_kwargs_list=agent_kwargs_list,
        pipeline_config=pipeline_config,
        args=args,
        harbor_exe=harbor_exe,
    )


if __name__ == "__main__":
    main()
