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
        reward = result.get("verifier_result", {}).get("rewards", {}).get("reward", 0.0)
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
    python_exe: str | None = None,
    start_index: int = 0,
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

    # Check if we should skip based on pass/fail
    judgement = _get_judgement(trial_dir)
    status = "PASS" if judgement else "FAIL"

    if only_passed and not judgement:
        print(f"  SKIP extraction: task {status}, only_passed=True")
        return False

    extract_script = str(Path(__file__).resolve().parent / "extract_memories.py")
    cmd = [
        python_exe, extract_script,
        "--jobs-dir", str(jobs_dir),
        "--memory-dir", str(pool_dir),
        "--limit", "1",
        "--start", str(start_index),
    ]

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")

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
    )

    try:
        results = retriever.retrieve(original)
    except Exception as e:
        print(f"  WARNING: Retrieval failed: {e}")
        return None

    if not results:
        print(f"  Memory retrieval: no relevant memories found")
        return original  # No memories to inject, but return original for restore

    # Format memory context
    lines = [
        "## Memory Context (from previous tasks)",
        "",
        "The following experiences from similar tasks may be helpful:",
        "",
    ]
    for i, mem in enumerate(results, 1):
        title = mem.get("title", "")
        description = mem.get("description", "")
        content = mem.get("content", "")
        dimension = mem.get("dimension", "")
        score = mem.get("final_score", mem.get("score", 0))

        lines.append(f"### Memory {i}: {title}")
        if dimension:
            lines.append(f"  Dimension: {dimension}  |  Score: {score:.3f}")
        if description:
            lines.append(f"  {description}")
        if content:
            lines.append(f"  {content}")
        lines.append("")

        # Attach linked insights if any
        insights = mem.get("_linked_insights", [])
        for ins in insights:
            lines.append(f"  **Insight**: {ins.get('title', '')}")
            ins_content = ins.get("content", "")
            if ins_content:
                lines.append(f"  {ins_content}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(original)

    augmented = "\n".join(lines)
    instruction_path.write_text(augmented, encoding="utf-8")
    print(f"  Memory retrieval: injected {len(results)} memories into instruction")
    return original


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

    if agent_kwargs_list is None:
        agent_kwargs_list = []
    if pipeline_config is None:
        pipeline_config = {}

    # ---- One run timestamp for ALL rounds ----
    run_ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    run_job_dir = Path(jobs_dir) / run_ts
    run_job_dir.mkdir(parents=True, exist_ok=True)

    pool_dir = Path(memory_path) if memory_path else None
    if enable_memory and pool_dir:
        pool_dir.mkdir(parents=True, exist_ok=True)

    only_passed = pipeline_config.get("only_passed", True)
    python_exe = _find_python_exe() if enable_memory else None
    jobs_dir_path = Path(jobs_dir)
    tasks_dir_path = Path(tasks_dir)

    print(f"  Run dir:      {run_job_dir}")
    if enable_memory:
        print(f"  Memory:       ENABLED  (pool: {memory_path})")
        print(f"  Extract py:   {python_exe}")
    else:
        print(f"  Memory:       DISABLED")

    passed_count = 0
    total_count = 0
    start_time = time.time()
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

        try:
            result = subprocess.run(cmd, check=False, timeout=3600)
            if result.returncode != 0:
                print(f"  WARNING: harbor returned exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: harbor timed out for {task_name}")

        # Restore original instruction.md
        _restore_instruction(task_dir, original_instruction)

        # Identify harbor's newly created timestamp directory
        after_dirs = set(d.name for d in jobs_dir_path.iterdir() if d.is_dir())
        new_dirs = after_dirs - before_dirs
        harbor_job_dir = jobs_dir_path / new_dirs.pop() if new_dirs else None

        # ---- Move trial from harbor's timestamp dir into our run_job_dir ----
        trial_dir = None
        if harbor_job_dir:
            found = _find_trial_dir(harbor_job_dir, task_name)
            if found:
                dest = run_job_dir / found.name
                shutil.move(str(found), str(dest))
                trial_dir = dest
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
                python_exe=python_exe,
                start_index=extraction_start,
            )
        else:
            print(f"  [2/3] SKIP extraction: trial directory not found")

        # ---- Step 3: Record result ----
        passed = _get_judgement(trial_dir) if trial_dir else False
        if passed:
            passed_count += 1

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
            "passed": passed,
            "memories_extracted": extracted,
            "pool_tasks_before": _count_pool_tasks(pool_dir) if pool_dir else 0,
            "elapsed_sec": round(elapsed, 1),
        })

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

    # ---- Task selection: CLI slice > pipeline random sample > pipeline slice ----
    cli_overrides_selection = '--start' in sys.argv or '--end' in sys.argv
    num_tasks = pipeline_config.get("num_tasks")
    random_seed = pipeline_config.get("random_seed", 42)

    if not cli_overrides_selection and num_tasks is not None and isinstance(num_tasks, int) and num_tasks > 0:
        # Random sampling from pipeline config (reproducible via random_seed)
        import random as _random
        rng = _random.Random(random_seed)
        if num_tasks < len(all_tasks):
            selected = sorted(rng.sample(all_tasks, num_tasks))
        else:
            selected = all_tasks.copy()
        print(f"Randomly sampled {len(selected)}/{len(all_tasks)} tasks (seed={random_seed})")
    else:
        # Slice-based selection (existing behavior)
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
