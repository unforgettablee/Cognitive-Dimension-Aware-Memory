#!/usr/bin/env python3

"""MTL-augmented SWE-bench runner: sequential execution with incremental memory pool.

Key differences from ``swebench.py``:

* **Sequential execution** (no ThreadPoolExecutor) — the memory pool grows after
  each task, so the next task benefits from all prior experiences.
* **Memory retrieval** before each task via `CognitiveRetriever` against ``pool_dir/*.pkl``.
* **Memory extraction** after each task via the MTL extraction pipeline (cognitive
  + traditional), writing new ``.pkl`` files into ``pool_dir/``.
* **MTLAgent** is used so that ``{{memory_context}}`` is rendered into the
  ``instance_template`` alongside the task description.

Usage::

    mini-extra swebench-mtl \\
      --subset verified --split test --slice 0:500 \\
      --model deepseek/deepseek-v4-flash \\
      --environment-class local \\
      --pool-dir memories/smoke-test \\
      --jobs-dir jobs/smoke-test \\
      -o outputs/baseline \\
      --retrieval-config harbor/config/cognitive_retrieval.yaml \\
      --only-passed
"""

import json
import os
import time
import traceback
from pathlib import Path

import typer
from jinja2 import StrictUndefined, Template

from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

# ---------------------------------------------------------------------------
# Graceful MTL import (it may not be installed)
# ---------------------------------------------------------------------------
_MTL_AVAILABLE = False
try:
    from mtl.llm import configure as mtl_configure  # noqa: F401
    from mtl.retrieval import CognitiveRetriever

    from mtl.extraction.extractor import convert_trajectory, get_judgement
    from mtl.extraction.cognitive_extract import extract_cognitive_memories
    from mtl.extraction.traditional_extract import extract_rawtraj_memory

    _MTL_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Shared imports from swebench.py (dataset, environment helpers)
# ---------------------------------------------------------------------------
from minisweagent.run.benchmarks.swebench import (  # noqa: E402
    DATASET_MAPPING,
    DEFAULT_CONFIG_FILE,
    ProgressTrackingAgent,
    filter_instances,
    get_sb_environment,
    remove_from_preds_file,
    update_preds_file,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_HELP_TEXT = """Run MTL-augmented SWE-bench sequentially with incremental memory pool.

[not dim]
Tasks are executed one-by-one. After each task:
1. The trajectory is saved to ``--jobs-dir/{instance_id}/``.
2. Memories are extracted (cognitive + traditional) and written as ``.pkl`` files
   into ``--pool-dir/``.
3. The memory pool grows, so the next task retrieves richer context.

Supports ``--resume-from N`` to continue interrupted runs.
[/not dim]
"""


# ---------------------------------------------------------------------------
# Memory helpers (ported from harbor/trial.py)
# ---------------------------------------------------------------------------

_PIPELINE_SEARCH_DIRS = [
    Path("harbor/configs/experiments"),
    Path("configs/experiments"),
]


def _find_pipeline_config(name_or_path: str) -> Path | None:
    """Resolve a pipeline config name to an actual YAML file.

    Search order:
      1. Exact path (if the string is an existing file)
      2. ``harbor/configs/experiments/{name}.yaml``
      3. ``configs/experiments/{name}.yaml``
    """
    direct = Path(name_or_path)
    if direct.exists():
        return direct

    for base in _PIPELINE_SEARCH_DIRS:
        candidate = base / f"{name_or_path}.yaml"
        if candidate.exists():
            return candidate

    return None


def _load_pipeline_yaml(pipeline_path: Path) -> dict:
    """Load a pipeline experiment YAML and return its configuration dict."""
    import yaml

    with open(pipeline_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_retrieval_config(config_path: Path | None) -> dict:
    """Load CognitiveRetriever parameters from a YAML file.

    Args:
        config_path: Path to YAML file.  If ``None`` or missing, returns ``{}``
            (callers apply their own defaults).
    """
    if config_path is None or not config_path.exists():
        return {}
    import yaml

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _init_retriever(pool_dir: Path, retrieval_config: dict) -> "CognitiveRetriever":
    """Create a ``CognitiveRetriever`` from a pool directory and YAML config dict."""
    features = retrieval_config.get("features", {})
    weights = retrieval_config.get("weights", {})
    retrieval = retrieval_config.get("retrieval", {})
    threshold = retrieval_config.get("threshold", {})

    return CognitiveRetriever(
        str(pool_dir),
        use_cognitive_rerank=features.get("use_cognitive_rerank", True),
        use_llm_synergy=features.get("use_llm_synergy", True),
        alpha_semantic=weights.get("alpha_semantic", 0.35),
        alpha_cognitive=weights.get("alpha_cognitive", 0.65),
        alpha_dual_task=weights.get("alpha_dual_task", 0.70),
        top_n_candidates=retrieval.get("top_n_candidates", 20),
        top_k=retrieval.get("top_k", 3),
        min_memories=retrieval.get("min_memories", 1),
        score_threshold_floor=threshold.get("score_threshold_floor", 0.45),
        score_threshold_std=threshold.get("score_threshold_std", 0.5),
    )


def _format_insight_brief(insight_type: str, content: dict) -> str:
    """Extract a one-line summary from an insight memory for prompt inclusion."""
    if insight_type == "causal":
        name = content.get("principle_name", "")
        stmt = content.get("principle_statement", "")
        if name and stmt:
            return f"{name}: {stmt[:200]}"
        return stmt[:200] if stmt else str(content)[:200]
    if insight_type == "contrastive":
        aps = content.get("anti_patterns", [])
        if aps:
            ap = aps[0]
            return f"Avoid: {ap.get('name', '')} -- {ap.get('description', '')[:150]}"
        return content.get("positive_pattern", "")[:200]
    if insight_type == "strategic":
        name = content.get("methodology_name", "")
        idea = content.get("core_idea", "")
        if name and idea:
            return f"{name}: {idea[:200]}"
        return idea[:200] if idea else ""
    if insight_type == "environment":
        advices = content.get("tool_agnostic_advice", [])
        if advices:
            return advices[0][:250]
        return content.get("language_transfer", "")[:200]
    return str(content)[:200]


def _build_memory_context(memories: list[dict]) -> str:
    """Format retrieved memories into a Markdown block for the agent instruction."""
    if not memories:
        return ""

    parts = ["## Relevant Past Experiences\n"]
    parts.append(
        "The following experiences from solving similar tasks may be helpful:\n"
    )

    for i, mem in enumerate(memories, 1):
        mem_type = mem.get("type", "unknown")
        mem_level = mem.get("level", "unknown")
        task_name = mem.get("task_name", "unknown")
        score = mem.get("combined_score", 0)
        memory_content = mem.get("memory", {})

        parts.append(
            f"### Experience {i} [{mem_type}/{mem_level}] (relevance: {score:.2f})"
        )
        parts.append(f"**Source task:** {task_name}")

        if isinstance(memory_content, dict):
            for key, value in memory_content.items():
                if value and key not in ("key_embedding", "embedding"):
                    val_str = str(value)
                    if len(val_str) > 500:
                        val_str = val_str[:500] + "..."
                    parts.append(f"- **{key}:** {val_str}")
        elif isinstance(memory_content, str):
            if len(memory_content) > 800:
                memory_content = memory_content[:800] + "..."
            parts.append(str(memory_content))

        # Render attached insight-layer memories
        for ins in mem.get("_linked_insights", []):
            ins_type = ins.get("type", "?")
            ins_content = ins.get("memory", {})
            parts.append(
                f"  > **Insight [{ins_type}]:** "
                f"{_format_insight_brief(ins_type, ins_content)}"
            )

        parts.append("")

    parts.append("---\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Memory extraction (post-task)
# ---------------------------------------------------------------------------

def _get_pass_fail_from_trajectory(traj_path: Path) -> bool:
    """Determine pass/fail from a mini-swe-agent trajectory JSON.

    Returns ``True`` if the agent submitted (``exit_status == "Submitted"``).
    """
    if not traj_path.exists():
        return False
    try:
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        exit_status = traj.get("info", {}).get("exit_status", "")
        return exit_status == "Submitted"
    except Exception:
        return False


def _extract_memories_from_trajectory(
    traj_path: Path,
    pool_dir: Path,
    task_name: str,
    only_passed: bool,
    derive_traditional: bool = True,
) -> bool:
    """Extract cognitive + traditional memories from a mini-swe-agent trajectory.

    Args:
        traj_path: Path to the ``{instance_id}.traj.json`` file.
        pool_dir: Directory where ``.pkl`` files are written.
        task_name: Instance ID used as the task label in memory entries.
        only_passed: If ``True``, skip extraction for failed tasks.
        derive_traditional: If ``False``, skip traditional memory derivation
            (only extract 4x4 cognitive matrix).

    Returns:
        ``True`` if memories were extracted, ``False`` otherwise.
    """
    if not traj_path.exists():
        logger.warning(f"No trajectory file at {traj_path}")
        return False

    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    messages = traj.get("messages", [])
    if not messages:
        logger.warning(f"Empty message list in {traj_path}")
        return False

    trajectory, commands, task = convert_trajectory(messages)

    passed = traj.get("info", {}).get("exit_status", "") == "Submitted"

    if only_passed and not passed:
        logger.info(
            f"  SKIP extraction: task not submitted "
            f"(exit_status={traj.get('info', {}).get('exit_status', '?')}), "
            f"only_passed=True"
        )
        return False

    # log_dir = pool_dir/task_name → pkl files go to pool_dir/*.pkl
    # (extraction functions use log_dir.rsplit("/", 1)[0] as the parent dir)
    log_dir = str(pool_dir / task_name).replace("\\", "/")

    try:
        extract_rawtraj_memory(
            passed, trajectory, log_dir,
            task_name, task, commands, "swebench-verified",
        )
        extract_cognitive_memories(
            passed, trajectory, log_dir,
            task_name, task, commands, "swebench-verified",
            derive_traditional=derive_traditional,
        )
        return True
    except Exception as e:
        logger.error(f"  ERROR extracting memories: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("verified", "--subset", help="SWEBench subset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:500')", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory for preds.json + log", rich_help_panel="Basic"),
    pipeline: str = typer.Option("", "-p", "--pipeline", help="Pipeline config name or path (e.g. 'e31_seq_full' or path to YAML). Searches harbor/configs/experiments/", rich_help_panel="Pipeline"),
    pool_dir: str = typer.Option("", "--pool-dir", help="Memory pool directory (read *.pkl + write new memories). Overrides pipeline setting.", rich_help_panel="Memory"),
    jobs_dir: str = typer.Option("", "--jobs-dir", help="Directory to save per-instance trajectories. Overrides pipeline setting.", rich_help_panel="Memory"),
    retrieval_config_path: str = typer.Option("", "--retrieval-config", help="Path to cognitive_retrieval.yaml. Overrides pipeline retrieval_config.", rich_help_panel="Memory"),
    llm_api_key: str | None = typer.Option(None, "--llm-api-key", help="MTL LLM API key (for extraction/retrieval/rerank)", rich_help_panel="Memory"),
    llm_base_url: str | None = typer.Option(None, "--llm-base-url", help="MTL LLM base URL", rich_help_panel="Memory"),
    llm_model: str | None = typer.Option(None, "--llm-model", help="MTL LLM model name", rich_help_panel="Memory"),
    only_passed: bool = typer.Option(True, "--only-passed/--all-memories", help="Only extract memories from submitted tasks", rich_help_panel="Memory"),
    derive_traditional_memory: bool | None = typer.Option(None, "--derive-traditional/--no-derive-traditional", help="Derive traditional memories from cognitive matrix (default from pipeline config, or True)", rich_help_panel="Memory"),
    model: str | None = typer.Option(None, "-m", "--model", help="Agent model", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Agent config files (key=value or file paths)", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type (docker, local, etc.)", rich_help_panel="Advanced"),
    resume_from: int = typer.Option(0, "--resume-from", help="Resume from 0-based task index", rich_help_panel="Basic"),
) -> None:
    # fmt: on
    if not _MTL_AVAILABLE:
        logger.error(
            "MTL (Memory Transfer Learning) package is required for swebench-mtl.\n"
            "Install it with:  pip install -e /path/to/mtl\n"
            "Or set PYTHONPATH to include the mtl source tree."
        )
        raise typer.Exit(1)

    # --- Load pipeline config (sets defaults; CLI flags override) ---
    pipeline_config: dict = {}
    if pipeline:
        pipeline_path = _find_pipeline_config(pipeline)
        if pipeline_path is None:
            logger.error(
                f"Pipeline config not found: {pipeline}\n"
                f"Searched: {', '.join(str(d) for d in _PIPELINE_SEARCH_DIRS)}\n"
                f"Available pipelines: e1_full, e2_no_rerank, e3_no_synergy, "
                f"e23_pure_embedding, e24_no_memory, "
                f"e31_seq_full, e32_seq_embedding, e33_seq_no_memory, e34_seq_mtl_original"
            )
            raise typer.Exit(1)
        pipeline_config = _load_pipeline_yaml(pipeline_path)
        logger.info(f"Loaded pipeline: {pipeline_config.get('name', pipeline_path.stem)}")
        logger.info(f"  {pipeline_config.get('description', '')}")

    # Apply pipeline defaults (CLI flags take precedence if explicitly provided)
    pipeline_retrieval = pipeline_config.get("retrieval_config", {})
    pipeline_llm = pipeline_config.get("llm", {})

    # --- Validate and create directories ---
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "minisweagent_mtl.log")
    logger.info(f"Results will be saved to {output_path.resolve()}")

    # jobs_dir: CLI > pipeline > output/trajectories
    _jobs_dir = jobs_dir or pipeline_config.get("jobs_dir", "")
    jobs_path = Path(_jobs_dir) if _jobs_dir else output_path / "trajectories"
    jobs_path.mkdir(parents=True, exist_ok=True)

    # pool_dir: CLI > pipeline > output/memory_pool
    _pool_dir = pool_dir or pipeline_config.get("pool_dir", "")
    use_memory = True
    if pipeline_config and _pool_dir == "":
        # pipeline explicitly disabled memory (e.g., e24_no_memory, e33_seq_no_memory)
        use_memory = False
    pool_path = Path(_pool_dir) if _pool_dir else output_path / "memory_pool"
    if use_memory:
        pool_path.mkdir(parents=True, exist_ok=True)

    # only_passed: CLI flag takes precedence if it differs from default,
    # otherwise use pipeline setting
    _only_passed = only_passed
    if pipeline_config and only_passed == True:  # default value: check pipeline
        _only_passed = pipeline_config.get("only_passed", True)

    # derive_traditional_memory: CLI > pipeline config > default (True)
    _derive_traditional = derive_traditional_memory
    if _derive_traditional is None:
        _derive_traditional = pipeline_config.get("derive_traditional_memory", True)

    # --- Configure MTL LLM (CLI > env var > pipeline) ---
    # Env vars: API_KEY, BASE_URL, MODEL (or MTL_LLM_* / DEEPSEEK_API_KEY)
    _llm_api_key = (
        llm_api_key
        or os.getenv("MTL_LLM_API_KEY") or os.getenv("API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        or pipeline_llm.get("api_key") or None
    )
    _llm_base_url = (
        llm_base_url
        or os.getenv("MTL_LLM_BASE_URL") or os.getenv("BASE_URL")
        or pipeline_llm.get("base_url") or None
    )
    _llm_model = (
        llm_model
        or os.getenv("MTL_LLM_MODEL") or os.getenv("MODEL")
        or pipeline_llm.get("model") or None
    )
    mtl_configure(api_key=_llm_api_key, base_url=_llm_base_url, model=_llm_model)

    # --- Load retrieval config (CLI --retrieval-config > pipeline retrieval_config) ---
    retrieval_config = _load_retrieval_config(
        Path(retrieval_config_path) if retrieval_config_path else None
    )
    if not retrieval_config and pipeline_retrieval:
        retrieval_config = pipeline_retrieval
    elif retrieval_config and pipeline_retrieval:
        # CLI --retrieval-config wins entirely if provided (not merged)
        pass

    # --- Agent model: CLI -m > MODEL env > pipeline model ---
    _env_model = os.getenv("MODEL")
    if _env_model and "/" not in _env_model:
        _env_model = f"deepseek/{_env_model}"
    _agent_model = model or _env_model or pipeline_config.get("model") or None
    logger.info(f"Agent model: {_agent_model}")

    # --- Load dataset ---
    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))
    instances = filter_instances(
        instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle
    )

    # Skip already-completed instances unless --redo-existing
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = set(
            json.loads((output_path / "preds.json").read_text()).keys()
        )
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [
            i for i in instances if i["instance_id"] not in existing_instances
        ]

    total = len(instances)
    if total == 0:
        logger.info("No instances to run. Exiting.")
        return

    logger.info(f"Running on {total} instances sequentially...")
    if use_memory:
        logger.info(f"Memory pool: {pool_path.resolve()}")
    else:
        logger.info(f"Memory: DISABLED (no-memory pipeline)")
    logger.info(f"Trajectories: {jobs_path.resolve()}")
    if pipeline_config:
        logger.info(f"Pipeline: {pipeline_config.get('name', pipeline)}")
        logger.info(f"Retrieval: rerank={retrieval_config.get('features',{}).get('use_cognitive_rerank',True)}, "
                    f"synergy={retrieval_config.get('features',{}).get('use_llm_synergy',True)})")

    # --- Build agent config (mirrors swebench.py) ---
    from minisweagent.config import get_config_from_spec  # local import

    configs = [get_config_from_spec(spec) for spec in config_spec]

    # Collect env-var-driven agent model kwargs
    _agent_model_kwargs = {}
    if _llm_api_key:
        _agent_model_kwargs["api_key"] = _llm_api_key
    if _llm_base_url:
        _agent_model_kwargs["api_base"] = _llm_base_url

    configs.append(
        {
            "environment": {"environment_class": environment_class or UNSET},
            "model": {
                "model_name": _agent_model or UNSET,
                "model_class": model_class or UNSET,
                "model_kwargs": _agent_model_kwargs,
            },
            "agent": {
                "agent_class": "mtl",  # Use MTLAgent so {{memory_context}} is rendered
            },
        }
    )
    config = recursive_merge(*configs)

    # When using local environment, SWE-bench's /testbed doesn't exist.
    # Override cwd to the current directory so commands don't all fail.
    if config.get("environment", {}).get("environment_class") == "local":
        local_cwd = config.get("environment", {}).get("cwd", "")
        if not local_cwd or not Path(local_cwd).exists():
            config.setdefault("environment", {})["cwd"] = os.getcwd()
            logger.info(f"Local environment: overriding cwd to {os.getcwd()}")

    # Inject {{memory_context}} into instance_template if not already present
    instance_template = config.setdefault("agent", {}).get("instance_template", "")
    if "memory_context" not in instance_template:
        config["agent"]["instance_template"] = (
            "{{memory_context}}\n" + instance_template
        )
        logger.info("Prepended {{memory_context}} to instance_template.")

    # --- Initialize retriever ---
    retriever = None
    if use_memory:
        has_pool = bool(list(pool_path.glob("*.pkl")))
        if has_pool:
            logger.info(
                f"Initializing retriever from {len(list(pool_path.glob('*.pkl')))} "
                f"pkl file(s)..."
            )
            retriever = _init_retriever(pool_path, retrieval_config)

    # --- Main sequential loop ---
    passed_count = 0
    extracted_count = 0
    start_time = time.time()
    results = []

    from minisweagent.agents import get_agent
    from minisweagent.models import get_model

    for idx, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        task_text = instance["problem_statement"]

        if idx < resume_from:
            logger.info(
                f"[{idx + 1}/{total}] SKIP {instance_id} "
                f"(resume_from={resume_from})"
            )
            continue

        round_start = time.time()

        logger.info(f"\n{'=' * 60}")
        pool_file_count = len(list(pool_path.glob("*.pkl"))) if use_memory else 0
        logger.info(
            f"[Round {idx + 1}/{total}] {instance_id}  "
            f"(pool: {pool_file_count} pkl files)"
        )

        # ---- Step 1: Retrieve memory context ----
        memory_context = ""
        if retriever is not None:
            try:
                retriever.reload()  # pick up new memories from previous rounds
                memories = retriever.retrieve(task_text)
                if memories:
                    memory_context = _build_memory_context(memories)
                    logger.info(
                        f"  Retrieved {len(memories)} memories "
                        f"({len(memory_context)} chars)"
                    )
                    for m in memories:
                        logger.debug(
                            f"    [{m.get('type', '?')}/{m.get('level', '?')}] "
                            f"score={m.get('combined_score', 0):.3f}  "
                            f"from {m.get('task_name', '?')}"
                        )
                else:
                    logger.info("  No relevant memories found.")
            except Exception as e:
                logger.warning(f"  Retrieval failed: {e}, proceeding without memory")

        # ---- Step 2: Run agent ----
        instance_dir = jobs_path / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        traj_path = instance_dir / f"{instance_id}.traj.json"

        # Remove stale files for this instance
        remove_from_preds_file(output_path / "preds.json", instance_id)
        traj_path.unlink(missing_ok=True)

        agent = None
        exit_status = "unknown"
        submission = ""
        extra_info = {}

        logger.info(f"  [1/3] Running agent (model={config['model'].get('model_name', '?')})...")

        # ---- Auto-checkout correct commit for local environments ----
        if config.get("environment", {}).get("environment_class") == "local":
            base_commit = instance.get("environment_setup_commit", instance.get("base_commit", ""))
            local_cwd = config.get("environment", {}).get("cwd", "")
            if base_commit and local_cwd:
                import subprocess
                repo = Path(local_cwd)
                if repo.is_dir():
                    logger.info(f"  Checking out base commit: {base_commit[:8]}...")
                    _out = subprocess.run(
                        ["git", "-C", str(repo), "checkout", "-f", base_commit],
                        capture_output=True, text=True
                    )
                    if _out.returncode != 0:
                        logger.warning(f"  git checkout warning: {_out.stderr.strip()}")
                    subprocess.run(
                        ["git", "-C", str(repo), "clean", "-fdx"],
                        capture_output=True, text=True
                    )
                    logger.info(f"  Repo at {local_cwd} → commit {base_commit[:8]}")

        try:
            env = get_sb_environment(config, instance)
            agent_model = get_model(config=config.get("model", {}))
            agent = get_agent(
                agent_model,
                env,
                config.get("agent", {}),
                default_type="mtl",
            )
            info = agent.run(task_text, memory_context=memory_context)
            exit_status = info.get("exit_status", "unknown")
            submission = info.get("submission", "")
        except Exception as e:
            logger.error(f"Error processing {instance_id}: {e}", exc_info=True)
            exit_status = type(e).__name__
            submission = ""
            extra_info = {
                "traceback": traceback.format_exc(),
                "exception_str": str(e),
            }
        finally:
            if agent is not None:
                agent.save(
                    traj_path,
                    {
                        "info": {
                            "exit_status": exit_status,
                            "submission": submission,
                            **extra_info,
                        },
                        "instance_id": instance_id,
                    },
                )
                logger.info(f"  Saved trajectory to {traj_path}")
            update_preds_file(
                output_path / "preds.json",
                instance_id,
                config.get("model", {}).get("model_name", "unknown"),
                submission,
            )

        is_pass = exit_status == "Submitted"
        if is_pass:
            passed_count += 1

        round_elapsed = time.time() - round_start
        logger.info(
            f"  Agent result: {'PASS' if is_pass else 'FAIL'} "
            f"(exit_status={exit_status}, elapsed={round_elapsed:.0f}s)"
        )

        # ---- Step 3: Extract memories from trajectory ----
        if use_memory and traj_path.exists():
            logger.info(f"  [2/3] Extracting memories...")
            extracted = _extract_memories_from_trajectory(
                traj_path, pool_path, instance_id, _only_passed,
                derive_traditional=_derive_traditional,
            )
            if extracted:
                extracted_count += 1
                # Initialize or refresh retriever for next round
                if retriever is None:
                    retriever = _init_retriever(pool_path, retrieval_config)
                else:
                    retriever.reload()
        else:
            extracted = False

        # ---- Record results ----
        cum_pass_rate = passed_count / (idx + 1) * 100
        mem_count = len(list(pool_path.glob("*.pkl"))) if use_memory else 0
        logger.info(
            f"  [3/3] Cumulative: {passed_count}/{idx + 1} ({cum_pass_rate:.1f}%) "
            f"| Pool: {mem_count} pkl | Extracted: {extracted_count}"
        )

        results.append(
            {
                "round": idx + 1,
                "task": instance_id,
                "passed": is_pass,
                "memories_extracted": extracted,
                "memory_context_chars": len(memory_context),
                "elapsed_sec": round(round_elapsed, 1),
            }
        )

    # --- Summary ---
    total_elapsed = time.time() - start_time
    final_pass_rate = passed_count / total * 100 if total > 0 else 0

    summary = {
        "name": pipeline_config.get("name", "swebench-mtl"),
        "pipeline": pipeline or None,
        "total_tasks": total,
        "passed": passed_count,
        "pass_rate": round(final_pass_rate, 1),
        "extracted_tasks": extracted_count,
        "total_elapsed_sec": round(total_elapsed, 0),
        "memory_enabled": use_memory,
        "pool_dir": str(pool_path.resolve()) if use_memory else None,
        "jobs_dir": str(jobs_path.resolve()),
        "retrieval_config": {
            "features": retrieval_config.get("features", {}),
            "weights": retrieval_config.get("weights", {}),
        },
        "results": results,
    }

    summary_path = output_path / "sequential_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    logger.info(f"\n{'=' * 70}")
    logger.info("Sequential MTL experiment complete.")
    logger.info(f"  Final pass rate: {passed_count}/{total} ({final_pass_rate:.1f}%)")
    logger.info(f"  Memories extracted: {extracted_count} tasks")
    logger.info(f"  Total time: {total_elapsed / 60:.1f} min")
    logger.info(f"  Summary saved to: {summary_path}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    app()
