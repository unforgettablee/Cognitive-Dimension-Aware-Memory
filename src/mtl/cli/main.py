"""MTL CLI — Memory Transfer Learning Pipeline command-line interface.

Usage:
    mtl extract --jobs-dir jobs/2026-06-01/xxx --memory-dir memories/swebench-verified
    mtl retrieve --memory-dir memories/swebench-verified --query "Fix KeyError..."
    mtl run --start 100 --end 200 --memory-path memories/swebench-verified
    mtl experiment --config harbor/configs/experiments/e1_full.yaml
    mtl info --memory-dir memories/swebench-verified
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="mtl",
    help="Memory Transfer Learning (MTL) Pipeline CLI",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
@app.command()
def extract(
    jobs_dir: str = typer.Option(..., "--jobs-dir", "-j", help="Directory containing harbor job output subdirectories"),
    memory_dir: str = typer.Option("memories/swebench-verified", "--memory-dir", "-m", help="Directory to store memory files"),
    benchmark: str = typer.Option("swebench-verified", "--benchmark", "-b", help="Benchmark name"),
    start: int = typer.Option(0, "--start", "-s", help="Start task index (0-based)"),
    limit: int = typer.Option(100, "--limit", "-l", help="Max number of tasks to process"),
    only_passed: bool = typer.Option(False, "--only-passed/--all", help="Only extract from passed tasks"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="LLM API key (default: MTL_LLM_API_KEY or DEEPSEEK_API_KEY env var)"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="LLM API base URL (default: https://api.deepseek.com)"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM model name (default: deepseek-chat)"),
):
    """Extract memories from harbor job trajectories."""
    from mtl.llm import configure
    from mtl.extraction import MemoryExtractor

    configure(api_key=api_key, base_url=api_base, model=llm_model)

    jobs_path = Path(jobs_dir)
    if not jobs_path.exists():
        typer.echo(f"ERROR: jobs-dir not found: {jobs_dir}", err=True)
        raise typer.Exit(1)

    extractor = MemoryExtractor(
        jobs_dir=jobs_dir,
        memory_dir=memory_dir,
        benchmark=benchmark,
        only_passed=only_passed,
    )
    extractor.extract_batch(start=start, limit=limit)


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------
@app.command()
def retrieve(
    memory_dir: str = typer.Option("memories/swebench-verified", "--memory-dir", "-m", help="Directory containing *.pkl memory files"),
    query: str = typer.Option(None, "--query", "-q", help="Query text (if omitted, reads from stdin or interactive)"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of memories to retrieve"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to retrieval config YAML"),
    output_json: bool = typer.Option(False, "--json", help="Output results as JSON"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="LLM API key (default: MTL_LLM_API_KEY or DEEPSEEK_API_KEY env var)"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="LLM API base URL (default: https://api.deepseek.com)"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM model name (default: deepseek-chat)"),
):
    """Retrieve memories for a given query."""
    from mtl.llm import configure
    from mtl.retrieval import CognitiveRetriever

    configure(api_key=api_key, base_url=api_base, model=llm_model)

    retriever_config = _load_retrieval_config(config)

    retriever = CognitiveRetriever(
        memory_dir,
        **retriever_config,
    )

    if not query:
        if not sys.stdin.isatty():
            query = sys.stdin.read().strip()
        else:
            query = typer.prompt("Enter query text")

    if not query:
        typer.echo("ERROR: empty query", err=True)
        raise typer.Exit(1)

    results = retriever.retrieve(query, top_k=top_k)

    if output_json:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo(f"\nQuery: {query[:120]}...")
        typer.echo(f"Found {len(results)} memories:\n")
        for i, r in enumerate(results):
            typer.echo(f"  [{i+1}] [{r.get('level','?')}/{r.get('type','?')}] "
                       f"score={r.get('combined_score',0):.3f} "
                       f"task={r.get('task_name','?')}")
            n_insights = len(r.get("_linked_insights", []))
            if n_insights:
                typer.echo(f"       +{n_insights} linked insights")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
@app.command()
def run(
    tasks_dir: str = typer.Option("harbor-tasks/swebench-verified", "--tasks-dir", "-p", help="Directory containing task subdirectories"),
    start: int = typer.Option(0, "--start", "-s", help="Start task index (0-based, inclusive)"),
    end: Optional[int] = typer.Option(None, "--end", "-e", help="End task index (exclusive)"),
    agent: str = typer.Option("mini-swe-agent", "--agent", "-a", help="Agent name"),
    model: str = typer.Option("deepseek/deepseek-chat", "--model", "-m", help="Model name"),
    jobs_dir: str = typer.Option("jobs", "--jobs-dir", help="Jobs output directory"),
    memory_path: Optional[str] = typer.Option(None, "--memory-path", help="Path to memory files (omit for no memory)"),
    concurrent: int = typer.Option(2, "-n", "--concurrent", help="Number of concurrent jobs"),
    force_build: bool = typer.Option(False, "--force-build", help="Force rebuild of Docker image"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command without executing"),
):
    """Run harbor tasks (with optional memory retrieval)."""
    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        typer.echo(f"ERROR: tasks-dir not found: {tasks_dir}", err=True)
        raise typer.Exit(1)

    all_tasks = sorted(d.name for d in tasks_path.iterdir() if d.is_dir())
    selected = all_tasks[start:end]

    typer.echo(f"Total tasks found: {len(all_tasks)}")
    typer.echo(f"Selected range: [{start}:{end or len(all_tasks)}] = {len(selected)} tasks")

    if not selected:
        typer.echo("(empty range, nothing to run)")
        raise typer.Exit(0)

    typer.echo(f"  First: {selected[0]}")
    typer.echo(f"  Last:  {selected[-1]}")

    cmd = [
        sys.executable, "-m", "harbor.cli.main", "jobs", "start",
        "-p", str(tasks_dir),
        "-a", agent,
        "-m", model,
        "--jobs-dir", jobs_dir,
        "-n", str(concurrent),
    ]

    if memory_path:
        cmd.extend(["--memory-path", memory_path])
    if force_build:
        cmd.append("--force-build")
    for name in selected:
        cmd.extend(["-t", name])

    if dry_run:
        typer.echo(f"\nWould execute: {' '.join(cmd)}")
        return

    import subprocess
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------
@app.command()
def experiment(
    name: str = typer.Option("mtl-experiment", "--name", help="Experiment name"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to experiment config YAML"),
    source_start: int = typer.Option(0, "--source-start", help="Source task start index"),
    source_end: int = typer.Option(100, "--source-end", help="Source task end index"),
    target_start: int = typer.Option(100, "--target-start", help="Target task start index"),
    target_end: int = typer.Option(200, "--target-end", help="Target task end index"),
    tasks_dir: str = typer.Option("harbor-tasks/swebench-verified", "--tasks-dir", "-p"),
    agent: str = typer.Option("mini-swe-agent", "--agent", "-a"),
    model: str = typer.Option("deepseek/deepseek-chat", "--model", "-m"),
    jobs_dir: str = typer.Option("jobs", "--jobs-dir"),
    memory_dir: str = typer.Option("memories/swebench-verified", "--memory-dir", "-m"),
    only_passed: bool = typer.Option(True, "--only-passed/--all"),
    concurrent: int = typer.Option(2, "-n", "--concurrent"),
    force_build: bool = typer.Option(False, "--force-build"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    use_cognitive_rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    use_llm_synergy: bool = typer.Option(True, "--synergy/--no-synergy"),
    alpha_dual_task: float = typer.Option(0.70, "--alpha-dual", help="Task vs cognitive embedding weight"),
    top_k: int = typer.Option(3, "--top-k", help="Final memories to inject"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="LLM API key (default: MTL_LLM_API_KEY or DEEPSEEK_API_KEY env var)"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="LLM API base URL (default: https://api.deepseek.com)"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM model name (default: deepseek-chat)"),
):
    """Run a complete MTL experiment end-to-end."""
    from mtl.pipeline import ExperimentPipeline, ExperimentConfig

    if config_path:
        from mtl.pipeline.experiment import load_experiment_config
        config = load_experiment_config(config_path)
        # CLI flags override config file values
        if api_key:
            config.llm["api_key"] = api_key
        if api_base:
            config.llm["base_url"] = api_base
        if llm_model:
            config.llm["model"] = llm_model
    else:
        config = ExperimentConfig(
            name=name,
            tasks_dir=tasks_dir,
            source_start=source_start,
            source_end=source_end,
            target_start=target_start,
            target_end=target_end,
            agent=agent,
            model=model,
            llm={
                "api_key": api_key or "",
                "base_url": api_base or "https://api.deepseek.com",
                "model": llm_model or "deepseek-chat",
            },
            jobs_dir=jobs_dir,
            memory_dir=memory_dir,
            only_passed=only_passed,
            concurrent=concurrent,
            force_build=force_build,
            dry_run=dry_run,
            retrieval_config={
                "features": {
                    "use_cognitive_rerank": use_cognitive_rerank,
                    "use_llm_synergy": use_llm_synergy,
                },
                "weights": {
                    "alpha_dual_task": alpha_dual_task,
                    "alpha_semantic": 0.35,
                    "alpha_cognitive": 0.65,
                },
                "retrieval": {
                    "top_n_candidates": 20,
                    "top_k": top_k,
                    "min_memories": 1,
                },
            },
        )

    pipeline = ExperimentPipeline(config)
    results = pipeline.run()

    typer.echo(f"\nDone. Pass rate: {results.get('pass_rate', 0):.1f}%")


# ---------------------------------------------------------------------------
# sequential
# ---------------------------------------------------------------------------
@app.command()
def sequential(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to sequential experiment config YAML"),
    name: str = typer.Option("sequential-experiment", "--name", help="Experiment name"),
    tasks_dir: str = typer.Option("harbor-tasks/swebench-verified", "--tasks-dir", "-p"),
    start: int = typer.Option(0, "--start", "-s", help="Start task index (0-based)"),
    end: int = typer.Option(500, "--end", "-e", help="End task index (exclusive)"),
    agent: str = typer.Option("mini-swe-agent", "--agent", "-a"),
    model: str = typer.Option("deepseek/deepseek-chat", "--model", "-m", help="Agent model (for harbor)"),
    jobs_dir: str = typer.Option("jobs/sequential", "--jobs-dir"),
    pool_dir: str = typer.Option("memories/sequential-pool", "--pool-dir", help="Memory pool directory (accumulates over rounds)"),
    only_passed: bool = typer.Option(True, "--only-passed/--all", help="Only use passed-task memories"),
    force_build: bool = typer.Option(False, "--force-build"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    resume_from: int = typer.Option(0, "--resume-from", help="Resume from round N (0-based task index)"),
    use_cognitive_rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    use_llm_synergy: bool = typer.Option(True, "--synergy/--no-synergy"),
    alpha_dual_task: float = typer.Option(0.70, "--alpha-dual"),
    top_k: int = typer.Option(3, "--top-k"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="LLM API key"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="LLM API base URL"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM model name for extraction/retrieval"),
):
    """Run sequential experiment: each task sees all previous tasks' memories.

    Round 1: task[0] no memory -> extract
    Round 2: task[1] memory=[task 0] -> extract
    Round 3: task[2] memory=[tasks 0,1] -> extract
    ...and so on through all tasks.
    """
    from mtl.pipeline.sequential import SequentialPipeline, SequentialConfig, load_sequential_config

    if config_path:
        config = load_sequential_config(config_path)
        if api_key:
            config.llm["api_key"] = api_key
        if api_base:
            config.llm["base_url"] = api_base
        if llm_model:
            config.llm["model"] = llm_model
    else:
        config = SequentialConfig(
            name=name,
            tasks_dir=tasks_dir,
            start_index=start,
            end_index=end,
            agent=agent,
            model=model,
            llm={
                "api_key": api_key or "",
                "base_url": api_base or "https://api.deepseek.com",
                "model": llm_model or "deepseek-chat",
            },
            pool_dir=pool_dir,
            jobs_dir=jobs_dir,
            only_passed=only_passed,
            force_build=force_build,
            dry_run=dry_run,
            resume_from=resume_from,
            retrieval_config={
                "features": {
                    "use_cognitive_rerank": use_cognitive_rerank,
                    "use_llm_synergy": use_llm_synergy,
                },
                "weights": {"alpha_dual_task": alpha_dual_task},
                "retrieval": {"top_k": top_k},
            },
        )

    pipeline = SequentialPipeline(config)
    summary = pipeline.run()

    if summary:
        typer.echo(f"\nFinal pass rate: {summary['pass_rate']:.1f}% "
                   f"({summary['passed']}/{summary['total_tasks']})")

# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
@app.command()
def info(
    memory_dir: str = typer.Option("memories/swebench-verified", "--memory-dir", "-m", help="Path to memory directory"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show information about a memory store."""
    from mtl.retrieval import CognitiveRetriever

    try:
        retriever = CognitiveRetriever(memory_dir, use_cognitive_rerank=False, use_llm_synergy=False)
    except FileNotFoundError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)

    if json_output:
        data = {
            "stats": retriever.stats,
            "breakdown": retriever.memory_type_breakdown(),
        }
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        typer.echo(retriever.describe())
        typer.echo(f"\nMemory type breakdown:")
        for key, count in sorted(retriever.memory_type_breakdown().items()):
            typer.echo(f"  {key}: {count}")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@app.command()
def show_config(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to retrieval config YAML"),
):
    """Show the current retrieval configuration."""
    config = _load_retrieval_config(config_path)
    typer.echo(json.dumps(config, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_retrieval_config(config_path: str | None = None) -> dict:
    """Load retrieval configuration from YAML with defaults."""
    import yaml

    # Default config
    config = {
        "alpha_semantic": 0.35,
        "alpha_cognitive": 0.65,
        "alpha_dual_task": 0.70,
        "top_n_candidates": 20,
        "top_k": 3,
        "use_cognitive_rerank": True,
        "use_llm_synergy": True,
        "score_threshold_floor": 0.45,
        "score_threshold_std": 0.5,
        "min_memories": 1,
    }

    # Try loading from file
    resolved_path = config_path or os.environ.get("COGNITIVE_RETRIEVAL_CONFIG")
    if not resolved_path:
        # Try default locations
        candidates = [
            "harbor/config/cognitive_retrieval.yaml",
            "harbor/configs/cognitive_retrieval.yaml",
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if resolved_path and os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        features = data.get("features", {})
        weights = data.get("weights", {})
        retrieval = data.get("retrieval", {})
        threshold = data.get("threshold", {})

        config.update({
            "use_cognitive_rerank": features.get("use_cognitive_rerank", config["use_cognitive_rerank"]),
            "use_llm_synergy": features.get("use_llm_synergy", config["use_llm_synergy"]),
            "alpha_dual_task": weights.get("alpha_dual_task", config["alpha_dual_task"]),
            "alpha_semantic": weights.get("alpha_semantic", config["alpha_semantic"]),
            "alpha_cognitive": weights.get("alpha_cognitive", config["alpha_cognitive"]),
            "top_n_candidates": retrieval.get("top_n_candidates", config["top_n_candidates"]),
            "top_k": retrieval.get("top_k", config["top_k"]),
            "min_memories": retrieval.get("min_memories", config["min_memories"]),
            "score_threshold_floor": threshold.get("score_threshold_floor", config["score_threshold_floor"]),
            "score_threshold_std": threshold.get("score_threshold_std", config["score_threshold_std"]),
        })

    return config


def main():
    app()


if __name__ == "__main__":
    main()
