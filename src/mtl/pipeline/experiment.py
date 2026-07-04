"""Full experiment pipeline orchestration.

Runs a complete MTL experiment:
  Phase 1: Run agents on source tasks (via harbor)
  Phase 2: Extract memories from trajectories
  Phase 3: Run agents on target tasks with memory retrieval
  Phase 4: Evaluate and compare results
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for a full MTL experiment."""

    # Experiment identity
    name: str = "mtl-experiment"
    description: str = ""

    # Task configuration
    tasks_dir: str = "harbor-tasks/swebench-verified"
    source_start: int = 0
    source_end: int = 100  # Source tasks (for memory extraction)
    target_start: int = 100
    target_end: int = 200  # Target tasks (for evaluation)

    # Agent configuration
    agent: str = "mini-swe-agent"
    model: str = "deepseek/deepseek-chat"

    # LLM configuration (for retrieval, rerank, synergy, extraction)
    llm: dict = field(default_factory=lambda: {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    })

    # Memory configuration
    memory_dir: str = "memories/swebench-verified"
    only_passed: bool = True  # Only use passed task memories

    # Retrieval configuration
    retrieval_config: dict = field(default_factory=lambda: {
        "features": {
            "use_cognitive_rerank": True,
            "use_llm_synergy": True,
        },
        "weights": {
            "alpha_dual_task": 0.70,
            "alpha_semantic": 0.35,
            "alpha_cognitive": 0.65,
        },
        "retrieval": {
            "top_n_candidates": 20,
            "top_k": 3,
            "min_memories": 1,
        },
        "threshold": {
            "score_threshold_floor": 0.45,
            "score_threshold_std": 0.5,
        },
    })

    # Execution
    jobs_dir: str = "jobs"
    concurrent: int = 2
    force_build: bool = False
    dry_run: bool = False


class ExperimentPipeline:
    """Orchestrate a full MTL experiment end-to-end."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._results: dict = {}

    def run(self) -> dict:
        """Run the complete experiment pipeline."""
        from mtl.llm import configure

        # Configure LLM first (before any extraction/retrieval imports)
        llm = self.config.llm
        configure(
            api_key=llm.get("api_key") or None,
            base_url=llm.get("base_url") or None,
            model=llm.get("model") or None,
        )

        print(f"\n{'='*70}")
        print(f"MTL Experiment: {self.config.name}")
        print(f"{'='*70}")
        print(f"Source tasks: [{self.config.source_start}:{self.config.source_end}]")
        print(f"Target tasks: [{self.config.target_start}:{self.config.target_end}]")
        print(f"Agent: {self.config.agent} | Model: {self.config.model}")
        print(f"LLM: {llm.get('model')} @ {llm.get('base_url')}")
        print(f"Memory: {self.config.memory_dir} | Only passed: {self.config.only_passed}")
        print()

        # Phase 1: Run source tasks (without memory)
        print("=" * 50)
        print("PHASE 1: Running source tasks (without memory)")
        print("=" * 50)
        if not self.config.dry_run:
            self._run_tasks(
                start=self.config.source_start,
                end=self.config.source_end,
                memory_path=None,
            )

        # Phase 2: Extract memories
        print("\n" + "=" * 50)
        print("PHASE 2: Extracting memories")
        print("=" * 50)
        if not self.config.dry_run:
            self._extract_memories()

        # Phase 3: Run target tasks (with memory)
        print("\n" + "=" * 50)
        print("PHASE 3: Running target tasks (with memory)")
        print("=" * 50)
        if not self.config.dry_run:
            self._run_tasks(
                start=self.config.target_start,
                end=self.config.target_end,
                memory_path=self.config.memory_dir,
            )

        # Phase 4: Evaluate
        print("\n" + "=" * 50)
        print("PHASE 4: Evaluation")
        print("=" * 50)
        if not self.config.dry_run:
            self._evaluate()

        return self._results

    def _run_tasks(self, start: int, end: int, memory_path: str | None):
        """Run harbor tasks via the CLI."""
        cmd = [
            sys.executable, "-m", "harbor.cli.main", "jobs", "start",
            "-p", self.config.tasks_dir,
            "-a", self.config.agent,
            "-m", self.config.model,
            "--jobs-dir", self.config.jobs_dir,
            "-n", str(self.config.concurrent),
        ]

        if memory_path:
            cmd.extend(["--memory-path", memory_path])

        if self.config.force_build:
            cmd.append("--force-build")

        # Collect task names
        tasks_dir = Path(self.config.tasks_dir)
        task_names = sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())
        selected = task_names[start:end]

        for name in selected:
            cmd.extend(["-t", name])

        print(f"Running {len(selected)} tasks ({selected[0]} ... {selected[-1]})...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: harbor returned exit code {result.returncode}")

    def _extract_memories(self):
        """Extract memories from source task trajectories."""
        from mtl.extraction import MemoryExtractor

        llm = self.config.llm
        extractor = MemoryExtractor(
            jobs_dir=self.config.jobs_dir,
            memory_dir=self.config.memory_dir,
            only_passed=self.config.only_passed,
            llm_api_key=llm.get("api_key") or None,
            llm_base_url=llm.get("base_url") or None,
            llm_model=llm.get("model") or None,
        )

        n_tasks = self.config.source_end - self.config.source_start
        success = extractor.extract_batch(
            start=self.config.source_start,
            limit=n_tasks,
        )
        self._results["extracted_tasks"] = len(success)

    def _evaluate(self):
        """Compute pass rates from job results."""
        jobs_dir = Path(self.config.jobs_dir)
        total = 0
        passed = 0

        for d in jobs_dir.iterdir():
            if not d.is_dir():
                continue
            result_path = d / "result.json"
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    reward = result.get("verifier_result", {}).get("rewards", {}).get("reward", 0.0)
                    total += 1
                    if reward >= 1.0:
                        passed += 1
                except Exception:
                    pass

        pass_rate = passed / total * 100 if total > 0 else 0
        self._results["total_tasks"] = total
        self._results["passed_tasks"] = passed
        self._results["pass_rate"] = pass_rate

        print(f"\nResults:")
        print(f"  Total tasks evaluated: {total}")
        print(f"  Passed: {passed}")
        print(f"  Pass rate: {pass_rate:.1f}%")

    @property
    def results(self) -> dict:
        return self._results


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file.

    Searches: exact path → harbor/configs/experiments/ → configs/experiments/
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        # Try harbor/configs/experiments/
        alt = Path("harbor/configs/experiments") / config_path.name
        if alt.exists():
            config_path = alt
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ExperimentConfig(**data)
