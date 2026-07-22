"""Sequential Experiment Pipeline.

Implements the sequential (incremental) memory transfer paradigm:
  Round 1: Task 1, no memory -> extract -> pool = [task 1]
  Round 2: Task 2, memory = [task 1] -> extract -> pool = [tasks 1, 2]
  Round 3: Task 3, memory = [tasks 1, 2] -> extract -> pool = [tasks 1, 2, 3]
  ...
  Round N: Task N, memory = [tasks 1..N-1]

Unlike Batch paradigm (fixed pool, parallel execution), Sequential measures the
marginal benefit of each additional task's memory.
"""
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SequentialConfig:
    """Configuration for a sequential MTL experiment."""

    name: str = "sequential-experiment"
    description: str = ""

    # Task configuration
    tasks_dir: str = "harbor-tasks/swebench-verified"
    start_index: int = 0
    end_index: int = 500  # Total tasks to run (exclusive)
    num_tasks: int | None = None  # Random-sample N tasks (None = use start_index/end_index slice)
    random_seed: int = 42  # Seed for task sampling and reproducibility

    # Agent configuration
    agent: str = "mini-swe-agent"
    model: str = "deepseek/deepseek-chat"

    # LLM configuration (for extraction + retrieval)
    llm: dict = field(default_factory=lambda: {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    })

    # Memory pool directory (accumulates over rounds)
    pool_dir: str = "memories/sequential-pool"

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

    # Memory extraction
    derive_traditional_memory: bool = True  # Derive traditional memories from cognitive matrix

    # Ablation: exclude specific cognitive dimensions (e7--e10) or abstraction levels (e11--e13)
    excluded_dimensions: list = field(default_factory=list)
    excluded_levels: list = field(default_factory=list)

    # Execution
    jobs_dir: str = "jobs/sequential"
    only_passed: bool = True  # Only use passed-task memories
    force_build: bool = False
    dry_run: bool = False
    resume_from: int = 0  # Resume from round N (0-based task index)


class SequentialPipeline:
    """Run tasks one-by-one with an incrementally growing memory pool.

    After each task completes, its trajectory is extracted and added to
    the shared memory pool. The next task sees all previous tasks' memories.
    """

    def __init__(self, config: SequentialConfig):
        self.config = config
        self.pool_dir = Path(config.pool_dir)
        self.results: list[dict] = []
        self._round = 0

    def run(self) -> dict:
        """Run the complete sequential experiment."""
        from mtl.llm import configure

        llm = self.config.llm
        configure(
            api_key=llm.get("api_key") or None,
            base_url=llm.get("base_url") or None,
            model=llm.get("model") or None,
        )

        use_memory = bool(self.config.pool_dir)
        if use_memory:
            self.pool_dir.mkdir(parents=True, exist_ok=True)

        tasks_dir = Path(self.config.tasks_dir)
        all_tasks = sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())
        selected = all_tasks[self.config.start_index:self.config.end_index]

        print(f"\n{'='*70}")
        print(f"Sequential Experiment: {self.config.name}")
        print(f"{'='*70}")
        print(f"Total tasks: {len(selected)}")
        print(f"Range: [{self.config.start_index}:{self.config.end_index}]")
        print(f"Agent: {self.config.agent} | Model: {self.config.model}")
        print(f"LLM: {llm.get('model')} @ {llm.get('base_url')}")
        print(f"Memory pool: {'ENABLED' if use_memory else 'DISABLED (no-memory baseline)'}")
        if use_memory:
            print(f"Pool dir: {self.pool_dir.resolve()}")
            print(f"Only passed memories: {self.config.only_passed}")
        print(f"Resume from: round {self.config.resume_from}")
        print()

        if self.config.dry_run:
            print("[dry-run] Would execute the above. Exiting.")
            return {}

        # Initialize tracking
        passed_count = 0
        total_count = 0
        start_time = time.time()

        for i, task_name in enumerate(selected):
            self._round = i + 1
            if i < self.config.resume_from:
                print(f"[{self._round}/{len(selected)}] SKIP {task_name} "
                      f"(resume_from={self.config.resume_from})")
                continue

            total_count += 1
            round_start = time.time()

            print(f"\n{'─'*60}")
            print(f"[Round {self._round}/{len(selected)}] {task_name}")

            # Check if memory pool has entries
            if use_memory:
                pkl_files = list(self.pool_dir.glob("*.pkl"))
                has_pool = len(pkl_files) > 0
                memory_path = str(self.pool_dir.resolve()) if has_pool else None
                if has_pool:
                    print(f"  Memory pool: {len(pkl_files)} pkl files "
                          f"(from {self._count_pool_tasks()} previous tasks)")
                else:
                    print(f"  Memory pool: empty (first task, no memory)")
            else:
                has_pool = False
                memory_path = None
                print(f"  Memory: DISABLED (no-memory baseline)")

            # Step 1: Run harbor for this single task
            print(f"  [1/3] Running agent...")
            job_dir = self._run_single_task(task_name, memory_path)

            # Step 2: Extract memories (only if memory is enabled)
            if use_memory:
                print(f"  [2/3] Extracting memories...")
                extracted = self._extract_from_task(task_name, job_dir)
            else:
                print(f"  [2/3] Skipping extraction (no-memory mode)")
                extracted = False

            # Step 3: Record result
            passed = self._get_judgement(job_dir)
            if passed:
                passed_count += 1

            elapsed = time.time() - round_start

            self.results.append({
                "round": self._round,
                "task": task_name,
                "passed": passed,
                "memories_extracted": extracted,
                "pool_tasks_before": self._count_pool_tasks(),
                "elapsed_sec": round(elapsed, 1),
            })

            status = "PASS" if passed else "FAIL"
            cum_pass_rate = passed_count / total_count * 100
            print(f"  [3/3] Result: {status} | "
                  f"Round time: {elapsed:.0f}s | "
                  f"Cumulative pass rate: {passed_count}/{total_count} "
                  f"({cum_pass_rate:.1f}%)")

        total_elapsed = time.time() - start_time
        final_pass_rate = passed_count / total_count * 100 if total_count > 0 else 0

        summary = {
            "name": self.config.name,
            "total_tasks": total_count,
            "passed": passed_count,
            "pass_rate": round(final_pass_rate, 1),
            "total_elapsed_sec": round(total_elapsed, 0),
            "results": self.results,
        }

        # Save summary
        summary_path = Path(self.config.jobs_dir) / "sequential_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        print(f"\n{'='*70}")
        print(f"Sequential experiment complete.")
        print(f"  Final pass rate: {passed_count}/{total_count} ({final_pass_rate:.1f}%)")
        print(f"  Total time: {total_elapsed/60:.1f} min")
        print(f"  Summary saved to: {summary_path}")
        print(f"{'='*70}")

        return summary

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _run_single_task(self, task_name: str, memory_path: str | None) -> Path:
        """Run harbor for a single task. Returns the job directory for that task."""
        job_name = f"seq-round-{self._round:04d}-{task_name[:30]}"
        cmd = [
            sys.executable, "-m", "harbor.cli.main", "jobs", "start",
            "-p", str(self.config.tasks_dir),
            "-a", self.config.agent,
            "-m", self.config.model,
            "--jobs-dir", str(self.config.jobs_dir),
            "--job-name", job_name,
            "-n", "1",
            "-t", task_name,
        ]
        if memory_path:
            cmd.extend(["--memory-path", memory_path])
        if self.config.force_build:
            cmd.append("--force-build")

        try:
            result = subprocess.run(cmd, check=False, timeout=3600)
            if result.returncode != 0:
                print(f"  WARNING: harbor returned exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: harbor timed out for {task_name}")

        # Find the job directory
        job_dir = self._find_job_dir(task_name, job_name)
        if job_dir is None:
            raise RuntimeError(
                f"Could not find job directory for {task_name} "
                f"in {self.config.jobs_dir}/{job_name}"
            )
        return job_dir

    def _extract_from_task(self, task_name: str, job_dir: Path) -> bool:
        """Extract memories from a single task's trajectory into the pool."""
        from mtl.extraction.extractor import (
            convert_trajectory,
            get_judgement,
        )
        from mtl.extraction.cognitive_extract import extract_cognitive_memories
        from mtl.extraction.traditional_extract import extract_rawtraj_memory

        traj_path = job_dir / "agent" / "mini-swe-agent.trajectory.json"
        if not traj_path.exists():
            print(f"  WARNING: No trajectory file at {traj_path}")
            return False

        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        messages = traj.get("messages", [])
        trajectory, commands, task = convert_trajectory(messages)

        judgement = get_judgement(job_dir)

        if self.config.only_passed and not judgement:
            print(f"  SKIP extraction: task failed, only_passed=True")
            return False

        log_dir = str(self.pool_dir / task_name).replace("\\", "/")

        try:
            extract_rawtraj_memory(
                judgement, trajectory, log_dir,
                task_name, task, commands, "swebench-verified",
            )
            extract_cognitive_memories(
                judgement, trajectory, log_dir,
                task_name, task, commands, "swebench-verified",
                derive_traditional=self.config.derive_traditional_memory,
                excluded_dimensions=self.config.excluded_dimensions,
                excluded_levels=self.config.excluded_levels,
            )
            return True
        except Exception as e:
            print(f"  ERROR extracting memories: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_judgement(self, job_dir: Path) -> bool:
        """Check if the task passed."""
        result_path = job_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            reward = result.get("verifier_result", {}).get("rewards", {}).get("reward", 0.0)
            return reward >= 1.0
        reward_path = job_dir / "verifier" / "reward.txt"
        if reward_path.exists():
            try:
                return float(reward_path.read_text().strip()) >= 1.0
            except ValueError:
                pass
        return False

    def _find_job_dir(self, task_name: str, job_name: str) -> Path | None:
        """Find the job directory for a task within a named job."""
        job_parent = Path(self.config.jobs_dir) / job_name
        if not job_parent.exists():
            return None
        for d in job_parent.iterdir():
            if d.is_dir() and d.name.startswith(task_name + "__"):
                return d
        return None

    def _count_pool_tasks(self) -> int:
        """Count how many unique tasks have contributed to the memory pool."""
        task_dirs = [d for d in self.pool_dir.iterdir()
                     if d.is_dir() and not d.name.startswith(".")]
        return len(task_dirs)


def load_sequential_config(config_path: str | Path) -> SequentialConfig:
    """Load a sequential experiment configuration from a YAML file."""
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        alt = Path("harbor/configs/experiments") / config_path.name
        if alt.exists():
            config_path = alt
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return SequentialConfig(**{k: v for k, v in data.items()
                               if k in SequentialConfig.__dataclass_fields__})
