"""Memory extraction orchestrator.

Extracts memories from harbor job trajectories using cognitive + traditional methods.
Supports both single-task and batch extraction.
"""
import json
import os
import pickle
import sys
from pathlib import Path

from mtl.extraction.cognitive_extract import extract_cognitive_memories
from mtl.extraction.traditional_extract import extract_rawtraj_memory

DEFAULT_BENCHMARK = "swebench-verified"


def _find_job_dir(jobs_dir: Path, task_name: str) -> Path | None:
    """Find the job directory for a given task_name (ignoring random suffix)."""
    for d in jobs_dir.iterdir():
        if d.is_dir() and d.name.startswith(task_name + "__"):
            return d
    return None


def convert_trajectory(messages: list) -> tuple[list[dict], list[str], str]:
    """Convert harbor/mini-swe-agent messages to (trajectory, commands, task) format."""
    trajectory = []
    commands = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            trajectory.append({"role": "system", "content": content})
        elif role == "user":
            trajectory.append({"role": "user", "content": content})
        elif role == "assistant":
            if content:
                trajectory.append({"role": "assistant", "content": content})
            for action in msg.get("extra", {}).get("actions", []):
                commands.append(action.get("command", ""))
        elif role == "tool":
            raw = msg.get("extra", {}).get("raw_output", "")
            returncode = msg.get("extra", {}).get("returncode", "")
            obs = f"<returncode>{returncode}</returncode>\n<output>\n{raw}\n</output>"
            trajectory.append({"role": "user", "content": obs})

    task = trajectory[1]["content"] if len(trajectory) > 1 else ""
    return trajectory, commands, task


def get_judgement(job_dir: Path) -> bool:
    """Read reward from harbor result.json (1.0 = pass, 0.0 = fail)."""
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


class MemoryExtractor:
    """Extract memories from harbor job trajectories."""

    def __init__(
        self,
        jobs_dir: str | Path,
        memory_dir: str | Path = "memories/swebench-verified",
        benchmark: str = DEFAULT_BENCHMARK,
        only_passed: bool = False,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
    ):
        """
        Args:
            jobs_dir: Directory containing harbor job output subdirectories
            memory_dir: Directory to store memory files
            benchmark: Benchmark name for memory file naming
            only_passed: Only extract memories from tasks that passed verification
            llm_api_key: API key for LLM calls (default: MTL_LLM_API_KEY or DEEPSEEK_API_KEY env var)
            llm_base_url: API base URL (default: MTL_LLM_BASE_URL env var or https://api.deepseek.com)
            llm_model: Model name for extraction (default: MTL_LLM_MODEL env var or deepseek-chat)
        """
        from mtl.llm import configure

        configure(api_key=llm_api_key, base_url=llm_base_url, model=llm_model)

        self.jobs_dir = Path(jobs_dir)
        self.memory_dir = Path(memory_dir)
        self.benchmark = benchmark
        self.only_passed = only_passed
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_task(self, task_name: str) -> bool:
        """Extract memories for a single task.

        Returns True if extraction succeeded, False otherwise.
        """
        job_dir = _find_job_dir(self.jobs_dir, task_name)
        if job_dir is None:
            print(f"SKIP {task_name}: no job directory found")
            return False

        traj_path = job_dir / "agent" / "mini-swe-agent.trajectory.json"
        if not traj_path.exists():
            print(f"SKIP {task_name}: no trajectory file at {traj_path}")
            return False

        print(f"\n{'='*60}")
        print(f"Processing: {task_name} ({job_dir.name})")

        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        messages = traj.get("messages", [])
        trajectory, commands, task = convert_trajectory(messages)

        judgement = get_judgement(job_dir)
        status = "PASS" if judgement else "FAIL"
        print(f"  Judgement: {status} | Commands: {len(commands)} | Messages: {len(messages)}")

        if self.only_passed and not judgement:
            print(f"  SKIP {task_name}: failed task, only_passed=True")
            return False

        log_dir = str(self.memory_dir / task_name).replace("\\", "/")

        # Phase 1: Raw trajectory (no LLM call, embedding only)
        print(f"  [1/2] rawtraj_memory...")
        extract_rawtraj_memory(judgement, trajectory, log_dir, task_name, task, commands, self.benchmark)

        # Phase 2: Cognitive + Traditional memories (5 LLM calls total)
        print(f"  [2/2] cognitive_memories (summary + 4 combined level extractions)...")
        extract_cognitive_memories(judgement, trajectory, log_dir, task_name, task, commands, self.benchmark)

        print(f"  DONE: {task_name}")
        return True

    def extract_batch(self, start: int = 0, limit: int = 100) -> list[str]:
        """Extract memories for a batch of tasks.

        Args:
            start: Start index (0-based)
            limit: Maximum number of tasks to process

        Returns:
            List of successfully processed task names
        """
        # Collect task names from job directories
        task_names = sorted(
            d.name.rsplit("__", 1)[0]
            for d in self.jobs_dir.iterdir()
            if d.is_dir() and "__" in d.name
        )
        seen = set()
        unique_names = []
        for name in task_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        task_names = unique_names

        selected = task_names[start:start + limit]

        print(f"Found {len(task_names)} job directories total")
        print(f"Selected: [{start}:{start + limit}] = {len(selected)} tasks")
        print(f"Jobs dir: {self.jobs_dir.resolve()}")
        print(f"Memory output: {self.memory_dir.resolve()}")

        if not selected:
            print("No tasks selected. Check start/limit.")
            return []

        success = []
        for i, name in enumerate(selected, 1):
            print(f"\n[{i}/{len(selected)}] {name}")
            try:
                if self.extract_from_task(name):
                    success.append(name)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Print summary
        print(f"\n{'='*60}")
        print(f"All done. Processed {len(success)}/{len(selected)} tasks successfully.")
        print("Memory files:")
        for pkl in sorted(self.memory_dir.glob("*.pkl")):
            print(f"  {pkl}")

        return success
