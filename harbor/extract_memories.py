"""
Extract cognitive + traditional memories from harbor job trajectories.

Usage (PowerShell, from project root):
    .venv-harbor/Scripts/activate
    $env:DEEPSEEK_API_KEY = "sk-xxx"
    python harbor/extract_memories.py --jobs-dir jobs/2026-06-01/2026-06-01__11-07-32 --limit 20
    python harbor/extract_memories.py --jobs-dir jobs/2026-06-01/2026-06-01__11-07-32 --start 50 --limit 30
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from experiments.utils.memory_extract import (
    extract_rawtraj_memory,
)
from experiments.utils.cognitive_memory_extract import extract_cognitive_memories

BENCHMARK = "swebench-verified"


def _find_job_dir(jobs_dir: Path, task_name: str) -> Path | None:
    """Find the job directory for a given task_name (ignoring random suffix)."""
    for d in jobs_dir.iterdir():
        if d.is_dir() and d.name.startswith(task_name + "__"):
            return d
    return None


def convert_trajectory(messages: list) -> tuple[list[dict], list[str], str]:
    """Convert harbor/mini-extra messages to (trajectory, commands, task) format."""
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
    # Fallback: verifier/reward.txt
    reward_path = job_dir / "verifier" / "reward.txt"
    if reward_path.exists():
        try:
            return float(reward_path.read_text().strip()) >= 1.0
        except ValueError:
            pass
    print(f"  WARNING: no reward found for {job_dir.name}, defaulting to FAIL")
    return False


def extract_for_task(task_name: str, jobs_dir: Path, memory_dir: Path):
    job_dir = _find_job_dir(jobs_dir, task_name)
    if job_dir is None:
        print(f"SKIP {task_name}: no job directory found")
        return

    traj_path = job_dir / "agent" / "mini-swe-agent.trajectory.json"
    if not traj_path.exists():
        print(f"SKIP {task_name}: no trajectory file at {traj_path}")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {task_name} ({job_dir.name})")

    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    messages = traj.get("messages", [])
    trajectory, commands, task = convert_trajectory(messages)

    judgement = get_judgement(job_dir)
    status = "PASS" if judgement else "FAIL"
    print(f"  Judgement: {status} | Commands: {len(commands)} | Messages: {len(messages)}")

    log_dir = str(memory_dir / task_name).replace("\\", "/")

    # Phase 1: Raw trajectory (no LLM call, embedding only)
    print(f"  [1/2] rawtraj_memory...")
    extract_rawtraj_memory(judgement, trajectory, log_dir, task_name, task, commands, BENCHMARK)

    # Phase 2: Cognitive + Traditional memories (5 LLM calls total)
    print(f"  [2/2] cognitive_memories (summary + 4 combined level extractions + traditional derivation)...")
    extract_cognitive_memories(judgement, trajectory, log_dir, task_name, task, commands, BENCHMARK)

    print(f"  DONE: {task_name}")


def main():
    parser = argparse.ArgumentParser(description="Extract memories from harbor job trajectories")
    parser.add_argument("--jobs-dir", required=True,
                        help="Directory containing job output subdirectories")
    parser.add_argument("--memory-dir", default="memories/swebench-verified",
                        help="Directory to store memory files (default: memories/swebench-verified)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max number of tasks to process (default: 100)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from Nth task, 0-based (default: 0)")
    args = parser.parse_args()

    if not os.getenv("API_KEY"):
        print("ERROR: API_KEY not set")
        print('  export API_KEY="sk-xxx"')
        sys.exit(1)

    jobs_dir = Path(args.jobs_dir)
    if not jobs_dir.exists():
        print(f"ERROR: jobs-dir not found: {jobs_dir}")
        sys.exit(1)

    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Collect task names from job directories (strip random suffix after last __)
    task_names = sorted(
        d.name.rsplit("__", 1)[0]
        for d in jobs_dir.iterdir()
        if d.is_dir() and "__" in d.name
    )
    # De-duplicate (in case of re-runs keeping same task_name)
    seen = set()
    unique_names = []
    for name in task_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    task_names = unique_names

    # Apply start/limit
    selected = task_names[args.start:args.start + args.limit]

    print(f"Found {len(task_names)} job directories total")
    print(f"Selected: [{args.start}:{args.start + args.limit}] = {len(selected)} tasks")
    print(f"Jobs dir: {jobs_dir.resolve()}")
    print(f"Memory output: {memory_dir.resolve()}")

    if not selected:
        print("No tasks selected. Check --start and --limit.")
        sys.exit(0)

    for i, name in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {name}")
        try:
            extract_for_task(name, jobs_dir, memory_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All done. Memory files:")
    parent = memory_dir.parent
    for pkl in sorted(parent.glob("*.pkl")):
        print(f"  {pkl}")


if __name__ == "__main__":
    main()
