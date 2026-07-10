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
        verifier = result.get("verifier_result") or {}
        reward = (verifier or {}).get("rewards", {}).get("reward", 0.0)
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


def extract_for_task(task_name: str, jobs_dir: Path, memory_dir: Path,
                     derive_traditional: bool = True,
                     excluded_dimensions: list | None = None,
                     excluded_levels: list | None = None) -> bool:
    """Extract memories for a single task.

    Returns:
        True if cognitive extraction (Phase 2) succeeded.
        False if it failed critically (trajectory summarization LLM call failed).
        Returns True for "skip" cases (no job dir, no trajectory file) since
        those are expected and not errors.
    """
    job_dir = _find_job_dir(jobs_dir, task_name)
    if job_dir is None:
        print(f"SKIP {task_name}: no job directory found")
        return True  # not an error

    # Try trajectory sources in priority order:
    #   1. mini-swe-agent.trajectory.json  (raw agent messages, preferred)
    #   2. trajectory.json                  (ATIF-converted, partial but usable)
    #   3. mini-swe-agent.txt              (plain-text agent log, last resort)
    agent_dir = job_dir / "agent"
    traj_path = None
    traj_source = ""
    messages = []

    if agent_dir.exists():
        # Priority 1: full raw trajectory
        raw_traj = agent_dir / "mini-swe-agent.trajectory.json"
        if raw_traj.exists():
            traj_path = raw_traj
            traj_source = "mini-swe-agent.trajectory.json"
        # Priority 2: ATIF-converted trajectory (partial but structured)
        if traj_path is None:
            atif_traj = agent_dir / "trajectory.json"
            if atif_traj.exists():
                traj_path = atif_traj
                traj_source = "trajectory.json (ATIF)"

    if traj_path is None:
        print(f"SKIP {task_name}: no trajectory file found in {agent_dir}")
        # Try to build a minimal trajectory from agent text log
        txt_log = agent_dir / "mini-swe-agent.txt" if agent_dir.exists() else None
        if txt_log and txt_log.exists():
            print(f"  Falling back to text log: {txt_log}")
            task_text = ""
            # Try to read task from instruction.md in the trial dir
            instr_path = job_dir / ".." / ".." / ".." / ".." / "harbor-tasks" / "swebench-verified" / task_name / "instruction.md"
            # Build a minimal trajectory shell
            trajectory = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": task_text or task_name},
            ]
            # Pass the text log content as observation
            log_text = txt_log.read_text(encoding="utf-8", errors="replace")
            if len(log_text) > 100000:
                log_text = log_text[:50000] + "\n... (truncated) ...\n" + log_text[-50000:]
            trajectory.append({"role": "assistant", "content": log_text[:50000]})
            judgement = get_judgement(job_dir)
            commands = []
            task = task_text or task_name
            log_dir = str(memory_dir / task_name).replace("\\", "/")
            print(f"  Attempting extraction from partial text log ({len(log_text)} chars)...")
            # Only run Phase 2 (cognitive + traditional) on partial data
            cog_ok = extract_cognitive_memories(
                judgement, trajectory, log_dir, task_name, task, commands, BENCHMARK,
                derive_traditional=derive_traditional,
                excluded_dimensions=excluded_dimensions,
                excluded_levels=excluded_levels,
            )
            print(f"  DONE: {task_name} (from text log, cognitive_ok={cog_ok})")
            return cog_ok
        return True  # not an error — nothing to extract from

    print(f"  Trajectory source: {traj_source}")

    print(f"\n{'='*60}")
    print(f"Processing: {task_name} ({job_dir.name})")

    traj = json.loads(traj_path.read_text(encoding="utf-8"))

    # Normalize trajectory format:
    #   - mini-swe-agent.trajectory.json → {"messages": [{"role": ..., "content": ...}, ...]}
    #   - trajectory.json (ATIF)          → {"steps": [{"source": ..., "message": ...}, ...]}
    if "messages" in traj:
        messages = traj["messages"]
    elif "steps" in traj:
        # Convert ATIF steps to message format
        messages = []
        for step in traj["steps"]:
            role = step.get("source", "")
            content = step.get("message", "")
            if role == "tool":
                role = "user"  # harbor convert_trajectory treats tool as user
            messages.append({"role": role, "content": content})
        print(f"  (converted {len(messages)} ATIF steps to messages)")
    elif "trajectory" in traj:
        messages = traj["trajectory"]
    else:
        # Last resort: wrap the entire JSON as a single message
        print(f"  WARNING: unknown trajectory format, using raw JSON dump")
        messages = [{"role": "user", "content": json.dumps(traj, ensure_ascii=False)[:50000]}]

    trajectory, commands, task = convert_trajectory(messages)

    judgement = get_judgement(job_dir)
    status = "PASS" if judgement else "FAIL"
    print(f"  Judgement: {status} | Commands: {len(commands)} | Messages: {len(messages)}")

    log_dir = str(memory_dir / task_name).replace("\\", "/")

    # Phase 1: Raw trajectory (no LLM call, embedding only)
    print(f"  [1/2] rawtraj_memory...")
    extract_rawtraj_memory(judgement, trajectory, log_dir, task_name, task, commands, BENCHMARK)

    # Phase 2: Cognitive + Traditional memories (5 LLM calls total)
    print(f"  [2/2] cognitive_memories (summary + 4 combined level extractions"
          f"{' + traditional derivation' if derive_traditional else ''})...")
    cog_ok = extract_cognitive_memories(
        judgement, trajectory, log_dir, task_name, task, commands, BENCHMARK,
        derive_traditional=derive_traditional,
        excluded_dimensions=excluded_dimensions,
        excluded_levels=excluded_levels,
    )

    if cog_ok:
        print(f"  DONE: {task_name}")
    else:
        print(f"  DONE: {task_name} (cognitive extraction FAILED — "
              f"trajectory summarization LLM error)")

    return cog_ok


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
    parser.add_argument("--no-derive-traditional", action="store_true",
                        help="Skip deriving traditional memories (workflow/local/summary/insight) "
                             "from cognitive output. Only cognitive 4x4 matrix is extracted.")
    parser.add_argument("--excluded-dimensions", nargs="*", default=None,
                        help="Cognitive dimensions to EXCLUDE from extraction "
                             "(e.g. --excluded-dimensions causal contrastive)")
    parser.add_argument("--excluded-levels", nargs="*", default=None,
                        help="Abstraction levels to EXCLUDE from extraction "
                             "(e.g. --excluded-levels trajectory workflow)")
    args = parser.parse_args()

    api_key = os.getenv("API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: Neither API_KEY nor DEEPSEEK_API_KEY is set")
        print('  export API_KEY="sk-xxx"')
        sys.exit(1)
    # Ensure both are set so sub-modules can find them regardless of which name they use
    os.environ.setdefault("API_KEY", api_key)
    os.environ.setdefault("DEEPSEEK_API_KEY", api_key)

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

    derive_traditional = not args.no_derive_traditional

    if not selected:
        print("No tasks selected. Check --start and --limit.")
        sys.exit(0)

    failed_tasks: list[str] = []
    for i, name in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {name}")
        try:
            ok = extract_for_task(name, jobs_dir, memory_dir,
                                  derive_traditional=derive_traditional,
                                  excluded_dimensions=args.excluded_dimensions,
                                  excluded_levels=args.excluded_levels)
            if not ok:
                failed_tasks.append(name)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed_tasks.append(name)

    print(f"\n{'='*60}")
    print("All done. Memory files:")
    parent = memory_dir.parent
    for pkl in sorted(parent.glob("*.pkl")):
        print(f"  {pkl}")

    if failed_tasks:
        print(f"\nERROR: {len(failed_tasks)} task(s) had critical extraction failures:")
        for name in failed_tasks:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
