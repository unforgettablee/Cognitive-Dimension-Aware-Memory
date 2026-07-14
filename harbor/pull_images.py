#!/usr/bin/env python3
"""Pull SWE-bench Docker base images for E2 experiment (seed=42, 100 tasks).

Skips images already present locally. Supports parallel pulls.
"""

import subprocess
import sys
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

TASKS_DIR = Path(__file__).resolve().parent.parent / "harbor-tasks" / "swebench-verified"
SEED = 42
NUM_TASKS = 100
PARALLEL = 1  # concurrent pulls (set to 1 for large images on slow connections)
MAX_RETRIES = 5  # retry failed pulls


def get_all_tasks() -> list[str]:
    return sorted(d.name for d in TASKS_DIR.iterdir() if d.is_dir())


def task_to_image(task_name: str) -> str:
    """Convert task name to swebench Docker image name.

    Example: django__django-13807 -> swebench/sweb.eval.x86_64.django_1776_django-13807:latest
    """
    parts = task_name.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse task name: {task_name}")
    org, repo_issue = parts
    repo, issue = repo_issue.rsplit("-", 1)
    # Map org to the swebench Docker Hub org prefix
    org_map = {
        "django": "django",
        "astropy": "astropy",
        "matplotlib": "matplotlib",
        "mwaskom": "mwaskom",
        "psf": "psf",
        "pydata": "pydata",
        "pylint-dev": "pylint-dev",
        "pytest-dev": "pytest-dev",
        "scikit-learn": "scikit-learn",
        "sphinx-doc": "sphinx-doc",
        "sympy": "sympy",
    }
    docker_org = org_map.get(org, org)
    # Issue may have sub-number like 1776_4.0 -> strip to just the issue number
    issue_simple = issue.split("_")[0] if "_" in issue else issue
    image_name = f"swebench/sweb.eval.x86_64.{docker_org}_1776_{repo}-{issue_simple}"
    return f"{image_name}:latest"


def image_exists(image: str) -> bool:
    """Check if a Docker image already exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def pull_image(image: str, max_retries: int = MAX_RETRIES) -> tuple[str, bool]:
    """Pull a single Docker image with retries. Returns (image, success).

    Every attempt streams Docker's progress output directly to the terminal.
    """
    for attempt in range(1, max_retries + 1):
        task_name = image.split("/")[-1].replace(":latest", "").replace("sweb.eval.x86_64.", "")
        label = f"[{attempt}/{max_retries}]"
        print(f"\n{'='*60}")
        print(f"PULL {label} {task_name}")
        print(f"{'='*60}")

        # Always show real-time progress
        result = subprocess.run(["docker", "pull", image])
        if result.returncode == 0:
            print(f"✅ DONE: {task_name}\n")
            return image, True

        # Failed — retry if we have attempts left
        if attempt < max_retries:
            wait = attempt * 5
            print(f"Failed. Retrying in {wait}s...")
            time.sleep(wait)

    task_name = image.split("/")[-1].replace(":latest", "")
    print(f"  ❌ FAILED (after {max_retries} tries): {task_name}\n")
    return image, False


def main():
    print(f"Loading tasks from: {TASKS_DIR}")
    all_tasks = get_all_tasks()
    print(f"Total tasks available: {len(all_tasks)}")

    rng = random.Random(SEED)
    selected = sorted(rng.sample(all_tasks, NUM_TASKS))
    print(f"Selected {len(selected)} tasks (seed={SEED})")

    # Build image list
    images = []
    for task in selected:
        try:
            img = task_to_image(task)
            images.append((task, img))
        except ValueError as e:
            print(f"  SKIP parse error: {e}")

    # Filter already-pulled
    to_pull = []
    skipped = 0
    for task, img in images:
        if image_exists(img):
            skipped += 1
        else:
            to_pull.append((task, img))

    print(f"Already cached: {skipped}")
    print(f"Need to pull:   {len(to_pull)}")

    if not to_pull:
        print("All images already cached. Nothing to do.")
        return

    print(f"\nPulling {len(to_pull)} images with {PARALLEL} parallel workers...\n")

    ok_count = 0
    fail_count = 0
    failed_images = []

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {
            pool.submit(pull_image, img): (task, img)
            for task, img in to_pull
        }
        for i, future in enumerate(as_completed(futures), 1):
            img, ok = future.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                failed_images.append(img)
            print(f"  Progress: {i}/{len(to_pull)} (OK={ok_count} FAIL={fail_count})")

    print(f"\n{'='*60}")
    print(f"Done. OK={ok_count} FAIL={fail_count}")
    if failed_images:
        print(f"\nFailed images (retry manually):")
        for img in failed_images:
            print(f"  docker pull {img}")


if __name__ == "__main__":
    main()
