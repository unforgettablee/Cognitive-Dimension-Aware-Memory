#!/usr/bin/env python3
"""
Compare ablation experiment results against the no-memory baseline (E6).

Given a baseline ``sequential_summary.json`` (E6, all 500 tasks) and one or more
experiment summaries (100 randomly-sampled tasks), extracts the subset of
baseline results that match the experiment's task list and produces a side-by-side
comparison.

Usage::

    # Compare one experiment
    python harbor/compare_ablation.py \\
        --baseline jobs/sequential/e6_no_memory/2026-06-30__12-00-00/sequential_summary.json \\
        --experiment jobs/sequential/e4_embedding/2026-06-30__14-00-00/sequential_summary.json \\
        --output comparison_e4.json

    # Compare multiple experiments at once
    python harbor/compare_ablation.py \\
        --baseline jobs/sequential/e6_no_memory/2026-06-30__12-00-00/sequential_summary.json \\
        --experiments jobs/sequential/e1_full/2026-06-30__14-00-00/sequential_summary.json \\
                     jobs/sequential/e2_no_rerank/2026-06-30__15-00-00/sequential_summary.json \\
        --output-dir comparisons/

Output per experiment::

    {
      "baseline": {
        "name": "e6_no_memory",
        "total_tasks": 500,
        "pass_rate": 12.0
      },
      "experiment": {
        "name": "e4_pure_embedding",
        "total_tasks": 100,
        "pass_rate": 18.0
      },
      "matched_tasks": 100,
      "baseline_matched": {
        "passed": 13,
        "total": 100,
        "pass_rate": 13.0
      },
      "experiment_matched": {
        "passed": 18,
        "total": 100,
        "pass_rate": 18.0
      },
      "delta": {
        "absolute_pass_rate": 5.0,
        "relative_improvement": 38.5
      },
      "details": [
        {
          "task": "astropy__astropy-12907",
          "baseline": "PASS",
          "experiment": "FAIL"
        },
        ...
      ]
    }
"""

import argparse
import json
import sys
from pathlib import Path


def load_summary(path: Path) -> dict:
    """Load a sequential_summary.json file."""
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_baseline_index(baseline: dict) -> dict[str, dict]:
    """Build a task-name -> result lookup from baseline results."""
    index = {}
    for r in baseline.get("results", []):
        task_name = r.get("task", "")
        if task_name:
            index[task_name] = r
    return index


def compare(baseline_path: Path, experiment_path: Path) -> dict:
    """Compare one experiment against the baseline, returning a comparison dict."""
    baseline = load_summary(baseline_path)
    experiment = load_summary(experiment_path)

    baseline_index = build_baseline_index(baseline)

    # Build details for each experiment task
    details = []
    matched_passed_exp = 0
    matched_passed_base = 0
    matched_total = 0

    for r in experiment.get("results", []):
        task_name = r.get("task", "")
        base_result = baseline_index.get(task_name)
        if base_result is None:
            # Task not in baseline — skip (shouldn't happen if using same task set)
            continue

        matched_total += 1
        exp_passed = r.get("passed", False)
        base_passed = base_result.get("passed", False)

        if exp_passed:
            matched_passed_exp += 1
        if base_passed:
            matched_passed_base += 1

        details.append({
            "task": task_name,
            "baseline": "PASS" if base_passed else "FAIL",
            "experiment": "PASS" if exp_passed else "FAIL",
        })

    # Compute deltas
    base_rate = round(matched_passed_base / matched_total * 100, 1) if matched_total else 0
    exp_rate = round(matched_passed_exp / matched_total * 100, 1) if matched_total else 0
    abs_delta = round(exp_rate - base_rate, 1)
    rel_delta = round((exp_rate - base_rate) / base_rate * 100, 1) if base_rate else 0

    return {
        "baseline": {
            "name": baseline.get("name", "unknown"),
            "total_tasks": baseline.get("total_tasks", 0),
            "pass_rate": baseline.get("pass_rate", 0),
        },
        "experiment": {
            "name": experiment.get("name", "unknown"),
            "total_tasks": experiment.get("total_tasks", 0),
            "pass_rate": experiment.get("pass_rate", 0),
        },
        "matched_tasks": matched_total,
        "baseline_matched": {
            "passed": matched_passed_base,
            "total": matched_total,
            "pass_rate": base_rate,
        },
        "experiment_matched": {
            "passed": matched_passed_exp,
            "total": matched_total,
            "pass_rate": exp_rate,
        },
        "delta": {
            "absolute_pass_rate": abs_delta,
            "relative_improvement_pct": rel_delta,
        },
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare ablation experiments against the no-memory baseline (E6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Path to E6 baseline sequential_summary.json (500 tasks)",
    )
    parser.add_argument(
        "--experiment",
        help="Path to one experiment sequential_summary.json",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        help="Paths to multiple experiment sequential_summary.json files",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path (single experiment mode)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for multiple experiments (auto-names files)",
    )

    args = parser.parse_args()

    # Collect experiment paths
    experiment_paths = []
    if args.experiment:
        experiment_paths.append(Path(args.experiment))
    if args.experiments:
        experiment_paths.extend(Path(p) for p in args.experiments)

    if not experiment_paths:
        print("ERROR: At least one --experiment or --experiments required")
        sys.exit(1)

    baseline_path = Path(args.baseline)

    results = []
    for exp_path in experiment_paths:
        print(f"Comparing: {exp_path.name}  (parent: {exp_path.parent.name})")
        comparison = compare(baseline_path, exp_path)
        results.append((exp_path, comparison))

    # Output
    if len(results) == 1 and args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results[0][1], indent=2, ensure_ascii=False))
        print(f"Saved to: {out_path}")
    elif args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for exp_path, comparison in results:
            exp_name = comparison["experiment"]["name"]
            out_path = out_dir / f"comparison_{exp_name}.json"
            out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
            print(f"Saved to: {out_path}")
    else:
        # Print to stdout
        for exp_path, comparison in results:
            print(f"\n{'=' * 70}")
            print(f"Experiment: {comparison['experiment']['name']}")
            print(f"Baseline:   {comparison['baseline']['name']}")
            print(f"Matched tasks: {comparison['matched_tasks']}")
            print(f"Baseline matched pass rate: {comparison['baseline_matched']['pass_rate']}%")
            print(f"Experiment pass rate:       {comparison['experiment_matched']['pass_rate']}%")
            print(f"Delta: +{comparison['delta']['absolute_pass_rate']} pp "
                  f"({comparison['delta']['relative_improvement_pct']:+.1f}%)")
            print(f"\nPer-task details:")
            for d in comparison["details"]:
                arrow = "↑" if d["experiment"] == "PASS" and d["baseline"] == "FAIL" else \
                        "↓" if d["experiment"] == "FAIL" and d["baseline"] == "PASS" else " "
                print(f"  {arrow} {d['task']:50s}  baseline={d['baseline']:4s}  experiment={d['experiment']:4s}")


if __name__ == "__main__":
    main()
