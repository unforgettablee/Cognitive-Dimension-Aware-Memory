#!/bin/bash
# Manage mini-swe-agent copies for task Docker builds
#   setup   - copy from template to all task directories (before docker build)
#   cleanup - remove from all task directories (after docker build, before git commit)
#
# Usage:
#   bash scripts/setup_mini_swe_agent.sh setup
#   bash scripts/setup_mini_swe_agent.sh cleanup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$PROJECT_ROOT/harbor/adapters/swebench/template/mini-swe-agent"
TASKS_DIR="$PROJECT_ROOT/harbor-tasks/swebench-verified"

ACTION="${1:-setup}"

if [ ! -d "$TEMPLATE" ]; then
    echo "ERROR: template not found at $TEMPLATE"
    exit 1
fi

if [ ! -d "$TASKS_DIR" ]; then
    echo "ERROR: tasks directory not found at $TASKS_DIR"
    exit 1
fi

case "$ACTION" in
    setup)
        echo "Creating mini-swe-agent directories from template..."
        count=0
        for task_dir in "$TASKS_DIR"/*/; do
            dst="$task_dir/environment/mini-swe-agent"
            if [ -d "$dst" ]; then
                continue
            fi
            mkdir -p "$(dirname "$dst")"
            cp -r "$TEMPLATE" "$dst"
            count=$((count + 1))
        done
        echo "Done. Created $count mini-swe-agent directories."
        ;;

    cleanup)
        echo "Removing mini-swe-agent directories..."
        count=0
        for task_dir in "$TASKS_DIR"/*/; do
            dst="$task_dir/environment/mini-swe-agent"
            if [ -d "$dst" ]; then
                rm -rf "$dst"
                count=$((count + 1))
            fi
        done
        echo "Done. Removed $count mini-swe-agent directories."
        ;;

    *)
        echo "ERROR: unknown action '$ACTION'. Use 'setup' or 'cleanup'."
        exit 1
        ;;
esac
