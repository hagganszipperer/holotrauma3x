#!/usr/bin/env bash
set -euo pipefail

echo "HoloTrauma 3X ships a deterministic synthetic trauma cohort generated at runtime."
echo "No external download is required; clinical datasets are controlled-access (see docs/project-context.md)."
echo "A single planning demonstration:"
holotrauma3x plan --config "${1:-configs/experiment/_smoke.yaml}"
