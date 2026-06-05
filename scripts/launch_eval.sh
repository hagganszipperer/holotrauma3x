#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
holotrauma3x evaluate --config "configs/experiment/${EXPERIMENT}.yaml"
