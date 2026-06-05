#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
holotrauma3x train --config "configs/experiment/${EXPERIMENT}.yaml"
