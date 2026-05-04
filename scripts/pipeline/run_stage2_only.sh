#!/bin/bash
# Run Stage 2 only (requires precompute and stage1 output)
set -e
bash "$(dirname "$0")/../stage2/run.sh" "$@"
