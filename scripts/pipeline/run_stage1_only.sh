#!/bin/bash
# Run Stage 1 only
set -e
bash "$(dirname "$0")/../stage1/run.sh" "$@"
