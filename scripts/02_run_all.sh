#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export VIT_RESULT_ROOT="${VIT_RESULT_ROOT:-reproduction}"
"${PYTHON:-python}" -m src.run_suite "$@"
