#!/usr/bin/env bash
set -euo pipefail

PYTHON_ROOT="/Volumes/t7/program/python"
BASE_ENV="$PYTHON_ROOT/environments/segformer"
ENV_DIR="$PYTHON_ROOT/environments/dlka2d"
PYTHON="$PYTHON_ROOT/versions/python-3.9.25/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$PYTHON" ]]; then
  echo "Expected Python 3.9.25 at $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$BASE_ENV/lib/python3.9/site-packages/torch" ]]; then
  echo "Expected existing PyTorch install at $BASE_ENV" >&2
  exit 1
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$PYTHON" -m venv "$ENV_DIR"
fi

# Keep the new environment's own dependencies isolated while reusing the large
# preinstalled PyTorch/timm/torchvision stack on the external SSD.
printf '%s\n' "$BASE_ENV/lib/python3.9/site-packages" \
  > "$ENV_DIR/lib/python3.9/site-packages/segformer-base.pth"
"$ENV_DIR/bin/python" -m pip install -r "$REPO_ROOT/requirements-test.txt"

cat <<EOF
Environment ready: $ENV_DIR
Run tests with:
  $ENV_DIR/bin/python -m pytest -q $REPO_ROOT/tests
EOF
