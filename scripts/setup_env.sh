#!/usr/bin/env bash
# Build (or rebuild) the project virtualenv in .venv and install Style Eval into it.
#
#   scripts/setup_env.sh            create .venv if missing, install package + test deps
#   scripts/setup_env.sh --fresh    delete .venv first and rebuild from scratch
#   PYTHON=python3.12 scripts/setup_env.sh    pick the interpreter explicitly
set -euo pipefail

MIN_MAJOR=3
MIN_MINOR=11

cd "$(dirname "$0")/.."

ok_python() {
  "$1" -c "import sys; sys.exit(0 if sys.version_info >= ($MIN_MAJOR, $MIN_MINOR) else 1)" 2>/dev/null
}

pick_python() {
  if [ -n "${PYTHON:-}" ]; then
    command -v "$PYTHON" || true
    return
  fi
  local candidate path
  for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    path=$(command -v "$candidate" || true)
    if [ -n "$path" ] && ok_python "$path"; then
      echo "$path"
      return
    fi
  done
}

if [ "${1:-}" = "--fresh" ]; then
  rm -rf .venv
fi

if [ ! -x .venv/bin/python ]; then
  py=$(pick_python)
  if [ -z "$py" ] || ! ok_python "$py"; then
    echo "error: need Python $MIN_MAJOR.$MIN_MINOR or newer on PATH (or set PYTHON=/path/to/python)." >&2
    exit 1
  fi
  echo "Creating .venv with $py ($("$py" --version))"
  "$py" -m venv .venv
fi

.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e ".[dev]"

echo
echo "Ready. Activate with:  source .venv/bin/activate"
echo "Then:                  style-eval --help"
echo "Run tests with:        python -m pytest"
