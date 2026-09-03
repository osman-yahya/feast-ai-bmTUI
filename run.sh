#!/usr/bin/env bash
# feast AI launcher (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")"

if ! python3 -c "import textual, psutil, httpx" 2>/dev/null; then
    echo "Installing dependencies..."
    python3 -m pip install -r requirements.txt
fi

exec python3 -m feast_ai "$@"
