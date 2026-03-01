#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Check for ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "Error: ffmpeg is not installed."
    echo "Install it with: brew install ffmpeg"
    exit 1
fi

# Check for Python 3
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# Create virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv and install deps
source "$VENV_DIR/bin/activate"
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Run the main script
python "$SCRIPT_DIR/process_video.py" "$@"
