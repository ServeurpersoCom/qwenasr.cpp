#!/bin/bash
# Download Qwen3-ASR checkpoints from HuggingFace. Both sizes: 0.6B for the
# embedded target, 1.7B for top quality. The forced aligner stays out until
# timestamp support lands.
# Usage: ./checkpoints.sh

set -eu

DIR="checkpoints"
mkdir -p "$DIR"

dl_repo() {
    local name="$1" repo="$2"
    local target="$DIR/$name"
    if [ -d "$target" ] && [ "$(ls "$target"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[OK] $name"
        return
    fi
    echo "[Download] $name <- $repo"
    hf download --quiet "$repo" --local-dir "$target"
}

dl_repo "Qwen3-ASR-0.6B" "Qwen/Qwen3-ASR-0.6B"
dl_repo "Qwen3-ASR-1.7B" "Qwen/Qwen3-ASR-1.7B"
