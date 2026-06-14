#!/bin/bash
# Download pre-quantized qwenasr.cpp GGUF models from HuggingFace.
# Usage: ./models.sh

set -eu

REPO="Serveurperso/qwenasr.cpp-GGUF"
DIR="models"
mkdir -p "$DIR"

dl() {
    local file="$1"
    if [ -f "$DIR/$file" ]; then
        echo "[OK] $file"
        return
    fi
    echo "[Download] $file"
    hf download --quiet "$REPO" "$file" --local-dir "$DIR"
}

dl "qwenasr-0.6B-Q8_0.gguf"
dl "qwenasr-1.7B-Q8_0.gguf"
