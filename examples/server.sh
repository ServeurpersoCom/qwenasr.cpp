#!/bin/bash
# Start the qwenasr OpenAI-compatible transcription server.

./build/asr-server \
    --model models/qwenasr-1.7B-BF16.gguf \
    --host 127.0.0.1 --port 8090
