#!/bin/bash
# Call the qwenasr OpenAI-compatible transcription server.

host="${1:-127.0.0.1}"
port="${2:-8090}"

# Default json response : {"text": "..."}.
curl -s -X POST "http://${host}:${port}/v1/audio/transcriptions" \
    -F file=@freeman.wav \
    -F response_format=json
echo

# Plain transcript written to a file.
curl -s -X POST "http://${host}:${port}/v1/audio/transcriptions" \
    -F file=@freeman.wav \
    -F response_format=text \
    --output output.txt
echo "wrote output.txt"
