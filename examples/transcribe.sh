#!/bin/bash

set -eu

../build/qwenasr-transcribe \
    --model ../models/qwenasr-1.7B-BF16.gguf \
    --file freeman.wav \
    --lang English \
    -o output.txt
