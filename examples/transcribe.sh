#!/bin/bash

set -eu

../build/qwenasr-transcribe \
    --model ../models/qwenasr-0.6B-BF16.gguf \
    --file freeman.wav \
    --lang English \
    -o output.txt
