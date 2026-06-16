@echo off

set PATH=%~dp0..\build\Release;%PATH%

qwenasr-transcribe.exe ^
    --model ..\models\qwenasr-1.7B-BF16.gguf ^
    --file freeman.wav ^
    --lang English ^
    -o output.txt

pause
