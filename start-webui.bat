@echo off
rem Double-click to start the Manga Translator web UI (opens in your browser).
cd /d "%~dp0"
curl -s -o nul http://localhost:11434/api/tags || (start "" /min ollama serve & timeout /t 3 >nul)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%USERPROFILE%\.venvs\manga-tl\Scripts\python.exe" webui.py
pause
