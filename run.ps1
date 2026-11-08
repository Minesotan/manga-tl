# Convenience wrapper: .\run.ps1 <pages-folder-or-image> [extra options]
# Starts Ollama if needed, then runs the translator in the project venv.
$py = "$env:USERPROFILE\.venvs\manga-tl\Scripts\python.exe"
try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags -TimeoutSec 2 | Out-Null }
catch { Start-Process ollama -ArgumentList serve -WindowStyle Hidden; Start-Sleep 3 }
& $py "$PSScriptRoot\manga_tl.py" @args
