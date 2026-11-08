# manga-tl

Translate Japanese manga pages to English **locally**: detect speech bubbles, OCR the Japanese,
translate with a local Ollama model, erase the original text and letter the English into the bubble.

## Use it

**Web UI:** double-click `start-webui.bat` (opens http://127.0.0.1:7860).
Drop in many pages or a `.zip` chapter, click **Translate**, then download the translated zip.
Results are also saved in `output/<date_time>/`.

**Command line:**
```powershell
.\run.ps1 C:\path\to\chapter1            # writes C:\path\to\chapter1_en\
.\run.ps1 page.png -o out --uppercase --debug
```
Options: `--model` (any Ollama model, default `qwen3-swallow:8b`), `--font`, `--uppercase`,
`--conf` (bubble detection threshold), `--debug` (saves an overlay of the detected bubbles).

Each output page gets a `.json` next to it with the Japanese OCR and English translation per bubble.

## How it works

| Step | Tool |
|---|---|
| Bubble detection + shape | [kitsumed/yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble) (YOLOv8 segmentation, GPL-3.0) |
| OCR (vertical Japanese) | [manga-ocr](https://github.com/kha-white/manga-ocr) |
| Translation | Ollama (`qwen3-swallow:8b` by default) — whole page at once, in right-to-left reading order, with earlier pages as context |
| Text removal | repaint text pixels inside the bubble with its background colour |
| Lettering | largest font size whose lines fit inside the actual bubble shape |

Models download automatically on first run to `~/.cache/huggingface`.

## Fonts

Put any `.ttf`/`.otf` in a `fonts/` folder and it appears in the web UI font list.
Comic lettering fonts such as *Wild Words* or *Anime Ace* (Blambot, free for personal use) look much
more like official translations than Comic Sans.

## Setup (already done on this PC)

```powershell
py -3.13 -m venv $env:USERPROFILE\.venvs\manga-tl     # kept outside OneDrive on purpose
$py = "$env:USERPROFILE\.venvs\manga-tl\Scripts\python.exe"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt
```

## License & credit

Licensed under the [Apache License 2.0](LICENSE). You are free to use, modify and redistribute this
software however you like, **as long as you credit the original author, Rod Morabito, and link to this
repository** (https://github.com/Minesotan/manga-tl). Keep the [NOTICE](NOTICE) file with any copy or
derivative work.

Third-party components keep their own licenses: the speech-bubble model is GPL-3.0, Ultralytics is
AGPL-3.0 and manga-ocr is Apache-2.0. They're downloaded at install/run time, not bundled here.

## Limitations

- Only text **inside speech bubbles** is translated; sound effects and free-floating narration are left alone.
- Bubbles on textured/gradient backgrounds are cleaned less perfectly than plain white ones.
- OCR can pick up stray marks from artwork that touches a bubble; check the `.json` if a line looks odd.
