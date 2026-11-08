"""
Web UI for manga_tl: upload many pages (or a .zip), get translated pages + a zip to download.

  python webui.py            -> open http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import zipfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import gradio as gr
import requests

from manga_tl import IMAGE_EXTS, Pipeline, Translator, Typesetter, natural_key

ROOT = Path(__file__).parent
OUTPUT_ROOT = ROOT / "output"
OLLAMA_URL = "http://localhost:11434"
FONTS = {
    "Comic Sans Bold": "C:/Windows/Fonts/comicbd.ttf",
    "Comic Sans": "C:/Windows/Fonts/comic.ttf",
    "Arial Bold": "C:/Windows/Fonts/arialbd.ttf",
    "Segoe UI Bold": "C:/Windows/Fonts/segoeuib.ttf",
}
# any .ttf/.otf you drop into ./fonts shows up in the font list too (e.g. Wild Words, Anime Ace)
for f in sorted((ROOT / "fonts").glob("*.[ot]tf")) if (ROOT / "fonts").is_dir() else []:
    FONTS[f.stem] = str(f)

_pipeline: Pipeline | None = None
_lock = threading.Lock()


def get_pipeline() -> Pipeline:
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = Pipeline()
    return _pipeline


def ollama_models() -> list[str]:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()["models"]
        return [m["name"] for m in tags]
    except Exception:
        return ["qwen3-swallow:8b"]


def collect_pages(files: list[str], workdir: Path) -> list[Path]:
    """Accept image files and/or zip archives; return image paths in natural page order."""
    pages: list[Path] = []
    for f in files or []:
        p = Path(f)
        if p.suffix.lower() == ".zip":
            dest = workdir / p.stem
            with zipfile.ZipFile(p) as z:
                for info in z.infolist():
                    name = Path(info.filename)
                    if info.is_dir() or name.suffix.lower() not in IMAGE_EXTS or name.name.startswith("."):
                        continue
                    target = dest / name.name  # flatten; never trust paths inside the archive
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                    pages.append(target)
        elif p.suffix.lower() in IMAGE_EXTS:
            pages.append(p)
    return sorted(pages, key=natural_key)


def run(files, model, font_name, uppercase):
    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(msg)

    if not files:
        yield [], None, "Upload some manga pages (images or a .zip) first."
        return
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
    except Exception:
        yield [], None, "Ollama isn't running. Start it (run `ollama serve` or open the Ollama app) and try again."
        return

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="manga_tl_"))
    pages = collect_pages(files, work)
    if not pages:
        yield [], None, "No images found in the upload."
        return

    log("Loading models (first run takes a minute)...")
    yield [], None, "\n".join(log_lines)
    pipe = get_pipeline()
    tl = Translator(model, OLLAMA_URL)  # fresh context per batch = per chapter
    ts = Typesetter(FONTS[font_name], uppercase)

    gallery: list[tuple[str, str]] = []
    for i, page in enumerate(pages, 1):
        log(f"[{i}/{len(pages)}] {page.name}")
        yield gallery, None, "\n".join(log_lines)
        try:
            out_path = pipe.process_file(page, out_dir, tl, ts, log=log)
            gallery.append((str(out_path), page.name))
        except Exception as e:
            log(f"  FAILED: {e}")
        yield gallery, None, "\n".join(log_lines)

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for img_path, _ in gallery:
            z.write(img_path, Path(img_path).name)
    shutil.rmtree(work, ignore_errors=True)
    log(f"Done: {len(gallery)}/{len(pages)} pages translated. Saved to {out_dir}")
    yield gallery, str(zip_path), "\n".join(log_lines)


def build_ui() -> gr.Blocks:
    models = ollama_models()
    default_model = "qwen3-swallow:8b" if "qwen3-swallow:8b" in models else models[0]
    with gr.Blocks(title="Manga Translator") as demo:
        gr.Markdown("# Manga Translator\nUpload Japanese manga pages (or a .zip of a chapter). "
                    "Everything runs locally on your PC.")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Manga pages or .zip", file_count="multiple",
                                file_types=["image", ".zip"], type="filepath")
                model = gr.Dropdown(models, value=default_model, label="Translation model (Ollama)")
                font = gr.Dropdown(list(FONTS), value=next(iter(FONTS)), label="English font")
                upper = gr.Checkbox(label="ALL CAPS lettering", value=False)
                go = gr.Button("Translate", variant="primary")
                zip_out = gr.File(label="Download translated pages (.zip)")
                status = gr.Textbox(label="Progress", lines=14, max_lines=14, autoscroll=True)
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Translated pages", columns=2, height=900,
                                     object_fit="contain", preview=False)
        go.click(run, [files, model, font, upper], [gallery, zip_out, status])
    return demo


if __name__ == "__main__":
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", inbrowser=True, allowed_paths=[str(OUTPUT_ROOT)])
