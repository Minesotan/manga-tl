"""
manga_tl.py - translate Japanese manga pages to English, fully locally.

Pipeline (per page):
  1. Detect speech bubbles   -> YOLOv8 segmentation model (kitsumed/yolov8m_seg-speech-bubble)
  2. OCR the Japanese text   -> manga-ocr (handles vertical text)
  3. Translate               -> your local Ollama model (whole page at once, in reading order)
  4. Erase the Japanese text -> repaint text pixels with the bubble's background colour
  5. Typeset the English     -> largest font size that fits inside the actual bubble shape

Usage:
  python manga_tl.py pages/              # translate every image in a folder
  python manga_tl.py page01.png -o out/  # translate a single page
  python manga_tl.py pages/ --model gemma3:12b --font "C:/Windows/Fonts/comicbd.ttf"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BUBBLE_REPO = "kitsumed/yolov8m_seg-speech-bubble"
BUBBLE_FILE = "model.pt"


# --------------------------------------------------------------------------- data

@dataclass
class Bubble:
    box: tuple[int, int, int, int]          # x1, y1, x2, y2 in page coords
    mask: np.ndarray                        # bool, full-page size
    ja: str = ""
    en: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- image io (unicode-safe on Windows)

def load_image(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"could not encode image: {path}")
    buf.tofile(str(path))


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


# --------------------------------------------------------------------------- 1. detection

class BubbleDetector:
    def __init__(self, conf: float = 0.4):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        self.model = YOLO(hf_hub_download(BUBBLE_REPO, BUBBLE_FILE))
        self.conf = conf

    def detect(self, img: np.ndarray) -> list[Bubble]:
        h, w = img.shape[:2]
        res = self.model(img, conf=self.conf, retina_masks=True, verbose=False)[0]
        if res.masks is None:
            return []
        masks = res.masks.data.cpu().numpy() > 0.5
        bubbles = []
        for box, m in zip(res.boxes.xyxy.cpu().numpy(), masks):
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if m.sum() < 400:  # tiny / spurious
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            bubbles.append(Bubble((max(0, x1), max(0, y1), min(w, x2), min(h, y2)), m))
        return reading_order(bubbles)


def reading_order(bubbles: list[Bubble]) -> list[Bubble]:
    """Manga order: rows top-to-bottom, right-to-left within a row."""
    rows: list[dict] = []
    for b in sorted(bubbles, key=lambda b: b.box[1]):
        y1, y2 = b.box[1], b.box[3]
        for row in rows:
            overlap = min(y2, row["y2"]) - max(y1, row["y1"])
            if overlap > 0.5 * min(y2 - y1, row["y2"] - row["y1"]):
                row["items"].append(b)
                row["y1"], row["y2"] = min(y1, row["y1"]), max(y2, row["y2"])
                break
        else:
            rows.append({"y1": y1, "y2": y2, "items": [b]})
    ordered = []
    for row in sorted(rows, key=lambda r: r["y1"]):
        ordered += sorted(row["items"], key=lambda b: -(b.box[0] + b.box[2]))
    return ordered


# --------------------------------------------------------------------------- 2. OCR

class OCR:
    def __init__(self):
        from manga_ocr import MangaOcr

        self.mocr = MangaOcr()

    def read(self, img: np.ndarray, b: Bubble) -> str:
        x1, y1, x2, y2 = b.box
        crop = img[y1:y2, x1:x2].copy()
        crop[~b.mask[y1:y2, x1:x2]] = 255  # blank out art outside the bubble
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return self.mocr(pil).strip()


JP_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]")


# --------------------------------------------------------------------------- 3. translation (Ollama)

SYSTEM_PROMPT = (
    "You are an expert manga translator. Translate Japanese manga dialogue into natural, "
    "concise English that fits in a speech bubble. Keep the speaker's tone and personality, "
    "keep character names consistent, and romanize names (no honorific changes unless needed). "
    "The input is OCR output and may contain small errors; infer the intended meaning. "
    "Sound effects or lone punctuation like '…' or '!?' should be kept short. "
    "Respond only with the requested JSON."
)


class Translator:
    def __init__(self, model: str, url: str = "http://localhost:11434", context_lines: int = 12):
        self.model = model
        self.url = url.rstrip("/")
        self.history: list[tuple[str, str]] = []  # (ja, en) from previous pages
        self.context_lines = context_lines

    def _chat(self, user: str, schema: dict) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 8192},
        }
        # Note: don't send think=False. Reasoning models like qwen3-swallow then leak their reasoning
        # into the answer and the JSON constraint yields placeholder "..." values.
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=600)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        m = re.search(r"\{.*\}", content, flags=re.S)  # tolerate stray text around the JSON
        return m.group(0) if m else content

    def translate_page(self, lines: list[str]) -> list[str]:
        if not lines:
            return []
        ctx = ""
        if self.history:
            recent = self.history[-self.context_lines:]
            ctx = "Earlier dialogue (for context only, do not translate again):\n" + "\n".join(
                f"- {ja} => {en}" for ja, en in recent) + "\n\n"
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
        user = (f"{ctx}Translate these {len(lines)} speech bubbles from one manga page, listed in "
                f"reading order. Return exactly {len(lines)} translations in the same order.\n\n{numbered}")
        schema = {
            "type": "object",
            "properties": {"translations": {"type": "array", "items": {"type": "string"},
                                            "minItems": len(lines), "maxItems": len(lines)}},
            "required": ["translations"],
        }
        try:
            out = json.loads(self._chat(user, schema))["translations"]
            if len(out) != len(lines):
                raise ValueError("count mismatch")
            if all(not re.search(r"[A-Za-z]", t) for t in out if t) and any(JP_CHARS.search(l) for l in lines):
                raise ValueError("model returned placeholders")
        except Exception as e:  # fall back to one bubble at a time
            print(f"    batch translation failed ({e}); translating bubbles one by one")
            out = [self._translate_one(t) for t in lines]
        out = [str(t).strip() for t in out]
        self.history += list(zip(lines, out))
        return out

    def _translate_one(self, text: str) -> str:
        schema = {"type": "object", "properties": {"translation": {"type": "string"}},
                  "required": ["translation"]}
        try:
            return json.loads(self._chat(f"Translate this speech bubble:\n{text}", schema))["translation"]
        except Exception as e:
            print(f"    could not translate '{text}': {e}")
            return ""


# --------------------------------------------------------------------------- 4. cleaning

def inner_mask(b: Bubble) -> np.ndarray:
    """Bubble mask shrunk a bit so we never paint over the bubble outline."""
    x1, y1, x2, y2 = b.box
    k = max(3, int(min(x2 - x1, y2 - y1) * 0.04))
    m = b.mask.astype(np.uint8)
    return cv2.erode(m, np.ones((k, k), np.uint8)) > 0


def clean_bubble(img: np.ndarray, b: Bubble) -> tuple[np.ndarray, np.ndarray]:
    """Repaint text pixels with the bubble's background colour. Returns (bg_bgr, inner_mask)."""
    inner = inner_mask(b)
    if inner.sum() == 0:
        return np.array([255, 255, 255]), inner
    pixels = img[inner]
    bg = np.median(pixels, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    bg_gray = int(cv2.cvtColor(bg.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
    text = (np.abs(gray - bg_gray) > 50) & inner
    text = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2) > 0
    img[text & inner] = bg
    return bg, inner


# --------------------------------------------------------------------------- 5. typesetting

class Typesetter:
    def __init__(self, font_path: str, uppercase: bool = False, min_size: int = 10):
        self.font_path = font_path
        self.uppercase = uppercase
        self.min_size = min_size
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    @staticmethod
    def wrap(text: str, font, max_w: float) -> list[str]:
        lines, cur = [], ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if not cur or font.getlength(trial) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def layout(self, text: str, region: np.ndarray, max_size: int):
        """Find the biggest font + wrapping whose every line rectangle lies inside `region` (bool mask)."""
        h, w = region.shape
        dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)  # most "central" point of the bubble
        ys, xs = np.nonzero(region)
        bw = xs.max() - xs.min() + 1
        for size in range(max_size, self.min_size - 1, -1):
            font = self.font(size)
            asc, desc = font.getmetrics()
            line_h = int((asc + desc) * 1.05)
            for frac in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
                lines = self.wrap(text, font, bw * frac)
                block_h = line_h * len(lines)
                top = int(cy - block_h / 2)
                placed, ok = [], True
                for i, line in enumerate(lines):
                    lw = int(font.getlength(line)) + 2
                    lx, ly = int(cx - lw / 2), top + i * line_h
                    if lx < 0 or ly < 0 or lx + lw > w or ly + line_h > h or \
                            not region[ly:ly + line_h, lx:lx + lw].all():
                        ok = False
                        break
                    placed.append((lx, ly, line))
                if ok:
                    return font, placed
        # nothing fits: smallest size, centred, best effort
        font = self.font(self.min_size)
        asc, desc = font.getmetrics()
        line_h = int((asc + desc) * 1.05)
        lines = self.wrap(text, font, bw * 0.9)
        top = int(cy - line_h * len(lines) / 2)
        return font, [(int(cx - font.getlength(l) / 2), top + i * line_h, l) for i, l in enumerate(lines)]

    def draw(self, pil: Image.Image, b: Bubble, region: np.ndarray, bg: np.ndarray, page_h: int):
        text = b.en.upper() if self.uppercase else b.en
        if not text:
            return
        x1, y1, x2, y2 = b.box
        local = region[y1:y2, x1:x2]
        if local.sum() == 0:
            return
        max_size = max(self.min_size, min(int(page_h / 28), int((y2 - y1) / 2)))
        font, placed = self.layout(text, local, max_size)
        bg_lum = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
        fill = (0, 0, 0) if bg_lum >= 128 else (255, 255, 255)
        stroke = tuple(int(c) for c in bg[::-1])  # BGR -> RGB
        d = ImageDraw.Draw(pil)
        sw = max(1, font.size // 14)
        for lx, ly, line in placed:
            d.text((x1 + lx, y1 + ly), line, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke)


# --------------------------------------------------------------------------- main loop

def debug_overlay(img: np.ndarray, bubbles: list[Bubble]) -> np.ndarray:
    out = img.copy()
    for i, b in enumerate(bubbles):
        color = (0, 180, 255)
        out[b.mask] = (0.6 * out[b.mask] + 0.4 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = b.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, str(i + 1), (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


class Pipeline:
    """Loads the GPU models once; translate many pages with it (used by the CLI and the web UI)."""

    def __init__(self, conf: float = 0.4):
        self.det = BubbleDetector(conf)
        self.ocr = OCR()

    def translate_image(self, img: np.ndarray, tl: Translator, ts: Typesetter, log=print):
        """img: BGR array (not modified). Returns (translated BGR image, bubbles)."""
        img = img.copy()
        bubbles = self.det.detect(img)
        log(f"  {len(bubbles)} bubbles found")
        for b in bubbles:
            b.ja = self.ocr.read(img, b)
        todo = [b for b in bubbles if JP_CHARS.search(b.ja) or re.fullmatch(r"[…\.!?！？・ー〜]+", b.ja or "x")]
        for b, en in zip(todo, tl.translate_page([b.ja for b in todo])):
            b.en = en
            log(f"    {b.ja}  ->  {en}")

        regions = [(b, *clean_bubble(img, b)) for b in todo if b.en]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for b, bg, inner in regions:
            ts.draw(pil, b, inner, bg, img.shape[0])
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), bubbles

    def process_file(self, path: Path, out_dir: Path, tl: Translator, ts: Typesetter,
                     debug: bool = False, log=print) -> Path:
        img = load_image(path)
        out, bubbles = self.translate_image(img, tl, ts, log)
        if debug:
            save_image(out_dir / f"{path.stem}_debug.png", debug_overlay(img, bubbles))
        out_path = out_dir / f"{path.stem}{path.suffix if path.suffix.lower() != '.webp' else '.png'}"
        save_image(out_path, out)
        sidecar = [{"box": list(b.box), "ja": b.ja, "en": b.en} for b in bubbles]
        (out_dir / f"{path.stem}.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        log(f"  saved {out_path}")
        return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Locally translate manga pages (Japanese -> English).")
    ap.add_argument("input", type=Path, help="an image file or a folder of pages")
    ap.add_argument("-o", "--output", type=Path, help="output folder (default: <input>_en)")
    ap.add_argument("--model", default="qwen3-swallow:8b", help="Ollama model name")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--font", default="C:/Windows/Fonts/comicbd.ttf", help="TTF/OTF font for English text")
    ap.add_argument("--uppercase", action="store_true", help="typeset in ALL CAPS (classic comic style)")
    ap.add_argument("--conf", type=float, default=0.4, help="bubble detection confidence threshold")
    ap.add_argument("--min-font", type=int, default=10)
    ap.add_argument("--debug", action="store_true", help="also save an image showing detected bubbles")
    args = ap.parse_args()

    if args.input.is_dir():
        pages = sorted((p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)
        out_dir = args.output or args.input.with_name(args.input.name + "_en")
    elif args.input.is_file():
        pages = [args.input]
        out_dir = args.output or args.input.parent / (args.input.parent.name + "_en")
    else:
        sys.exit(f"input not found: {args.input}")
    if not pages:
        sys.exit("no images found")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        requests.get(f"{args.ollama_url}/api/tags", timeout=5).raise_for_status()
    except Exception:
        sys.exit(f"Ollama is not reachable at {args.ollama_url} - start it with `ollama serve`.")

    print("loading models...")
    pipe = Pipeline(args.conf)
    tl = Translator(args.model, args.ollama_url)
    ts = Typesetter(args.font, args.uppercase, args.min_font)

    for i, page in enumerate(pages, 1):
        print(f"[{i}/{len(pages)}] {page.name}")
        try:
            pipe.process_file(page, out_dir, tl, ts, args.debug)
        except Exception as e:
            print(f"  FAILED: {e}")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
