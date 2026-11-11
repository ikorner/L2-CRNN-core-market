"""build_dataset.py — Lineage 2 «Обзор лавок» screenshots → CRNN OCR dataset.

Input: BMP screenshots with the shop overview window open.
Output: dataset/images/*.png + labels.csv (image_path,text) + cells.csv metadata.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

Cell = namedtuple("Cell", ["key", "image", "meta"])

# --- Relative geometry inside the window frame (origin = first interior pixel) ---
# Reference: 1136x833 screenshots, frame at x56..1059 y70..740.
CONTENT_X0 = 14          # left edge of table content (abs 71)
COL_SEPS = (72, 536, 764, 912, 966)  # abs 129, 593, 821, 969, 1023
CONTENT_X1 = 986         # right edge before scrollbar (abs 1043)
HEADER_Y0, HEADER_Y1 = 242, 260      # abs 313..331
ROW_Y0 = 261             # first data row top (abs 332)
ROW_PITCH = 56
ROW_H = 55
TABLE_BOTTOM = 660     # abs 731: bottom border of the table area
MAX_ROWS = 8

COLUMNS = ("type", "name", "price", "seller", "deal")

# zones for text extraction inside a row (relative to window interior)
NAME_TEXT_X = (119, 534)   # exclude item icon (abs ~139..172)
PRICE_TEXT_X = (556, 736)  # exclude seal icon (abs ~594-612) and gold scroll icon (abs ~799)


def detect_window(gray: np.ndarray):
    """Find the shop window frame. Returns (x, y, w, h) or None."""
    bright = ((gray > 110) & (gray < 210)).astype(np.uint8)
    h, w = bright.shape
    hlines = []
    for y in range(h):
        row = bright[y]
        d = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        if len(starts):
            lengths = ends - starts
            i = int(lengths.argmax())
            if lengths[i] > 700:
                hlines.append((y, int(starts[i]), int(ends[i])))
    for i, (yt, xt0, xt1) in enumerate(hlines):
        for yb, xb0, xb1 in hlines[i + 1:]:
            if yb - yt < 500:
                continue
            if abs(xb0 - xt0) > 15 or abs((xb1 - xb0) - (xt1 - xt0)) > 30:
                continue
            col_counts = bright[yt:yb + 1].sum(axis=0)
            span = yb - yt + 1
            left_range = col_counts[max(0, xt0 - 2):xt0 + 12]
            li = int(np.argmax(left_range))
            if left_range[li] < 0.9 * span:
                continue
            xl = max(0, xt0 - 2) + li
            right_range = col_counts[xb1 - 12:min(w, xb1 + 2)]
            ri = int(np.argmax(right_range))
            if right_range[ri] < 0.9 * span:
                continue
            xr = xb1 - 12 + ri
            return xl, yt, xr - xl + 1, yb - yt + 1
    return None


def _column_ranges():
    bounds = [CONTENT_X0] + list(COL_SEPS) + [CONTENT_X1]
    return {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(COLUMNS)}


def _row_is_empty(win_gray, y0, y1):
    zone = win_gray[y0:y1, NAME_TEXT_X[0]:NAME_TEXT_X[1]]
    bg = np.median(zone)
    return int((zone > bg + 25).sum()) < 30


def rows_and_columns(win_gray: np.ndarray, origin=(0, 0)) -> dict:
    """Segment the table: header + non-empty rows.

    Coordinates in header/rows/cols are relative to win_gray; origin is the
    (x, y) of win_gray's top-left pixel in the full screenshot.
    """
    cols = _column_ranges()
    header = {name: (x0, HEADER_Y0, x1, HEADER_Y1) for name, (x0, x1) in cols.items()}
    h = win_gray.shape[0]
    rows = []
    for k in range(MAX_ROWS):
        y0 = ROW_Y0 + k * ROW_PITCH
        y1 = y0 + ROW_H
        if y1 > min(TABLE_BOTTOM, h):
            break
        if not _row_is_empty(win_gray, y0, y1):
            rows.append((y0, y1))
    return {"header": header, "rows": rows, "cols": cols, "origin": origin}


def _text_mask(crop: np.ndarray) -> np.ndarray:
    """Binary text mask: 255 where a pixel deviates from the background color."""
    bg_color = np.median(crop.reshape(-1, 3), axis=0)
    return (np.abs(crop.astype(np.int16) - bg_color).max(axis=2) > 25).astype(np.uint8) * 255


def _trim(crop: np.ndarray, pad: int = 4):
    """Crop to the text bounding box plus padding. Returns (crop, dx, dy).

    Mask is color-based (max per-channel deviation from the background color),
    so colored text (blue/red/green) is detected as reliably as white.
    """
    mask = _text_mask(crop)
    if not mask.any():
        return crop, 0, 0
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(crop.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(crop.shape[1], xs.max() + 1 + pad)
    return crop[y0:y1, x0:x1], x0, y0


def trim_to_text(crop: np.ndarray, pad: int = 4) -> np.ndarray:
    """Crop to the text bounding box plus padding. Falls back to the input."""
    return _trim(crop, pad)[0]


def _cell_key(image: np.ndarray) -> str:
    """Shape key: hash of the binary text mask.

    Same text on alternating row backgrounds or in different colors hashes
    identically, so each unique text is labeled once. The original pixels are
    kept in Cell.image and in meta['pixel_key'] for the dataset output.
    """
    return hashlib.sha1(_text_mask(image).tobytes()).hexdigest()


def _pixel_key(image: np.ndarray) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()


def extract_cells(img: np.ndarray, structure: dict) -> list:
    """Cut all labelable cells: 5 header labels + rows x 5 columns.

    Returns a list of Cell(key, image, meta); key deduplicates identical crops.
    meta: col, row (-1 = header), rect (absolute x0,y0,x1,y1).
    """
    ox, oy = structure["origin"]
    cells = []

    def cut(col, row, rect):
        x0, y0, x1, y1 = rect
        dx0 = dy0 = 0
        crop = img[oy + y0 : oy + y1, ox + x0 : ox + x1]
        if col == "name":
            tx0, tx1 = NAME_TEXT_X
            dx0 = max(0, tx0 - x0)
            crop = crop[:, dx0 : tx1 - x0]
        elif col == "price":
            tx0, tx1 = PRICE_TEXT_X
            dx0 = max(0, tx0 - x0)
            crop = crop[:, dx0 : tx1 - x0]
        crop, tx, ty = _trim(crop)
        abs_rect = (int(ox + x0 + dx0 + tx), int(oy + y0 + ty),
                    int(ox + x0 + dx0 + tx + crop.shape[1]), int(oy + y0 + ty + crop.shape[0]))
        cells.append(Cell(_cell_key(crop), crop,
                          {"col": col, "row": row, "rect": abs_rect, "pixel_key": _pixel_key(crop)}))

    for col, rect in structure["header"].items():
        cut(col, -1, rect)
    for row, (y0, y1) in enumerate(structure["rows"]):
        for col, (x0, x1) in structure["cols"].items():
            cut(col, row, (x0, y0, x1, y1))
    return cells


# --- Labeling progress ---

PROGRESS_FILE = ".labels_progress.json"


def load_progress(dataset_dir) -> dict:
    path = Path(dataset_dir) / PROGRESS_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_label(progress: dict, key: str, text):
    progress[key] = {"text": text, "state": "labeled" if text else "skipped"}
    return progress


def _flush_progress(progress: dict, dataset_dir) -> None:
    path = Path(dataset_dir) / PROGRESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=1), encoding="utf-8")


def migrate_progress(cells: list, progress: dict) -> dict:
    """Re-key progress from old pixel hashes to shape keys (labels preserved)."""
    by_pixel = {_pixel_key(c.image): c.key for c in cells}
    migrated = {}
    for key, entry in progress.items():
        if key in by_pixel:
            migrated.setdefault(by_pixel[key], entry)
        else:
            migrated.setdefault(key, entry)
    return migrated


# --- Dataset assembly ---

def build_dataset(cells: list, progress: dict, dataset_dir) -> None:
    """Write images/*.png, labels.csv and cells.csv.

    images/: one PNG per unique pixel crop (all visual variants kept).
    labels.csv: one row per unique pixel crop; text comes from the shape key,
    so labeling a text once covers all its background/color variants.
    cells.csv: every cell occurrence with both keys and metadata.
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    for c in cells:
        variants.setdefault(c.meta.get("pixel_key", c.key), (c.image, c.key))
    with open(dataset_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "text"])
        for pixel_key, (img, shape_key) in variants.items():
            entry = progress.get(shape_key)
            if entry and entry["state"] == "labeled":
                cv2.imwrite(str(images_dir / f"{pixel_key}.png"), img)
                w.writerow([f"images/{pixel_key}.png", entry["text"]])
    with open(dataset_dir / "cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "pixel_key", "shot", "row", "col", "rect"])
        for c in cells:
            w.writerow([c.key, c.meta.get("pixel_key", ""), c.meta.get("shot", ""),
                        c.meta["row"], c.meta["col"], json.dumps(c.meta["rect"])])


# --- Interactive labeling UI ---

def label_cells(cells: list, dataset_dir) -> dict:
    """Tkinter labeling loop. Returns progress dict."""
    import tkinter as tk
    from PIL import Image, ImageTk

    dataset_dir = Path(dataset_dir)
    unique = {}
    for c in cells:
        unique.setdefault(c.key, c)
    items = list(unique.items())
    progress = load_progress(dataset_dir)
    state = {"i": 0}

    def pending_indices():
        return [i for i, (k, _) in enumerate(items)
                if progress.get(k, {}).get("state") not in ("labeled", "skipped")]

    root = tk.Tk()
    root.title("Обзор лавок — разметка датасета")
    canvas = tk.Canvas(root, width=900, height=200, bg="#333")
    canvas.pack(fill="both", expand=True)
    info = tk.Label(root, text="", font=("Helvetica", 12))
    info.pack()
    entry = tk.Entry(root, font=("Helvetica", 18), width=50)
    entry.pack(pady=8)
    status = tk.Label(root, text="", font=("Helvetica", 10), fg="#666")
    status.pack()

    photo = [None]

    def show(idx):
        key, cell = items[idx]
        state["i"] = idx
        img = Image.fromarray(cv2.cvtColor(cell.image, cv2.COLOR_BGR2RGB))
        scale = max(2, min(6, 860 // max(1, img.width)))
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        photo[0] = ImageTk.PhotoImage(img)
        canvas.config(width=img.width + 20, height=img.height + 20)
        canvas.create_image(10, 10, anchor="nw", image=photo[0])
        info.config(text=f"[{cell.meta.get('col','?')} row={cell.meta.get('row','?')}]  "
                         f"{idx + 1}/{len(items)}")
        known = progress.get(key, {})
        entry.delete(0, "end")
        if known.get("state") == "labeled":
            entry.insert(0, known["text"])
        status.config(text="Enter — сохранить | Esc — пропустить | ← — назад")
        entry.focus_set()

    def save_and_next(event=None):
        key, _ = items[state["i"]]
        text = entry.get().strip()
        save_label(progress, key, text if text else None)
        _flush_progress(progress, dataset_dir)
        nxt = [i for i in pending_indices() if i > state["i"]]
        if nxt:
            show(nxt[0])
        else:
            rest = pending_indices()
            if rest:
                show(rest[0])
            else:
                root.destroy()

    def skip(event=None):
        save_and_next()

    def back(event=None):
        labeled = [i for i in range(len(items))
                   if progress.get(items[i][0], {}).get("state") in ("labeled", "skipped")]
        prev = [i for i in labeled if i < state["i"]]
        if prev:
            show(prev[-1])

    entry.bind("<Return>", save_and_next)
    root.bind("<Escape>", skip)
    root.bind("<Left>", back)

    pend = pending_indices()
    if not pend:
        root.destroy()
    else:
        show(pend[0])
        root.mainloop()
    return progress


# --- CLI ---

def collect_cells(input_dir: str) -> list:
    cells = []
    for bmp in sorted(Path(input_dir).glob("*.bmp")):
        img = cv2.imread(str(bmp))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        win = detect_window(gray)
        if win is None:
            print(f"пропуск {bmp.name}: окно «Обзор лавок» не найдено", file=sys.stderr)
            continue
        x, y, w, h = win
        structure = rows_and_columns(gray[y + 1 : y + h, x + 1 : x + w], origin=(x + 1, y + 1))
        shot_cells = extract_cells(img, structure)
        for c in shot_cells:
            c.meta["shot"] = bmp.stem
        cells.extend(shot_cells)
        print(f"{bmp.name}: {len(structure['rows'])} строк, {len(shot_cells)} ячеек")
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description="Скриншоты «Обзор лавок» → датасет для CRNN")
    ap.add_argument("input_dir", nargs="?", default="screenshots")
    ap.add_argument("--dataset", default="dataset")
    args = ap.parse_args(argv)

    cells = collect_cells(args.input_dir)
    if not cells:
        sys.exit("не найдено ни одного скриншота с окном «Обзор лавок»")
    progress = label_cells(cells, args.dataset)
    build_dataset(cells, progress, args.dataset)
    labeled = sum(1 for e in progress.values() if e["state"] == "labeled")
    print(f"готово: {labeled} размечено, датасет в {args.dataset}/")


if __name__ == "__main__":
    main()
