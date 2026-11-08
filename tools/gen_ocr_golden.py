"""Генератор golden-набора OCR: нарезка эталонных кропов из сырья (task 4.6).

Читает индекс assets/ocr_golden/index.yaml (источник истины: id, скриншот,
rect, ожидаемый текст) и пишет кропы <id>.png нарезкой rect'ом из
base_screenshot/scr/<source>.png. Перезаписывает существующие файлы —
golden-кропы пересоздаются из сырья, ручные правки в них не предполагаются
(тест test_crop_matches_rect_and_source следит за совпадением с индексом).

Запуск вручную (вне CI), из корня репо:

    python tools/gen_ocr_golden.py

Критерий попадания карточки в набор — в шапке index.yaml (ruling R4):
текст читается человеком со скриншота без сомнений; description меты
ground truth не является.
"""
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "assets" / "ocr_golden"
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"


def main() -> None:
    index = yaml.safe_load((GOLDEN_DIR / "index.yaml").read_text(encoding="utf-8"))
    written = []
    for card in index["cards"]:
        source_path = SCR_DIR / f"{card['source']}.png"
        img = cv2.imread(str(source_path))
        assert img is not None, f"не читается скриншот-источник: {source_path}"
        x, y, w, h = card["rect"]
        crop = img[y : y + h, x : x + w]
        out_path = GOLDEN_DIR / card["file"]
        assert cv2.imwrite(str(out_path), crop), f"не записан кроп: {out_path}"
        written.append(f"{card['id']}: {card['file']} ({w}x{h}) <- {card['source']}")
    print(f"golden-набор пересоздан: {len(written)} кропов в {GOLDEN_DIR}")
    for line in written:
        print(" ", line)


if __name__ == "__main__":
    main()
