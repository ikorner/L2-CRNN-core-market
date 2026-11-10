"""Инструментальная досъёмка кадров в base_screenshot/scr/ (ручная, вне CI).

Захват одного кадра живого экрана (mss) + запись заглушки в meta.json
(«сырьё» датасета). Разметка rect'ов кропов — отдельно (задача 2.2/ручная).

Использование (игра запущена и видима):

    python tools/capture_frame.py <имя-кадра> "описание кадра"

Имя — без расширения (scr_nav_shop_sell_open). Кадр пишется в
base_screenshot/scr/<имя>.png; в meta.json добавляется запись
screenshots.<имя> с window «unknown» и пустым crops — разметчик
дозаполняет. Форматирование meta.json сохраняется (indent=2, ensure_ascii
False), существующие записи не трогаются.
"""

import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"
META_PATH = SCR_DIR / "meta.json"


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    name, description = argv[1], argv[2]
    if not name.replace("_", "").isalnum():
        print(f"имя кадра {name!r}: только буквы/цифры/подчёркивания")
        return 2

    from l2core.adapters.screen import MssScreen
    frame = MssScreen().grab()
    path = SCR_DIR / f"{name}.png"
    cv2.imwrite(str(path), frame)
    resolution = [int(frame.shape[1]), int(frame.shape[0])]

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if name in meta["screenshots"]:
        print(f"запись {name!r} уже есть в meta.json — кадр перезаписан, "
              f"мета не тронута (исправьте описание руками при нужде)")
    else:
        meta["screenshots"][name] = {
            "file": f"{name}.png",
            "window": "unknown",
            "state": None,
            "state_params": {},
            "resolution": resolution,
            "description": description,
            "crops": [],
        }
        META_PATH.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    print(f"кадр: {path} ({resolution[0]}x{resolution[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
