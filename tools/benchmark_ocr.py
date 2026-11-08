"""OCR-бенчмарк на golden-наборе (task 4.6, spec ocr-reading).

Гоняет эталонные кропы assets/ocr_golden/ через OcrClient (TorchOcr) на
заданных устройствах и выводит точность (доля точных совпадений текста) и
время инференса. Ручной инструмент (вне CI), по аналогии с генераторами
фикстур — запуск из корня репо:

    python tools/benchmark_ocr.py                 # cpu + cuda (если есть)
    python tools/benchmark_ocr.py --devices cpu   # только CPU
    python tools/benchmark_ocr.py --model /путь/crnn_best.pt --repeat 20

Честность окружения (ruling R6): в CPU-only среде GPU-устройство не
притворяется исполненным — оно помечается «skipped» с причиной, и это
явно видно в отчёте и выходном словаре. Числа имеют смысл только с
реальным чекпоинтом: fake-чекпоинт (tests/fixtures/ocr_vendor/) пригоден
для smoke-прогона механики (tests/test_ocr_golden.py).

При отсутствии чекпоинта — понятная ошибка и код выхода 2 (crnn_best.pt
хранится вне git, путь — paths.model в config.yaml, по умолчанию рядом
с корнем репо). Контроль чтения числовых значений (цены, «0 из N», адена)
на доснятых кропах — на машине владельца (docs/OWNER_QUESTIONS.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "assets" / "ocr_golden"
DEFAULT_INDEX = GOLDEN_DIR / "index.yaml"
DEFAULT_MODEL = REPO_ROOT / "crnn_best.pt"

import torch  # noqa: E402  (инструмент, не ядро: тяжёлая зависимость допустима)

sys.path.insert(0, str(REPO_ROOT))  # запуск как «python tools/benchmark_ocr.py»

from l2core.adapters.ocr import TorchOcr  # noqa: E402


def load_cards(index_path: Path) -> list[dict]:
    """Индекс golden-набора -> список карточек с абсолютными путями кропов."""
    index = yaml.safe_load(Path(index_path).read_text(encoding="utf-8"))
    golden_dir = Path(index_path).parent
    cards = []
    for card in index["cards"]:
        cards.append({**card, "path": str(golden_dir / card["file"])})
    return cards


def benchmark_client(ocr, cards: list[dict], repeat: int) -> dict:
    """Прогон всех кропов через OcrClient: repeat проходов по набору.

    Точность — доля кропов, прочитанных эталонным текстом (exact match).
    Время — среднее по кропам за вычетом первого (прогрев) прохода.
    """
    images = [cv2.imread(c["path"]) for c in cards]
    assert all(img is not None for img in images), "кроп golden-набора не читается"
    per_card = [{"id": c["id"], "text": None, "confidence": None, "ok": False,
                 "seconds": 0.0} for c in cards]
    passes = max(1, repeat)
    for r in range(passes):
        start = time.perf_counter()
        for i, img in enumerate(images):
            text, confidence = ocr.read(img)
            if r == passes - 1:  # фиксируем вывод последнего прохода
                per_card[i]["text"] = text
                per_card[i]["confidence"] = confidence
                per_card[i]["ok"] = text == cards[i]["text"]
        elapsed = time.perf_counter() - start
        if r > 0:  # прогрев (первый проход) не входит в среднее
            for i in range(len(cards)):
                per_card[i]["seconds"] += elapsed / len(cards) / (passes - 1)
    ok = sum(1 for p in per_card if p["ok"])
    return {
        "status": "ok",
        "n_cards": len(cards),
        "accuracy": ok / len(cards),
        "mean_seconds": sum(p["seconds"] for p in per_card) / len(cards),
        "per_card": per_card,
    }


def run_benchmark(
    model_path: Path,
    devices: list[str],
    index_path: Path = DEFAULT_INDEX,
    repeat: int = 10,
) -> dict:
    """Бенчмарк по устройствам: {device: результат | {status: skipped}}."""
    cards = load_cards(index_path)
    results = {}
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            results[device] = {
                "status": "skipped",
                "reason": "torch.cuda.is_available() == False: GPU-путь кода "
                "существует, но не исполнен в этой среде (ruling R6); "
                "прогон на машине с GPU",
            }
            continue
        ocr = TorchOcr(model_path, device=torch.device(device))
        results[device] = benchmark_client(ocr, cards, repeat)
    return results


def format_report(results: dict, model_path: Path) -> str:
    lines = [f"OCR-бенчмарк golden-набора", f"модель: {model_path}", ""]
    for device, res in results.items():
        if res["status"] == "skipped":
            lines.append(f"[{device}] НЕ ИСПОЛНЕН (skipped): {res['reason']}")
            continue
        lines.append(
            f"[{device}] точность: {res['accuracy']:.1%} "
            f"({sum(1 for p in res['per_card'] if p['ok'])} из {res['n_cards']}), "
            f"среднее время: {res['mean_seconds'] * 1000:.2f} мс/кроп"
        )
        for p in res["per_card"]:
            mark = "ok" if p["ok"] else "XX"
            lines.append(
                f"  {mark} {p['id']}: прочитано {p['text']!r} "
                f"(conf={p['confidence']})"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="чекпоинт CRNN (по умолчанию: crnn_best.pt рядом с корнем репо)")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"],
                        help="устройства инференса (по умолчанию: cpu cuda)")
    parser.add_argument("--repeat", type=int, default=10,
                        help="число проходов по набору для усреднения времени")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                        help="индекс golden-набора")
    parser.add_argument("--json", action="store_true",
                        help="вывести сырой JSON результатов (для архивации)")
    args = parser.parse_args(argv)

    if not args.model.is_file():
        print(
            f"ОШИБКА: чекпоинт CRNN не найден: {args.model}\n"
            "crnn_best.pt хранится вне git — положите файл рядом с корнем "
            "репо (paths.model в config.yaml) или укажите путь через --model. "
            "Без реальной модели бенчмарк не имеет смысла (fake-чекпоинт — "
            "только для smoke-теста механики).",
            file=sys.stderr,
        )
        return 2

    results = run_benchmark(args.model, args.devices, args.index, args.repeat)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    else:
        print(format_report(results, args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
