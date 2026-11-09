#!/usr/bin/env python3
"""Генерация эталонных векторов guard-тестов вендоренного OCR-кода (задача 4.1).

Запускается ВРУЧНУЮ, вне CI, один раз (и после каждого обновления вендоринга
из нового коммита источника):

    python tools/gen_ocr_vendor_fixtures.py

Условия: окружение l2core (те же версии cv2/numpy/torch, что у guard-тестов),
исходный репозиторий L2-CRNN-MODEL-TRAIN доступен read-only (путь — константа
SOURCE_DIR ниже или переменная окружения OCR_VENDOR_SOURCE).

Детерминизм: входы синтетические с фиксированными seed, CPU, без меток
времени. Выход — tests/fixtures/ocr_vendor/: preprocess_cases.npz,
decode_cases.npz, decode_texts.json, fake_crnn.pt, model_in.npy,
model_case.json. Эталоны коммитятся в git; guard-тесты (tests/test_ocr_vendor.py)
сравнивают ВЕНДОРЕННЫЙ код с этими файлами и не требуют исходного репозитория.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

SOURCE_DIR = Path(os.environ.get("OCR_VENDOR_SOURCE", r"E:\L2-CRNN-MODEL-TRAIN"))
FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "ocr_vendor"
COMMIT = "e720a8df8078337740df5c3ad8dc2b1eb4387529"  # фиксируем источник эталонов

sys.path.insert(0, str(SOURCE_DIR))
import autolabel  # noqa: E402  (исходный загрузчик, train.py подтянется им)
import train  # noqa: E402  (исходный preprocess / ctc_greedy_decode / CRNN)


def _gradient(h: int, w: int) -> np.ndarray:
    return (np.arange(h * w) % 256).astype(np.uint8).reshape(h, w)


def _noise(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def gen_preprocess() -> None:
    rng = np.random.default_rng(42)
    inputs = {
        "plain_0": _gradient(12, 40),       # апскейл
        "plain_1": _noise(rng, 32, 320),    # ровно контейнер
        "plain_2": _noise(rng, 64, 20),     # сжатие по высоте
        "plain_3": _noise(rng, 10, 600),    # сильное сжатие по ширине
        "plain_4": _noise(rng, 18, 73),     # произвольный кроп
    }
    arrays: dict[str, np.ndarray] = {}
    for name, img in inputs.items():
        arrays[f"in_{name}"] = img
        arrays[f"out_{name}"] = train.preprocess(img)
    aug_seed = 20240501
    arrays["in_aug"] = _noise(rng, 20, 90)
    arrays["out_aug"] = train.preprocess(arrays["in_aug"],
                                         rng=random.Random(aug_seed))
    arrays["aug_seed"] = np.array(aug_seed, dtype=np.int64)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURES / "preprocess_cases.npz", **arrays)


def gen_decode() -> None:
    alphabet = ["a", "b", "c", "d"]
    # Последовательности argmax, покрывающие: дедуп соседей, blank-разделитель,
    # полностью пустой путь, подряд идущие разные классы.
    seqs = [
        [0, 1, 1, 0, 2, 3, 3, 3, 1],   # -> "abca"
        [0, 0, 0, 0],                  # -> ""
        [4, 3, 2, 1],                  # -> "dcba"
    ]
    rng = np.random.default_rng(7)
    tensors = {f"logits_{i}": _logits_from_seq(s, rng)
               for i, s in enumerate(seqs)}
    texts = [train.ctc_greedy_decode(tensors[f"logits_{i}"], alphabet)
             for i in range(len(seqs))]
    arrays = {k: t.numpy() for k, t in tensors.items()}
    np.savez(FIXTURES / "decode_cases.npz", **arrays)
    (FIXTURES / "decode_texts.json").write_text(
        json.dumps({"alphabet": "".join(alphabet), "texts": texts},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


def _logits_from_seq(seq: list[int],
                     rng: np.random.Generator | None = None) -> torch.Tensor:
    """Logits (T, C+1) с заданной последовательностью argmax."""
    if rng is None:
        base = np.zeros((len(seq), 5), dtype=np.float32)
    else:
        base = rng.normal(0, 1, size=(len(seq), 5)).astype(np.float32)
    for t, cls in enumerate(seq):
        base[t, cls] += 8.0
    return torch.from_numpy(base)


def gen_model() -> None:
    alphabet = list("0123456789 ")
    torch.manual_seed(7)
    model = train.CRNN(n_classes=len(alphabet)).eval()
    torch.save({
        "model": model.state_dict(),
        "alphabet": "".join(alphabet),
        "img_size": [train.IMG_H, train.IMG_W],
        "epoch": 1,
        "val_exact": 0.0,
    }, FIXTURES / "fake_crnn.pt")
    img = _noise(np.random.default_rng(11), 15, 50)
    np.save(FIXTURES / "model_in.npy", img)
    with torch.no_grad():
        logits = model(torch.from_numpy(train.preprocess(img)[None]))
    expected = train.ctc_greedy_decode(logits[0], alphabet)
    (FIXTURES / "model_case.json").write_text(
        json.dumps({"alphabet": "".join(alphabet), "expected_text": expected},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


def main() -> None:
    assert (SOURCE_DIR / ".git").exists(), f"нет репозитория: {SOURCE_DIR}"
    head = os.popen(f'git -C "{SOURCE_DIR}" rev-parse HEAD').read().strip()
    assert head == COMMIT, \
        f"источник сдвинулся: {head} (ожидался {COMMIT}) — обнови COMMIT и вендоринг"
    gen_preprocess()
    gen_decode()
    gen_model()
    print(f"эталоны записаны в {FIXTURES} (источник {COMMIT})")


if __name__ == "__main__":
    main()
