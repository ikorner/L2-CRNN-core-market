"""Guard-тесты вендоренного OCR-кода (задача 4.1, spec ocr-reading).

Сравнивают вендоренную копию (l2core/adapters/ocr_vendor) с эталонными
векторами, сгенерированными ОРИГИНАЛЬНЫМ кодом L2-CRNN-MODEL-TRAIN
(коммит e720a8d, см. шапку вендоренных модулей). Эталоны закоммичены в
tests/fixtures/ocr_vendor/ — для запуска guard-тестов оригинальный
репозиторий НЕ нужен.

Сравнение exact (np.array_equal / == на строках): эталоны сгенерированы
тем же окружением (cv2/numpy/torch CPU), операции детерминированы
(cv2.INTER_AREA, argmax/схлопывание CTC, однократный forward LSTM на CPU).
Обоснование — отчёт task-4.1-report.md.

Перегенерация эталонов (вручную, вне CI):
    python tools/gen_ocr_vendor_fixtures.py
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

from l2core.adapters.ocr_vendor import (
    BLANK,
    CRNN,
    IMG_H,
    IMG_W,
    ctc_greedy_decode,
    load_model,
    preprocess,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ocr_vendor"


def _meta(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_constants_frozen():
    # Геометрия входа зашита в чекпоинты — guard на константы.
    assert (IMG_H, IMG_W, BLANK) == (32, 320, 0)


def test_preprocess_reference():
    # Эталон: preprocess на фиксированных входах == выход оригинала, байт-в-байт.
    with np.load(FIXTURES / "preprocess_cases.npz") as npz:
        names = sorted(n for n in npz.files if n.startswith("in_plain_"))
        assert len(names) >= 3
        for name in names:
            got = preprocess(npz[name])
            expected = npz["out_" + name[len("in_"):]]
            assert got.shape == (1, IMG_H, IMG_W), name
            assert got.dtype == np.float32, name
            assert np.array_equal(got, expected), \
                f"{name}: вендоренный preprocess отклонился от эталона"


def test_preprocess_augmented_reference():
    # Тот же путь, но с rng (тренировочные аугментации) — seed фиксирован.
    with np.load(FIXTURES / "preprocess_cases.npz") as npz:
        got = preprocess(npz["in_aug"], rng=random.Random(int(npz["aug_seed"])))
        assert np.array_equal(got, npz["out_aug"]), \
            "preprocess(rng) отклонился от эталона"


def test_ctc_greedy_decode_reference():
    # Эталон: фиксированные logits -> строки (схлопывание blank + дедуп соседей).
    meta = _meta("decode_texts.json")
    with np.load(FIXTURES / "decode_cases.npz") as npz:
        names = sorted(n for n in npz.files if n.startswith("logits_"))
        assert names and len(names) == len(meta["texts"])
        for name, expected in zip(names, meta["texts"]):
            logits = torch.from_numpy(npz[name])
            assert ctc_greedy_decode(logits, list(meta["alphabet"])) == expected, name


def test_load_model_reference():
    # Эталон: фейковый чекпоинт (в формате crnn_best.pt) грузится вендоренным
    # загрузчиком; forward вендоренной CRNN на эталонном входе даёт эталонную
    # строку. CPU-режим: forward выполняется без падения (GPU может отсутствовать).
    meta = _meta("model_case.json")
    model, alphabet, device = load_model(FIXTURES / "fake_crnn.pt")
    assert isinstance(model, CRNN)
    assert alphabet == list(meta["alphabet"])
    assert isinstance(device, torch.device)
    img = np.load(FIXTURES / "model_in.npy")
    with torch.no_grad():
        logits = model(torch.from_numpy(preprocess(img)[None]).to(device))
    assert ctc_greedy_decode(logits[0], alphabet) == meta["expected_text"]
