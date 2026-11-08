"""Вендоренная замороженная копия обучающего OCR-кода (задача 4.1).

Модули — байт-идентичные блоки из L2-CRNN-MODEL-TRAIN (см. шапки файлов,
guard-тесты tests/test_ocr_vendor.py на эталонных векторах). Правки
вендоренного кода запрещены (AGENTS.md, инвариант 5): изменения делать
в источнике, затем перевендоривать через tools/gen_ocr_vendor_fixtures.py.

Не входит сюда (другие задачи): обёртка OcrClient / инференс — задача 4.2
(TorchOcr в adapters), golden-набор assets/ocr_golden/ — задача 4.x бенчмарка.
"""
from l2core.adapters.ocr_vendor.crnn_train import (
    BLANK,
    CRNN,
    IMG_H,
    IMG_W,
    ctc_greedy_decode,
    preprocess,
)
from l2core.adapters.ocr_vendor.model_io import load_model

__all__ = [
    "BLANK",
    "CRNN",
    "IMG_H",
    "IMG_W",
    "ctc_greedy_decode",
    "load_model",
    "preprocess",
]
