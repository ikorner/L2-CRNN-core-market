"""Тесты TorchOcr и контракта OcrClient (задача 4.2, spec ocr-reading).

Покрытие:
- контракт read(region_img) -> (text, confidence): grayscale и BGR-входы;
- torch/модель загружаются один раз (счётчик загрузок при повторных read);
- авто-детект устройства cpu/cuda (инъекция доступности, оба пути);
- CPU-инференс на фейковом чекпоинте (tests/fixtures/ocr_vendor/fake_crnn.pt)
  детерминирован (exact, по аналогии с guard-тестами 4.1);
- точка подмены: фабрика create_ocr_client возвращает иную реализацию
  по аргументу; RemoteOcr — заглушка с понятным NotImplementedError;
- отсутствующий файл чекпоинта — осмысленная ошибка (FileNotFoundError);
- модуль контракта импортируется без torch (подмена без тяжёлой зависимости).

Confidence — средняя max-softmax по таймстепам (round 4), формула
оригинального autolabel.py; pin тестом на зафиксированном числе.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from l2core.adapters.ocr import (
    OcrClient,
    RemoteOcr,
    TorchOcr,
    _resolve_device,
    create_ocr_client,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ocr_vendor"
FAKE_CKPT = FIXTURES / "fake_crnn.pt"

# Зафиксированный вывод fake-чекпоинта на эталонном входе (замерено
# на torch 2.10.0+cpu, детерминировано — CPU-инференс, eval-режим).
EXPECTED_TEXT = json.loads((FIXTURES / "model_case.json").read_text("utf-8"))["expected_text"]
EXPECTED_CONFIDENCE = 0.094  # средняя max-softmax по таймстепам, round 4 (рандомные веса фейка)


def _region_bgr() -> np.ndarray:
    """Эталонный вход model_in.npy (grayscale) в BGR-представлении."""
    gray = np.load(FIXTURES / "model_in.npy")
    assert gray.ndim == 2
    return np.stack([gray, gray, gray], axis=-1)  # BGR == gray на всех каналах


def test_protocol_conformance():
    # TorchOcr, RemoteOcr и произвальный фейк удовлетворяют контракту;
    # runtime_checkable — точка подмены работает без наследования.
    class FakeOcr:
        def read(self, region_img):
            return ("text", 1.0)

    assert isinstance(FakeOcr(), OcrClient)
    assert isinstance(RemoteOcr(), OcrClient)
    assert isinstance(TorchOcr(FAKE_CKPT, device="cpu"), OcrClient)


def test_module_imports_without_torch():
    # Контракт подменяется без тяжёлой зависимости: импорт модуля не тянет torch.
    code = (
        "import sys; import l2core.adapters.ocr; "
        "assert 'torch' not in sys.modules, 'модуль контракта импортнул torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_read_grayscale_region():
    ocr = TorchOcr(FAKE_CKPT, device="cpu")
    img = np.load(FIXTURES / "model_in.npy")
    text, conf = ocr.read(img)
    assert text == EXPECTED_TEXT
    assert conf == EXPECTED_CONFIDENCE
    assert 0.0 <= conf <= 1.0


def test_read_bgr_region_same_result():
    # Контракт: вход — numpy-регион BGR; BGR (gray на всех каналах) == grayscale.
    ocr = TorchOcr(FAKE_CKPT, device="cpu")
    gray = np.load(FIXTURES / "model_in.npy")
    text_gray, conf_gray = ocr.read(gray)
    text_bgr, conf_bgr = ocr.read(_region_bgr())
    assert (text_bgr, conf_bgr) == (text_gray, conf_gray)


def test_read_deterministic_across_calls():
    ocr = TorchOcr(FAKE_CKPT, device="cpu")
    first = ocr.read(np.load(FIXTURES / "model_in.npy"))
    for _ in range(3):
        assert ocr.read(np.load(FIXTURES / "model_in.npy")) == first


def test_model_loaded_once(monkeypatch):
    # Модель загружается один раз при создании; повторные read не перегружают.
    # Считаем вызовы torch.load вендоренным загрузчиком: инъекция loader'ом
    # потребовала бы импорта ocr_vendor в тест — guard R13 это запрещает.
    torch = _torch()
    calls = []
    real_load = torch.load

    def counting_load(*args, **kwargs):
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    ocr = TorchOcr(FAKE_CKPT, device="cpu")
    for _ in range(3):
        ocr.read(np.load(FIXTURES / "model_in.npy"))
    assert len(calls) == 1


def test_device_autodetect_cpu_when_no_gpu():
    # Инъекция доступности: GPU нет -> авто-детект выбирает CPU (сценарий spec).
    device = _resolve_device(_torch(), cuda_available=lambda: False)
    assert device.type == "cpu"


def test_device_autodetect_cuda_when_available():
    # Инъекция доступности: GPU есть -> авто-детект выбирает CUDA.
    # Конструирование torch.device("cuda") не требует физического GPU.
    device = _resolve_device(_torch(), cuda_available=lambda: True)
    assert device.type == "cuda"


def test_device_autodetect_default_on_this_machine():
    # Без инъекции: на CPU-only окружении авто-детект падает на CPU без ошибок,
    # инференс работает (сценарий «Загрузка на CPU» из spec ocr-reading).
    ocr = TorchOcr(FAKE_CKPT)
    assert ocr.device.type == "cpu"
    text, conf = ocr.read(np.load(FIXTURES / "model_in.npy"))
    assert text == EXPECTED_TEXT


def test_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError, match="fake_missing.pt"):
        TorchOcr(FIXTURES / "fake_missing.pt", device="cpu")


def test_factory_returns_torch_by_default():
    ocr = create_ocr_client("torch", model_path=FAKE_CKPT, device="cpu")
    assert isinstance(ocr, TorchOcr)
    assert isinstance(ocr, OcrClient)


def test_factory_substitution_point_remote():
    # Точка подмены: аргумент фабрики выбирает иную реализацию без правки ядра.
    ocr = create_ocr_client("remote", endpoint="http://127.0.0.1:9000")
    assert isinstance(ocr, RemoteOcr)
    assert isinstance(ocr, OcrClient)
    with pytest.raises(NotImplementedError, match="RemoteOcr"):
        ocr.read(np.zeros((10, 10), dtype=np.uint8))


def test_factory_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown"):
        create_ocr_client("unknown")


def test_factory_torch_requires_model_path():
    with pytest.raises(ValueError, match="model_path"):
        create_ocr_client("torch")


def _torch():
    import torch

    return torch
