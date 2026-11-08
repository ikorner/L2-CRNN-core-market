"""OCR-адаптер: контракт OcrClient, TorchOcr (in-process инференс) и точка
подмены реализации (задача 4.2, spec ocr-reading).

Модуль импортируется БЕЗ torch: протокол OcrClient и заглушка RemoteOcr
доступны без тяжёлой зависимости (подмена in-process -> удалённый сервис
без правки ядра). torch и вендоренный код подтягиваются лениво при создании
TorchOcr. Препроцессинг и CTC-decode — только вендоренные
(l2core/adapters/ocr_vendor, инвариант 5).

Устройство: авто-детект cuda -> cpu (вендоренный load_model выбирает
mps/cpu по коду обучения на macOS — его выбор для ядра не используется,
модель переносится на устройство ядра явно).

Confidence — средняя max-softmax вероятность по таймстепам, round 4:
формула оригинального autolabel.py L2-CRNN-MODEL-TRAIN
(confidence_from_logits) — метрика совпадает с пайплайном автоматической
разметки, на которой калибруется порог ocr.confidence_threshold из конфига.

Грамматики (4.4), перечитывания при низком confidence (4.5) и бенчмарк
(4.6) здесь не реализуются (YAGNI); порог и N — уже в config.yaml
(l2core/config.py, OcrConfig).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OcrClient(Protocol):
    """Контракт чтения OCR-региона. Точка подмены: любой объект с read()."""

    def read(self, region_img: np.ndarray) -> tuple[str, float]:
        """BGR или grayscale uint8 регион -> (текст, confidence в [0, 1])."""
        ...


class RemoteOcr:
    """Заглушка удалённого OCR-сервиса (мультиклиент, design.md). Реализация
    — задача будущего; точка подмены уже работает через create_ocr_client."""

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint

    def read(self, region_img: np.ndarray) -> tuple[str, float]:
        raise NotImplementedError(
            "RemoteOcr не реализован: удалённый OCR-сервис — задача будущего "
            "(design.md); используйте TorchOcr"
        )


def _resolve_device(torch, cuda_available: Callable[[], bool] | None = None):
    """Авто-детект устройства: cuda, если доступен, иначе cpu.

    cuda_available — инъекция для тестов обоих путей без физического GPU.
    """
    if cuda_available is None:
        cuda_available = torch.cuda.is_available
    return torch.device("cuda" if cuda_available() else "cpu")


def _confidence_from_logits(logits_t_c) -> float:
    """Средняя max-softmax вероятность по таймстепам, round 4.

    Формула оригинального autolabel.py (L2-CRNN-MODEL-TRAIN): по всем
    таймстепам (не только по выбранному CTC-пути) — как в пайплайне
    автоматической разметки.
    """
    probs = logits_t_c.softmax(dim=-1).max(dim=-1).values
    return round(float(probs.mean()), 4)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """BGR/gray uint8 -> grayscale, как _to_gray в autolabel.py."""
    if img.ndim == 2:
        return img
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class TorchOcr:
    """In-process инференс CRNN поверх вендоренного кода (задача 4.2).

    Модель и алфавит загружаются из чекпоинта один раз при создании;
    повторные read() только прогоняют инференс. Устройство — авто-детект
    (cuda -> cpu), можно задать явно (device=...).

    Args:
        model_path: путь к чекпоинту crnn_best.pt (файл вне git, paths.model).
        device: устройство инференса; None — авто-детект.
        loader: инъекция загрузчика (model_path) -> (model, alphabet, device)
            для тестов; по умолчанию — вендоренный load_model.
        cuda_available: инъекция доступности GPU для авто-детекта (тесты).
    """

    def __init__(
        self,
        model_path: str | Path,
        device=None,
        *,
        loader: Callable | None = None,
        cuda_available: Callable[[], bool] | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"чекпоинт CRNN не найден: {model_path}")
        from l2core.adapters.ocr_vendor import (
            ctc_greedy_decode,
            load_model,
            preprocess,
        )

        import torch

        if loader is None:
            loader = load_model
        model, alphabet, _vendor_device = loader(model_path)
        self._torch = torch
        self._preprocess = preprocess
        self._decode = ctc_greedy_decode
        self._alphabet = list(alphabet)
        self._device = (
            device if device is not None else _resolve_device(torch, cuda_available)
        )
        self._model = model.to(self._device).eval()

    @property
    def device(self):
        """Устройство инференса (torch.device)."""
        return self._device

    def read(self, region_img: np.ndarray) -> tuple[str, float]:
        """Контракт OcrClient: BGR/gray регион -> (текст, confidence)."""
        gray = _to_gray(np.asarray(region_img))
        x = self._preprocess(gray)[None]  # (1, 1, IMG_H, IMG_W)
        with self._torch.no_grad():
            logits = self._model(self._torch.from_numpy(x).to(self._device))
        return (
            self._decode(logits[0], self._alphabet),
            _confidence_from_logits(logits[0]),
        )


def create_ocr_client(
    kind: str = "torch",
    *,
    model_path: str | Path | None = None,
    device=None,
    loader: Callable | None = None,
    cuda_available: Callable[[], bool] | None = None,
    endpoint: str | None = None,
) -> OcrClient:
    """Точка подмены реализации OcrClient: kind выбирает реализацию.

    Ядро и сценарии работают только с контрактом OcrClient; выбор
    реализации (здесь — аргумент, далее может читаться из конфига)
    не требует правки кода выше.
    """
    if kind == "torch":
        if model_path is None:
            raise ValueError("create_ocr_client('torch'): не задан model_path")
        return TorchOcr(
            model_path, device, loader=loader, cuda_available=cuda_available
        )
    if kind == "remote":
        return RemoteOcr(endpoint)
    raise ValueError(
        f"неизвестная реализация OCR: {kind!r} (доступны: 'torch', 'remote')"
    )
