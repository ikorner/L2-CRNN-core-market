"""
Вендоринг: загрузчик чекпоинта/алфавита CRNN.

Источник: L2-CRNN-MODEL-TRAIN/autolabel.py (load_model), коммит
e720a8df8078337740df5c3ad8dc2b1eb4387529.
Копия без редизайна; правки вендоренного кода запрещены (менять в источнике).

Адаптации (минимально необходимые): `train.CRNN` -> `CRNN` (квалификация
модуля источника убрана, класс вендорен рядом — crnn_train.py); тело функции
без изменений. Авто-детект устройства — как в источнике (mps, иначе cpu);
выбор устройства для l2core — задача 4.2 (TorchOcr), здесь не меняется.
"""
from __future__ import annotations

from pathlib import Path

import torch

from l2core.adapters.ocr_vendor.crnn_train import CRNN


def load_model(pt_path: Path) -> tuple[CRNN, list[str], torch.device]:
    ckpt = torch.load(pt_path, map_location="cpu")
    alphabet = list(ckpt["alphabet"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CRNN(n_classes=len(alphabet))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, alphabet, device
