"""
Вендоринг: preprocess + CRNN + ctc_greedy_decode из обучающего кода CRNN-OCR.

Источник: L2-CRNN-MODEL-TRAIN/train.py, коммит e720a8df8078337740df5c3ad8dc2b1eb4387529.
Копия без редизайна; правки вендоренного кода запрещены (менять в источнике).
Вендорятся только блоки, нужные инференсу: константы, preprocess (+ его
хелперы), CRNN, ctc_greedy_decode. Train loop, метрики и загрузка датасета
не вендорятся (YAGNI). Guard-тест на эталонных векторах:
tests/test_ocr_vendor.py (эталоны — tools/gen_ocr_vendor_fixtures.py).

Адаптации строго импортного уровня (тела функций и класса — байт-в-байт):
- убраны импорты argparse/csv/time, `import paths` и код train loop
  (нужны только обучению; абсолютные пути старого проекта в ядре недопустимы);
- `from PIL import Image` сохранён (fallback resize при отсутствии cv2),
  Pillow объявлен в pyproject.toml.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG_H, IMG_W = 32, 320
BLANK = 0


def _resize_center(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize с сохранением пропорций, центрирование на чёрном холсте."""
    ih, iw = img.shape
    scale = min(h / ih, w / iw)
    nh, nw = max(1, int(round(ih * scale))), max(1, int(round(iw * scale)))
    resized = _resize(img, nw, nh)
    canvas = np.zeros((h, w), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except ImportError:
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def _warp_affine(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    ih, iw = img.shape
    try:
        import cv2  # type: ignore
        return cv2.warpAffine(img, M, (iw, ih), borderValue=0)
    except ImportError:
        from PIL import Image as _I
        # PIL rotate: положительный угол = против часовой; cv2 M с таким же
        ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        r = _I.fromarray(img).rotate(-ang + ang, fillcolor=0, resample=_I.BILINEAR)
        # упрощение: только сдвиг через crop, поворот небольшой
        res = np.asarray(r)
        canvas = np.zeros((ih, iw), np.uint8)
        canvas[:min(ih, res.shape[0]), :min(iw, res.shape[1])] = res[
            :min(ih, res.shape[0]), :min(iw, res.shape[1])]
        return canvas


def preprocess(img: np.ndarray, rng: random.Random | None = None) -> np.ndarray:
    """[H,W] uint8 -> [1,IMG_H,IMG_W] float32 в [0,1].

    rng — при передаче применяются тренировочные аугментации."""
    out = _resize_center(img, IMG_H, IMG_W).astype(np.float32)
    if rng is not None:
        out = out * rng.uniform(0.85, 1.1) + rng.uniform(-6, 6)
        if rng.random() < 0.35:
            ang = math.radians(rng.uniform(-1.5, 1.5))
            shift = rng.uniform(-1.5, 1.5)
            M = np.float32([[math.cos(ang), -math.sin(ang), shift],
                            [math.sin(ang), math.cos(ang), 0.0]])
            out = _warp_affine(out.astype(np.uint8), M).astype(np.float32)
        out += rng.gauss(0, 1.0)
        out = np.clip(out, 0, 255)
    return (out / 255.0)[None]  # (1, H, W)


class CRNN(nn.Module):
    """1x32x320 -> CNN (H/8, W/4) -> BiLSTM -> FC (CTC logits)."""

    def __init__(self, n_classes: int, depth: int = 64, lstm_hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, depth, 3, 1, 1), nn.BatchNorm2d(depth), nn.ReLU(),
            nn.MaxPool2d(2, 2),                                  # H/2
            nn.Conv2d(depth, depth, 3, 1, 1), nn.BatchNorm2d(depth), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),                         # H/4
            nn.Conv2d(depth, depth * 2, 3, 1, 1), nn.BatchNorm2d(depth * 2), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),                         # H/8
            nn.Conv2d(depth * 2, depth * 2, (3, 1), (2, 1), (0, 1)),
            nn.BatchNorm2d(depth * 2), nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),                         # W/2
        )
        # фактический выход CNN на (1, 32, 320): (128, 1, 81) => feat_h=1, T=81
        self.bn_in = nn.BatchNorm1d(depth * 2)
        self.rnn = nn.LSTM(depth * 2, lstm_hidden, 1,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cnn(x)                        # (B, C, h, w)
        B, C, h, w = c.shape
        c = c.permute(0, 3, 1, 2).reshape(B, w, C * h)  # (B, T, C*h)
        c = c.permute(0, 2, 1)                          # (B, C*h, T) для BN1d
        c = self.bn_in(c)                               # (N, C, L)
        c = c.permute(0, 2, 1)                          # (B, T, C*h) для LSTM batch_first
        r, _ = self.rnn(c)
        return self.fc(r)                      # (B, T, C+1)


def ctc_greedy_decode(logits: torch.Tensor, alphabet: list[str]) -> str:
    idx = logits.argmax(dim=-1).tolist()
    out, prev = [], BLANK
    for i in idx:
        if i != prev and i != BLANK:
            out.append(alphabet[i - 1])
        prev = i
    return "".join(out)
