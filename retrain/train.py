#!/usr/bin/env python3
"""Обучение CRNN (CNN -> BiLSTM -> CTC) на размеченных ячейках из dataset/.

Вход:  dataset/labels.csv (image_path,text) + dataset/cells.csv (pixel_key, col)
Выход: runs/<ts>/crnn_best.pt, runs/<ts>/log.txt, runs/<ts>/val_errors.csv

Запуск:
    .venv-train/bin/python train.py [--epochs 500] [--batch 32] [--device mps]

Ключевые решения:
- split train/val по pixel_key (не по строкам): одинаковые кропы не в оба.
- контейнер 32x320 (H x W): height 32 — стандарт CRNN, ширина по медиане aspect ratio;
  длинные строки сжимаются с сохранением пропорций (без деформации).
- предобработка одна и та же для train/val/инференса:
  grayscale -> resize с сохранением пропорций -> центрирование на чёрном
  холсте 32x320 -> [0,1]. На train добавляются лёгкие аугментации.
- дефолты: --dataset и --runs-dir ведут в старый/новый проекты (paths.py),
  lr 5e-4 (1e-3 — на грани дивергенции), 500 эпох.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import paths
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).parent

IMG_H, IMG_W = 32, 320
BLANK = 0


# ---------- данные ----------

@dataclass
class Sample:
    img: np.ndarray      # uint8 grayscale, исходный кроп
    text: str
    pixel_key: str
    col: str             # type/name/price/seller/deal/header
    rel_path: str


def load_samples(dataset_dir: Path) -> list[Sample]:
    dataset_dir = Path(dataset_dir)
    labels = {
        Path(r["image_path"]).name: r["text"]
        for r in csv.DictReader((dataset_dir / "labels.csv").open())
        if r["text"].strip()
    }
    meta = {
        r["pixel_key"]: r
        for r in csv.DictReader((dataset_dir / "cells.csv").open())
    }
    samples: list[Sample] = []
    for name, text in labels.items():
        path = dataset_dir / "images" / name
        if not path.exists():
            continue
        stem = Path(name).stem  # имена файлов = pixel_key из cells.csv
        m = meta.get(stem, {})
        samples.append(Sample(
            img=np.asarray(Image.open(path).convert("L")),
            text=text,
            pixel_key=m.get("pixel_key", name),
            col=m.get("col", "header"),
            rel_path=f"images/{name}",
        ))
    return samples


def split_by_pixel_key(samples: list[Sample], val_frac: float, seed: int = 42
                       ) -> tuple[list[Sample], list[Sample]]:
    keys = sorted({s.pixel_key for s in samples})
    rng = random.Random(seed)
    rng.shuffle(keys)
    cut = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:cut])
    train = [s for s in samples if s.pixel_key not in val_keys]
    val = [s for s in samples if s.pixel_key in val_keys]
    return train, val


def build_alphabet(samples: list[Sample]) -> list[str]:
    chars = set()
    for s in samples:
        chars.update(c for c in s.text if c != " ")
    chars.add(" ")
    return sorted(chars)


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


class L2Dataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample], alphabet: list[str], train: bool, seed: int = 42):
        self.samples = samples
        self.index = {c: i + 1 for i, c in enumerate(alphabet)}
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[i]
        img = torch.from_numpy(preprocess(s.img, rng=self.rng if self.train else None))
        ids = [self.index[c] for c in s.text if c in self.index]
        return img, torch.tensor(ids, dtype=torch.int32)

    @staticmethod
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        imgs = torch.stack([b[0] for b in batch])
        lengths = torch.tensor([b[1].numel() for b in batch], dtype=torch.int32)
        targets = torch.cat([b[1] for b in batch])
        return imgs, targets, lengths


# ---------- модель ----------

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


# ---------- метрики ----------

def char_accuracy(pred: str, truth: str) -> float:
    """Доля позиций (по min-длине), где совпадает символ; пробелы учитываем."""
    if not truth:
        return 1.0 if not pred else 0.0
    n = min(len(pred), len(truth))
    return sum(1 for a, b in zip(pred, truth) if a == b) / max(len(truth), 1)


# ---------- train loop ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(paths.DATASET_DIR))
    ap.add_argument("--runs-dir", default=str(paths.RUNS_DIR))
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    samples = load_samples(Path(args.dataset))
    alphabet = build_alphabet(samples)
    train, val = split_by_pixel_key(samples, args.val_frac, seed=args.seed)
    print(f"samples={len(samples)} unique pixel_keys={len({s.pixel_key for s in samples})}")
    print(f"train={len(train)} val={len(val)} alphabet={len(alphabet)} chars: {''.join(alphabet)}")

    # col-метки: храним в датасете, collate тащит их
    train_ds = L2Dataset(train, alphabet, train=True, seed=args.seed)
    val_ds = L2Dataset(val, alphabet, train=False, seed=args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, collate_fn=L2Dataset.collate)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, collate_fn=L2Dataset.collate)

    model = CRNN(n_classes=len(alphabet)).to(device)
    with torch.no_grad():
        T = model(torch.zeros(1, 1, IMG_H, IMG_W, device=device)).shape[1]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M, device={device}, T={T}")

    max_len = max(len(s.text) for s in samples)
    if max_len > T:
        print(f"WARNING: max text len {max_len} > T={T} — такие строки не обучаются")

    loss_fn = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_dir = Path(args.runs_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    (run_dir / "config.json").write_text(json.dumps({
        "alphabet": "".join(alphabet), "img": [IMG_H, IMG_W],
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "device": str(device),
    }, ensure_ascii=False, indent=2))

    best = (-1.0, None)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss = 0.0
        nb = 0
        for imgs, targets, lengths in train_loader:
            opt.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            # CTC: input (T, B, C); input_lengths — число таймстепов T
            logits = model(imgs)               # (B, T, C+1)
            T = logits.shape[1]
            assert T >= int(lengths.max()), \
                f"T={T} < max target len {int(lengths.max())}"
            input_lengths = torch.full((imgs.size(0),), T,
                                       dtype=torch.int32, device=device)
            loss = loss_fn(logits.permute(1, 0, 2), targets,
                           input_lengths, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item()
            nb += 1
        sched.step()
        train_loss = tot_loss / nb

        # валидация + col-статистика вручную (val_loader без shuffle => порядок = val)
        model.eval()
        preds: list[str] = []
        with torch.no_grad():
            for imgs, targets, lengths in val_loader:
                logits = model(imgs.to(device))
                for i in range(len(lengths)):
                    preds.append(ctc_greedy_decode(logits[i], alphabet))
        n = len(preds)
        exact = sum(1 for p, s in zip(preds, val) if p == s.text) / n
        char_acc = sum(char_accuracy(p, s.text) for p, s in zip(preds, val)) / n
        col_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for p, s in zip(preds, val):
            col_stat[s.col][0] += int(p == s.text)
            col_stat[s.col][1] += 1
        col_report = {c: f"{v[0]}/{v[1]}" for c, v in sorted(col_stat.items())}

        line = (f"epoch {epoch:2d}  loss {train_loss:7.3f}  "
                f"val exact {exact:6.3f}  char_acc {char_acc:6.3f}  "
                f"cols {col_report}  {time.time()-t0:5.1f}s")
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

        if exact > best[0]:
            best = (exact, epoch)
            torch.save({
                "model": model.state_dict(),
                "alphabet": "".join(alphabet),
                "img_size": [IMG_H, IMG_W],
                "epoch": epoch,
                "val_exact": exact,
            }, run_dir / "crnn_best.pt")

        # ошибки валидации (для диагностики после финального прогона)
        if epoch == args.epochs or exact == best[0]:
            errs = [(s.text, p, char_accuracy(p, s.text), s.col)
                    for p, s in zip(preds, val) if p != s.text]
            with (run_dir / f"val_errors_epoch{epoch}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["truth", "pred", "char_acc", "col"])
                w.writerows(errs[:300])

    print(f"\nbest val exact: {best[0]:.3f} @ epoch {best[1]}  -> {run_dir/'crnn_best.pt'}")


if __name__ == "__main__":
    main()
