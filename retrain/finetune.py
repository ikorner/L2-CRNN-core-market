"""Дообучение CRNN: warm-start от base_model.pt на смешанном датасете.

Отличия от train.py:
- алфавет БЕРЁТСЯ из базового чекпоинта (полный warm-start головы,
  новые данные — подмножество алфавита, проверено build_mixed.py);
- дефолты: lr 1e-4, 100 эпох, cpu.

Запуск из retrain/:  python finetune.py [--epochs 100] [--lr 1e-4]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

import train as T  # Sample, split, L2Dataset, CRNN, decode, metrics

ROOT = Path(__file__).parent


def load_samples_utf8(dataset_dir: Path) -> list:
    """Копия T.load_samples с явным UTF-8 (train.py открывает CSV в cp1251)."""
    import numpy as np
    from PIL import Image

    dataset_dir = Path(dataset_dir)
    labels = {
        Path(r["image_path"]).name: r["text"]
        for r in csv.DictReader((dataset_dir / "labels.csv").open(encoding="utf-8"))
        if r["text"].strip()
    }
    meta = {
        r["pixel_key"]: r
        for r in csv.DictReader((dataset_dir / "cells.csv").open(encoding="utf-8"))
    }
    samples = []
    for name, text in labels.items():
        path = dataset_dir / "images" / name
        if not path.exists():
            continue
        stem = Path(name).stem
        m = meta.get(stem, {})
        samples.append(T.Sample(
            img=np.asarray(Image.open(path).convert("L")),
            text=text,
            pixel_key=m.get("pixel_key", name),
            col=m.get("col", "header"),
            rel_path=f"images/{name}",
        ))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "mixed_dataset"))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--base", default=str(ROOT / "base_model.pt"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="",
                    help="путь к crnn_last.pt — продолжить прерванный прогон")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    alphabet = list(base["alphabet"])
    print(f"base: epoch={base.get('epoch')} val_exact={base.get('val_exact')} "
          f"alphabet={len(alphabet)}")

    samples = load_samples_utf8(Path(args.dataset))
    # все символы данных должны входить в алфавит базы
    missing = sorted({c for s in samples for c in s.text} - set(alphabet))
    assert not missing, f"символы вне алфавита базы: {missing}"
    train, val = T.split_by_pixel_key(samples, args.val_frac, seed=args.seed)
    print(f"samples={len(samples)} unique pixel_keys={len({s.pixel_key for s in samples})}")
    print(f"train={len(train)} val={len(val)}")

    train_ds = T.L2Dataset(train, alphabet, train=True, seed=args.seed)
    val_ds = T.L2Dataset(val, alphabet, train=False, seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, collate_fn=T.L2Dataset.collate)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, collate_fn=T.L2Dataset.collate)

    model = T.CRNN(n_classes=len(alphabet)).to(device)
    start_epoch = 1
    if args.resume:
        last = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(last["model"])
        run_dir = Path(args.resume).parent
        best = tuple(json.loads((run_dir / "best.json").read_text(encoding="utf-8")))
        start_epoch = last["epoch"] + 1
        print(f"resume: {args.resume}, продолжаю с эпохи {start_epoch}, best={best[0]:.3f}@{best[1]}")
    else:
        model.load_state_dict(base["model"])
        run_dir = Path(args.runs_dir) / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        best = (-1.0, None)
        print("warm-start: веса базового чекпоинта загружены")

    loss_fn = nn.CTCLoss(blank=T.BLANK, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.resume:
        last = torch.load(args.resume, map_location="cpu", weights_only=False)
        opt.load_state_dict(last["opt"])
        sched.load_state_dict(last["sched"])

    log_path = run_dir / "log.txt"
    if not args.resume:
        (run_dir / "config.json").write_text(json.dumps({
            "alphabet": "".join(alphabet), "img": [T.IMG_H, T.IMG_W],
            "base": str(args.base), "dataset": str(args.dataset),
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "device": str(device),
        }, ensure_ascii=False, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss = 0.0
        nb = 0
        for imgs, targets, lengths in train_loader:
            opt.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            logits = model(imgs)
            T_steps = logits.shape[1]
            input_lengths = torch.full((imgs.size(0),), T_steps,
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

        model.eval()
        preds: list[str] = []
        with torch.no_grad():
            for imgs, targets, lengths in val_loader:
                logits = model(imgs.to(device))
                for i in range(len(lengths)):
                    preds.append(T.ctc_greedy_decode(logits[i], alphabet))
        n = len(preds)
        exact = sum(1 for p, s in zip(preds, val) if p == s.text) / n
        char_acc = sum(T.char_accuracy(p, s.text) for p, s in zip(preds, val)) / n
        col_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for p, s in zip(preds, val):
            col_stat[s.col][0] += int(p == s.text)
            col_stat[s.col][1] += 1
        col_report = {c: f"{v[0]}/{v[1]}" for c, v in sorted(col_stat.items())}

        line = (f"epoch {epoch:3d}  loss {train_loss:7.3f}  "
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
                "img_size": [T.IMG_H, T.IMG_W],
                "epoch": epoch,
                "val_exact": exact,
                "finetuned_from": str(args.base),
                "dataset": str(args.dataset),
            }, run_dir / "crnn_best.pt")
            errs = [(s.text, p, T.char_accuracy(p, s.text), s.col)
                    for p, s in zip(preds, val) if p != s.text]
            with (run_dir / "val_errors.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["truth", "pred", "char_acc", "col"])
                w.writerows(errs[:300])
            (run_dir / "best.json").write_text(json.dumps(best), encoding="utf-8")

        # состояние для возобновления (перезаписывается каждую эпоху,
        # атомарно через tmp + rename — прерывание чанка не портит файл)
        last_tmp = run_dir / "crnn_last.pt.tmp"
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "alphabet": "".join(alphabet),
            "img_size": [T.IMG_H, T.IMG_W],
        }, last_tmp)
        last_tmp.replace(run_dir / "crnn_last.pt")

    print(f"\nbest val exact: {best[0]:.3f} @ epoch {best[1]}  -> {run_dir/'crnn_best.pt'}")


if __name__ == "__main__":
    main()
