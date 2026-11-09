"""Golden-набор OCR: индекс, guard чтения эталонов и smoke бенчмарка
(задача 4.6, spec ocr-reading: «Golden-набор и OCR-бенчмарк»).

Устройство golden-набора (assets/ocr_golden/):
- index.yaml — машиночитаемый индекс карточек: id, файл кропа, скриншот-
  источник, rect в пикселях источника, ожидаемый текст, грамматика
  (digits/price/text из common.yaml), критерий известности текста и кто/
  когда подтвердил известность (ruling R4: текст читается человеком со
  скриншота без сомнений; описания меты — аннотации, НЕ ground truth);
- <id>.png — эталонные кропы, нарезанные из сырья инструментом
  tools/gen_ocr_golden.py (регенерация: python tools/gen_ocr_golden.py).

Покрытие:
- индекс валиден: обязательные поля, уникальные id, известные грамматики,
  rect внутри скриншота-источника, размер кропа == rect;
- кропы не протухли: нарезка rect'ом из источника совпадает с файлом
  пиксель-в-пиксель (base_screenshot/scr — в репозитории, сырьё не
  зависимость ядра, но зависимость guard-теста golden-набора);
- guard чтения через OcrClient: при наличии реального crnn_best.pt
  (paths.model из config.yaml) каждая карточка читается эталонным текстом;
  карточки, чей текст содержит символы вне алфавита модели, пропускаются
  с диагностикой (алфавит хранится в чекпоинте — до его получения честно
  неизвестно, тянет ли модель кириллицу/«тыс»/«млрд»). Без модели —
  pytest.skip с parked-диагностикой (прогон с реальной моделью — на
  машине владельца, docs/OWNER_QUESTIONS.md);
- smoke бенчмарка (tools/benchmark_ocr.py) на fake-чекпоинте 4.1: пайплайн
  исполняется на CPU и возвращает осмысленную структуру (точность/время);
  GPU-ветка кода присутствует и на CPU-only окружении честно помечается
  как «не исполнено» с причиной (ruling R6).
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "assets" / "ocr_golden"
INDEX_PATH = GOLDEN_DIR / "index.yaml"
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"
FAKE_CKPT = REPO_ROOT / "tests" / "fixtures" / "ocr_vendor" / "fake_crnn.pt"
BENCHMARK_SCRIPT = REPO_ROOT / "tools" / "benchmark_ocr.py"

KNOWN_GRAMMARS = {"digits", "price", "text"}
REQUIRED_FIELDS = frozenset(
    {
        "id",
        "file",
        "source",
        "rect",
        "text",
        "grammar",
        "known_by",
        "confirmed_by",
        "confirmed_at",
    }
)


def _load_index() -> dict:
    assert INDEX_PATH.is_file(), f"индекс golden-набора не найден: {INDEX_PATH}"
    data = yaml.safe_load(INDEX_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "index.yaml: корень — не словарь"
    return data


def _cards() -> list[dict]:
    return list(_load_index()["cards"])


def _source_image(source: str) -> np.ndarray:
    path = SCR_DIR / f"{source}.png"
    assert path.is_file(), f"скриншот-источник не найден: {path}"
    img = cv2.imread(str(path))
    assert img is not None
    return img


def _real_model_path() -> Path:
    from l2core.config import load_config

    cfg = load_config(REPO_ROOT / "config.yaml")
    return REPO_ROOT / cfg.paths.model


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_ocr", BENCHMARK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestIndex:
    def test_index_exists_and_has_version(self):
        data = _load_index()
        assert data["version"] == 1
        assert isinstance(data["criteria"], str) and data["criteria"].strip()

    def test_cards_have_required_fields(self):
        for card in _cards():
            missing = REQUIRED_FIELDS - set(card)
            assert not missing, f"{card.get('id')}: нет полей {missing}"

    def test_ids_unique(self):
        ids = [c["id"] for c in _cards()]
        assert len(ids) == len(set(ids)), "дубли id в индексе"

    def test_grammars_known(self):
        for card in _cards():
            assert card["grammar"] in KNOWN_GRAMMARS, (
                f"{card['id']}: неизвестная грамматика {card['grammar']!r}"
            )

    def test_at_least_one_card_per_grammar(self):
        grammars = {c["grammar"] for c in _cards()}
        assert grammars == KNOWN_GRAMMARS, (
            f"golden-набор должен покрывать все грамматики, есть: {grammars}"
        )


class TestFiles:
    @pytest.mark.parametrize("card", _cards(), ids=lambda c: c["id"])
    def test_crop_matches_rect_and_source(self, card):
        path = GOLDEN_DIR / card["file"]
        assert path.is_file(), f"кроп не найден: {path} (регенерация: tools/gen_ocr_golden.py)"
        stored = cv2.imread(str(path))
        assert stored is not None, f"кроп не декодируется: {path}"
        x, y, w, h = card["rect"]
        source = _source_image(card["source"])
        sh, sw = source.shape[:2]
        assert x >= 0 and y >= 0 and x + w <= sw and y + h <= sh, (
            f"{card['id']}: rect {card['rect']} вне скриншота {sw}x{sh}"
        )
        assert (stored.shape[1], stored.shape[0]) == (w, h), (
            f"{card['id']}: размер кропа {stored.shape[1]}x{stored.shape[0]} != rect {w}x{h}"
        )
        recut = source[y : y + h, x : x + w]
        assert np.array_equal(stored, recut), (
            f"{card['id']}: кроп протух — не совпадает с нарезкой rect из {card['source']} "
            "(пересоздайте: python tools/gen_ocr_golden.py)"
        )


class TestGuardReading:
    """Guard чтения эталонов через OcrClient — только с реальной моделью."""

    def test_golden_reading_with_real_model(self):
        model_path = _real_model_path()
        if not model_path.is_file():
            pytest.skip(
                f"parked: чекпоинт CRNN не найден ({model_path}); guard чтения golden "
                "прогоняется на машине владельца с crnn_best.pt "
                "(docs/OWNER_QUESTIONS.md, task 4.6)"
            )
        from l2core.adapters.ocr import TorchOcr

        ocr = TorchOcr(model_path)
        alphabet = set(ocr.alphabet)
        skipped, failed = [], []
        for card in _cards():
            unsupported = set(card["text"]) - alphabet
            if unsupported:
                skipped.append((card["id"], "".join(sorted(unsupported))))
                continue
            img = cv2.imread(str(GOLDEN_DIR / card["file"]))
            text, confidence = ocr.read(img)
            if text != card["text"]:
                failed.append((card["id"], card["text"], text, confidence))
        assert not failed, "эталоны не прочитаны ожидаемым текстом: " + json.dumps(
            failed, ensure_ascii=False
        )
        if skipped:
            pytest.skip(
                "карточки вне алфавита модели (не проверены): "
                + json.dumps(skipped, ensure_ascii=False)
            )


class TestBenchmarkSmoke:
    """Пайплайн бенчмарка исполняется на fake-чекпоинте (CPU); числа точности
    смысловые только с реальной моделью — здесь проверяется механика."""

    def test_benchmark_runs_on_cpu_with_fake_checkpoint(self):
        bm = _load_benchmark_module()
        results = bm.run_benchmark(
            model_path=FAKE_CKPT,
            devices=["cpu"],
            index_path=INDEX_PATH,
            repeat=2,
        )
        cpu = results["cpu"]
        assert cpu["status"] == "ok", cpu
        assert cpu["n_cards"] == len(_cards())
        assert 0.0 <= cpu["accuracy"] <= 1.0
        assert cpu["mean_seconds"] >= 0.0
        assert len(cpu["per_card"]) == len(_cards())
        sample = cpu["per_card"][0]
        assert {"id", "text", "confidence", "ok", "seconds"} <= set(sample)

    def test_gpu_branch_present_and_honestly_skipped_without_gpu(self):
        # ruling R6: GPU-путь кода существует, но не исполнен в CPU-only среде —
        # бенчмарк обязан честно пометить это, а не падать или молчать.
        bm = _load_benchmark_module()
        results = bm.run_benchmark(
            model_path=FAKE_CKPT,
            devices=["cpu", "cuda"],
            index_path=INDEX_PATH,
            repeat=1,
        )
        assert results["cpu"]["status"] == "ok"
        cuda = results["cuda"]
        if not bm.torch.cuda.is_available():
            assert cuda["status"] == "skipped"
            assert cuda["reason"]
        else:  # на машине с GPU ветка реально исполняется
            assert cuda["status"] == "ok"
