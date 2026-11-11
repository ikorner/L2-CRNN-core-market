"""Константы путей зоны ретрейна (локальная копия, адаптирована под Windows).

Зона retrain/ живёт внутри репо core-market и ничего не импортирует из
внешних репо: build_dataset.py и train.py скопированы сюда, dataset/ —
копия датасета обучения, screenshots_new/ — текущие скриншоты core-market,
конвертированные в BMP.
"""
from pathlib import Path

ROOT = Path(__file__).parent
SCREENSHOTS_DIR = ROOT / "screenshots_new"   # текущие скриншоты (BMP)
DATASET_DIR = ROOT / "dataset"               # копия датасета обучения
RUNS_DIR = ROOT / "runs"                    # выход дообучения
BASE_MODEL = ROOT / "base_model.pt"         # чекпоинт до обучения (val 0.983)
