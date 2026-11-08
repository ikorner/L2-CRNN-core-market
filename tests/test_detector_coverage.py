"""Покрытие детектора всеми 39 скриншотами сырья (задача 3.7).

Спека window-state-detection: «Для каждого из 39 существующих скриншотов
в base_screenshot/scr/ SHALL существовать тест с ожидаемым (window, params)
по мере разметки меты».

Ожидания читаются программно из meta.json (не хардкодом списком):
параметризация строится по ключам меты, window/state/state_params — из
записи скриншота. Для скринов с неразмеченным state (R4) действует явная
политика категорий со ссылками на KNOWN_GAPS (ui_model/KNOWN_GAPS.yaml):

- неизвестное как known не проходит: если детектор начнёт подтверждать
  состояние на скрине из HONEST_UNKNOWN (честный unknown по KNOWN_GAPS),
  тест упадёт — такое расхождение требует пересмотра разметки/модели,
  а не подгонки теста;
- незарегистрированный в политике скриншот (новый файл в meta.json или
  state, добавленный метой без покрытия) — падение, а не молчаливый
  пропуск: guard-тест test_meta_coverage_matches_policy следит за
  двусторонним совпадением множеств.

Конфликт сигнатур (несколько подтверждённых состояний -> unknown +
журнал конфликта) покрыт синтетикой tests/test_states.py::TestConflict и
здесь не дублируется.

Прогон: один проход детекции по 39 кадрам (~6 с, OpenCV-матчер);
кадры кэшируются между параметризованными случаями (lru_cache).
"""

import json
from functools import lru_cache
from pathlib import Path

import cv2
import pytest

from l2core.adapters.template import OpenCvTemplateMatcher
from l2core.detector.anchor import AnchorDetector
from l2core.detector.states import StateDetector
from l2core.model import load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"
ASSETS_DIR = REPO_ROOT / "assets"
UI_MODEL_DIR = REPO_ROOT / "ui_model"

META = json.loads((SCR_DIR / "meta.json").read_text(encoding="utf-8"))

# Скрины, где по мете state НЕ размечен (R4): явные категории ожиданий.
# window/params ожиданий — из меты; категории фиксируют только факт
# детекции. Каждая категория — со ссылкой на KNOWN_GAPS.yaml.

# Негативы: мете window задан, но окна торговли на кадре нет
# (scr_153511 — окно игры неактивно, scr_153527 — без открытых лавок,
# scr_all_close — всё закрыто). Якорь обязан НЕ находиться
# (tests/test_anchor.py::TestAnchorDetectorMarket проверяет то же по
# позициям — здесь только факт покрытия).
ANCHOR_ABSENT = ("scr_153511", "scr_153527", "scr_all_close")

# Мета задала state (open), но якорь честно не находится: заголовок
# «open»-состояния лавки («<Продажа>»/«<Покупка>») в модель не
# импортирован (KNOWN_GAPS.yaml п. 6, отчёт 3.3). До закрытия пробела
# владельцем ожидаем честную не-находку, а не подгонку.
ANCHOR_ABSENT_OPEN = ("scr_154318", "scr_154424")

# Ложное подтверждение params: сигнатуры filter_button_all_active/
# snaryag_all_active не дискриминируют фильтры (0.930 на чужом фильтре
# при пороге 0.9, замер 3.4) — KNOWN_GAPS.yaml п. 4. Мета этим скринам
# state не задавала (global_filter=sell / filter=weapon|armor|acces):
# ожидание фиксирует факт; при закрытии пробела правкой модели оно
# меняется на честный unknown.
FALSE_CONFIRM_EQUIPMENT = (
    "scr_153611", "scr_153619", "scr_153631", "scr_153646", "scr_153656",
)

# Мета state не размечена (R4); факт детектора — подтверждённое opening:
# scr_154222 (лавка продажи, вывеска активна). Сырья «open»-состояния
# лавки с активной вывеской нет; семантика scr_154222 — владельцу
# (KNOWN_GAPS.yaml п. 3, 5).
SHOP_FACT_OPENING = ("scr_154222",)

# Честный unknown при найденном якоре: состояния этих табов/фильтров/
# подсостояний в модели не размечены (KNOWN_GAPS.yaml п. 3, 5:
# scr_154355/154244 — сигнатуры opening на динамическом контенте
# противоречат остальным признакам, строгое «все сигнатуры» даёт
# unknown). Любое подтверждённое состояние здесь — падение.
HONEST_UNKNOWN = (
    "scr_153712", "scr_153722", "scr_153731", "scr_153740", "scr_153750",
    "scr_153802", "scr_153810", "scr_153819", "scr_153830", "scr_153840",
    "scr_153901", "scr_153929", "scr_153941", "scr_153949", "scr_154002",
    "scr_154012", "scr_154021", "scr_154033", "scr_154049", "scr_154103",
    "scr_154134", "scr_154244", "scr_154355",
)

_ANCHOR_ABSENT = object()  # маркер ожидания «якорь не найден»


@lru_cache(maxsize=None)
def frame_bgr(name: str):
    return cv2.imread(str(SCR_DIR / f"{name}.png"))


@pytest.fixture(scope="module")
def shipped_models():
    return {
        window_id: load_model(UI_MODEL_DIR / f"{window_id}.yaml")
        for window_id in ("market", "shop_buy", "shop_sell")
    }


@pytest.fixture(scope="module")
def real_detector(shipped_models):
    anchor = AnchorDetector.for_windows(
        shipped_models, OpenCvTemplateMatcher(), ASSETS_DIR)
    states = StateDetector(shipped_models, OpenCvTemplateMatcher(), ASSETS_DIR)
    return anchor, states


def _resolve_meta_state(entry, model):
    """(state, params) из размеченной меты: локальное имя состояния +
    точное совпадение params. Неоднозначность/отсутствие — падение
    (мета добавила state, не покрытый тестом)."""
    local = entry["state"].split(".", 1)[1] \
        if "." in entry["state"] else entry["state"]
    candidates = [
        state for sid, state in model.states.items()
        if not sid.startswith("recover_")
        and (sid == local or sid.startswith(f"{local}__"))
    ]
    params = dict(entry["state_params"])
    matched = [s for s in candidates if dict(s.params) == params]
    if len(matched) != 1:
        pytest.fail(
            f"мета добавила state {entry['state']!r} с params {params}, "
            f"не покрытый тестом: кандидаты {[s.id for s in candidates]}"
        )
    return matched[0].id, params


def _expected(name: str, entry, model):
    """Ожидаемый результат детекции для скриншота меты."""
    if entry["state"] is not None:
        if name in ANCHOR_ABSENT_OPEN:
            return _ANCHOR_ABSENT
        return _resolve_meta_state(entry, model)
    if name in ANCHOR_ABSENT:
        return _ANCHOR_ABSENT
    if name in FALSE_CONFIRM_EQUIPMENT:
        return (
            "list__filter_all__global_filter_all__tab_equipment",
            {"filter": "all", "global_filter": "all", "tab": "equipment"},
        )
    if name in SHOP_FACT_OPENING:
        return ("opening", {})
    if name in HONEST_UNKNOWN:
        return (None, {})
    pytest.fail(
        f"{name}: скриншот меты не зарегистрирован в политике ожиданий "
        f"tests/test_detector_coverage.py — покрытие расходится с разметкой"
    )


@pytest.mark.parametrize(
    "name", sorted(META["screenshots"]), ids=sorted(META["screenshots"]))
def test_detector_covers_meta_screenshots(real_detector, shipped_models, name):
    """Каждый из 39 скринов меты: ожидаемый (window, params) по разметке."""
    entry = META["screenshots"][name]
    window_id = entry["window"]  # ожидание window — из меты
    anchor_detector, state_detector = real_detector
    frame = frame_bgr(name)

    expected = _expected(name, entry, shipped_models[window_id])
    if expected is _ANCHOR_ABSENT:
        assert anchor_detector.find_anchor(frame, window_id) is None
        return

    anchor = anchor_detector.find_anchor(frame, window_id)
    assert anchor is not None, f"якорь {window_id} не найден на {name}"
    assert anchor.window_id == window_id

    expect_state, expect_params = expected
    report = state_detector.detect(frame, anchor)
    assert report.state == expect_state, (
        f"{name}: факт {report.state!r} != ожидание {expect_state!r} "
        f"(мета state={entry['state']!r}, params={entry['state_params']}); "
        f"если мета разметила новое состояние — покрой его здесь; если "
        f"поменялся детектор — пересмотри KNOWN_GAPS"
    )
    assert report.params == expect_params


def test_meta_coverage_matches_policy():
    """Двусторонняя сверка: все скрины меты зарегистрированы в политике
    и ни одной лишней записи (пробел закрыт — запись забыли убрать)."""
    meta_states = {
        name for name, entry in META["screenshots"].items()
        if entry["state"] is not None
    }
    covered = (
        set(meta_states) | set(ANCHOR_ABSENT) | set(ANCHOR_ABSENT_OPEN)
        | set(FALSE_CONFIRM_EQUIPMENT) | set(SHOP_FACT_OPENING)
        | set(HONEST_UNKNOWN)
    )
    assert covered == set(META["screenshots"])
    # Категории не пересекаются: один скрин — одна ожидаемая категория.
    categories = [
        set(ANCHOR_ABSENT), set(ANCHOR_ABSENT_OPEN),
        set(FALSE_CONFIRM_EQUIPMENT), set(SHOP_FACT_OPENING),
        set(HONEST_UNKNOWN),
    ]
    total = sum(len(c) for c in categories)
    assert total == len(set().union(*categories)), "категории пересекаются"


def test_screenshot_files_all_in_meta():
    """Файл сырья без записи в мете = скриншот без теста (и наоборот)."""
    pngs = {p.stem for p in SCR_DIR.glob("scr_*.png")}
    assert pngs == set(META["screenshots"])
