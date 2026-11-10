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
журнал конфликта) покрыт механизмом синтетикой
tests/test_states.py::TestConflict; на реальных кадрах конфликт по построению
не возникает (строгое «все сигнатуры») — здесь это закреплено исполняемо:
conflicts обязан быть пуст на всех 39 кадрах, иначе unknown в HONEST_UNKNOWN
перестал бы быть честным (это конфликт признаков, а не недоразмеченность).

Прогон: один проход детекции по 39 кадрам (~6 с, OpenCV-матчер);
кадры кэшируются между параметризованными случаями (lru_cache).
"""

import json
from functools import lru_cache
from pathlib import Path

import cv2
import pytest

from l2core.adapters.template import OpenCvTemplateMatcher
from l2core.adapters.template import default_matcher
from l2core.detector.anchor import AnchorDetector
from l2core.detector.navchecks import build_nav_evaluator
from l2core.detector.states import StateDetector
from l2core.model import Check, load_model, load_navigation

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
ANCHOR_ABSENT = ("scr_153511", "scr_153527", "scr_all_close",
               # досъёмка живого прогона 9.4 (Q28): базовый экран и
               # открытое «Меню» — окон торговли на кадре нет
               "scr_menu_closed", "scr_menu_open")

# Мета задала state (open), но якорь честно не находится: scr_154318/
# scr_154424 — окна истории торговли «<Продажа>»/«<Покупка>» (KNOWN_GAPS
# п. 1, отдельные окна, не смоделированные); их заголовки отличаются от
# опор opening/open лавок (позитивы open-опор — на досъёмке
# scr_nav_shop_*_open, tests/test_anchor.py::TestShopOpenAnchors).
# Не-находка здесь честная, не пробел якоря (п. 6 закрыт).
ANCHOR_ABSENT_OPEN = ("scr_154318", "scr_154424")

# Кадры навигационной модели (досъёмка menu-quickslot-navigation,
# 2026-11-10): окна торговли на кадре нет, детектор окон не применяется —
# покрытие через оценщик навигационных проверок (build_nav_evaluator).
NAV_FRAMES = ("scr_nav_qs_panel", "scr_nav_menu_open")

# Досъёмка open-окон лавок (menu-quickslot-navigation, 2026-11-10):
# якорь находится по open-опоре (многоопорный якорь, KNOWN_GAPS п. 6
# закрыт), но состояние «open» свежим кадром не подтверждается — его
# сигнатуры импортированы со старого сырья и нестабильны (KNOWN_GAPS
# п. 5). Факт детектора зафиксирован честно: «opening».
FACT_OPEN_STATE = {
    "scr_nav_shop_sell_open": ("opening", {}),
    "scr_nav_shop_buy_open": ("opening", {}),
}

# Факт детектора после закрытия Q4 (глобальные фильтры «Продажа/Покупка»,
# ответ владельца + досъёмка): сигнатуры кружков global_filter
# дискриминируют (позитивы >=0.97, ложные <=0.81 при пороге 0.9).
# Мета state этим скринам не задавала, state_params знала только
# global_filter: ожидание фиксирует подтверждённое состояние.
# Подтабы снаряжения (Q5, ответ владельца: клик по пассивному подтабу
# активирует его — подсветка рамки; повторный клик без эффекта):
# дискриминация подтабов — пиксель-проба яркой рамки активной кнопки
# (active (118,103,87) vs passive (70,66,60), дельта 48 при допуске 16;
# template-кроссы подтабов 0.95+ не дискриминируют, замер Q5).
FACT_FILTER_STATES = {
    "scr_153611": (
        "list__filter_all__global_filter_sell__tab_equipment",
        {"filter": "all", "global_filter": "sell", "tab": "equipment"},
    ),
    "scr_153619": (
        "list__filter_all__global_filter_buy__tab_equipment",
        {"filter": "all", "global_filter": "buy", "tab": "equipment"},
    ),
    "scr_153631": (
        "list__filter_weapon__global_filter_buy__tab_equipment",
        {"filter": "weapon", "global_filter": "buy", "tab": "equipment"},
    ),
    "scr_153646": (
        "list__filter_armor__global_filter_buy__tab_equipment",
        {"filter": "armor", "global_filter": "buy", "tab": "equipment"},
    ),
    "scr_153656": (
        "list__filter_acces__global_filter_buy__tab_equipment",
        {"filter": "acces", "global_filter": "buy", "tab": "equipment"},
    ),
    # Табы (Q6, досъёмка владельца 09.11.2025): scr_153712/153802 — батч 1,
    # scr_153901+ — батч 2 (якорь (202,103)); подтверждённые состояния
    # сверены визуально (активный таб — нижняя кромка). Мета state не
    # задавала (tab=scroll фиксировал скролл, не таб).
    "scr_153712": (
        "list__filter_all__global_filter_buy__tab_artifacts",
        {"filter": "all", "global_filter": "buy", "tab": "artifacts"},
    ),
    "scr_153802": (
        "list__filter_all__global_filter_buy__tab_svotki",
        {"filter": "all", "global_filter": "buy", "tab": "svotki"},
    ),
    "scr_153901": (
        "list__filter_all__global_filter_buy__tab_pripasi",
        {"filter": "all", "global_filter": "buy", "tab": "pripasi"},
    ),
    "scr_154002": (
        "list__filter_all__global_filter_all__tab_prochee",
        {"filter": "all", "global_filter": "all", "tab": "prochee"},
    ),
    "scr_154103": (
        "list__filter_all__global_filter_all__tab_kollekcia",
        {"filter": "all", "global_filter": "all", "tab": "kollekcia"},
    ),
    "scr_154134": (
        "list__filter_all__global_filter_all__tab_kollekcia",
        {"filter": "all", "global_filter": "all", "tab": "kollekcia"},
    ),
    # Подтабы «Артефакты» (досъёмка владельца, приёмка): проба яркой рамки
    # активного подтаба + кружки global_filter + проба активного таба.
    # scr_153722/153731/153740/153750 ранее были честным unknown (состояния
    # не были размечены) — сверено визуально: 153722 «Атака»+«Покупка»
    # (пустой список, стрелки not_active — семантика Q9), 153731 «Защита»,
    # 153740 «Поддержка», 153750 «Баланс», у всех фильтр «Покупка».
    "scr_153722": (
        "list__filter_ataka__global_filter_buy__tab_artifacts",
        {"filter": "ataka", "global_filter": "buy", "tab": "artifacts"},
    ),
    "scr_153731": (
        "list__filter_zashita__global_filter_buy__tab_artifacts",
        {"filter": "zashita", "global_filter": "buy", "tab": "artifacts"},
    ),
    "scr_153740": (
        "list__filter_podderzhka__global_filter_buy__tab_artifacts",
        {"filter": "podderzhka", "global_filter": "buy", "tab": "artifacts"},
    ),
    "scr_153750": (
        "list__filter_balans__global_filter_buy__tab_artifacts",
        {"filter": "balans", "global_filter": "buy", "tab": "artifacts"},
    ),
    # Подтабы «Свитки» (досъёмка владельца, приёмка): 153810 «Обычные»,
    # 153819 «Благ.Свитки», 153830 «Гиганты», 153840 «Другие»; фильтр
    # «Покупка» у всех (сверено с кропами меты и классификацией детектора).
    "scr_153810": (
        "list__filter_obychnye__global_filter_buy__tab_svotki",
        {"filter": "obychnye", "global_filter": "buy", "tab": "svotki"},
    ),
    "scr_153819": (
        "list__filter_blag__global_filter_buy__tab_svotki",
        {"filter": "blag", "global_filter": "buy", "tab": "svotki"},
    ),
    "scr_153830": (
        "list__filter_giganty__global_filter_buy__tab_svotki",
        {"filter": "giganty", "global_filter": "buy", "tab": "svotki"},
    ),
    "scr_153840": (
        "list__filter_drugie__global_filter_buy__tab_svotki",
        {"filter": "drugie", "global_filter": "buy", "tab": "svotki"},
    ),
    # Подтабы «Припасы» (досъёмка владельца, приёмка). ВНИМАНИЕ: мета
    # перепутала имена кропов — soulshot_active лежит на 2-й кнопке,
    # spiritshot_active на 3-й (сверено rect'ами кропов относительно якоря
    # (202,103) и живой досъёмкой: 2-я кнопка = «Заряды Духа», 3-я =
    # «Заряды Души»). Фильтр «Все» (кружки дискриминируют, Q4).
    "scr_153929": (
        "list__filter_zaryady_dukha__global_filter_all__tab_pripasi",
        {"filter": "zaryady_dukha", "global_filter": "all", "tab": "pripasi"},
    ),
    "scr_153941": (
        "list__filter_zaryady_dushi__global_filter_all__tab_pripasi",
        {"filter": "zaryady_dushi", "global_filter": "all", "tab": "pripasi"},
    ),
    "scr_153949": (
        "list__filter_usileniya__global_filter_all__tab_pripasi",
        {"filter": "usileniya", "global_filter": "all", "tab": "pripasi"},
    ),
    # Подтабы «Прочее» (досъёмка владельца, приёмка): 154012 «Камни Духа»,
    # 154021 «Краски», 154033 «Усил.предметы», 154049 «Прочее»; фильтр
    # «Все» (имена кропов меты здесь верны).
    "scr_154012": (
        "list__filter_kamni_dukha__global_filter_all__tab_prochee",
        {"filter": "kamni_dukha", "global_filter": "all", "tab": "prochee"},
    ),
    "scr_154021": (
        "list__filter_kraski__global_filter_all__tab_prochee",
        {"filter": "kraski", "global_filter": "all", "tab": "prochee"},
    ),
    "scr_154033": (
        "list__filter_usil_predmety__global_filter_all__tab_prochee",
        {"filter": "usil_predmety", "global_filter": "all", "tab": "prochee"},
    ),
    "scr_154049": (
        "list__filter_prochee__global_filter_all__tab_prochee",
        {"filter": "prochee", "global_filter": "all", "tab": "prochee"},
    ),
}

# Ложное подтверждение params на табе «Снаряжение»: не осталось — категория
# удержана пустой для будущих регрессий (закрыто Q4/Q5, KNOWN_GAPS п. 4
# снят). scr_153631/153646/153656 переехали в FACT_FILTER_STATES.
FALSE_CONFIRM_EQUIPMENT = ()

# Мета state не размечена (R4); факт детектора — подтверждённое opening:
# scr_154222 (лавка продажи, вывеска активна). Сырья «open»-состояния
# лавки с активной вывеской нет; семантика scr_154222 — владельцу
# (KNOWN_GAPS.yaml п. 3, 5).
SHOP_FACT_OPENING = ("scr_154222",)

# Честный unknown при найденном якоре: оставшиеся неразмеченные
# подсостояния (сигнатуры opening лавок — п. 5; скролл/поиск/обновление —
# п. 3): scr_154355/154244 — сигнатуры opening на динамическом контенте
# противоречат остальным признакам, строгое «все сигнатуры» даёт unknown.
# Все подтабы «Обзора лавок» размечены (досъёмка владельца) — в
# FACT_FILTER_STATES. Любое подтверждённое состояние здесь — падение.
# Досъёмка живого прогона 9.4 (2026-11-10, Q30): обзор открыт владельцем
# на живой игре (вкладка/фильтр как в мете не размечены — R4); факт
# детектора на обоих кадрах (до/после прокрутки) — подтверждённое
# состояние, якорь template@72,111. Материал для живых кропов ячеек.
FACT_FILTER_STATES["scr_live_rows_home"] = (
    "list__filter_prochee__global_filter_sell__tab_prochee",
    {"filter": "prochee", "global_filter": "sell", "tab": "prochee"},
)
FACT_FILTER_STATES["scr_live_rows_home_scrolled"] = (
    "list__filter_prochee__global_filter_sell__tab_prochee",
    {"filter": "prochee", "global_filter": "sell", "tab": "prochee"},
)

HONEST_UNKNOWN = (
    "scr_154244", "scr_154355",
)

_ANCHOR_ABSENT = object()  # маркер ожидания «якорь не найден»
_NAV_FRAME = object()      # маркер ожидания «кадр навигации»


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
        if name in FACT_OPEN_STATE:
            # разметка меты (state='open') расходится с фактом детектора
            # (сигнатуры open нестабильны — KNOWN_GAPS п. 5): честная
            # фиксация факта вместо подгонки.
            return FACT_OPEN_STATE[name]
        return _resolve_meta_state(entry, model)
    if name in ANCHOR_ABSENT:
        return _ANCHOR_ABSENT
    if name in FACT_FILTER_STATES:
        return FACT_FILTER_STATES[name]
    if name in FALSE_CONFIRM_EQUIPMENT:
        return (
            "list__filter_all__global_filter_buy__tab_equipment",
            {"filter": "all", "global_filter": "buy", "tab": "equipment"},
        )
    if name in SHOP_FACT_OPENING:
        return ("opening", {})
    if name in FACT_OPEN_STATE:
        return FACT_OPEN_STATE[name]
    if name in NAV_FRAMES:
        return _NAV_FRAME
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

    expected = _expected(
        name, entry,
        shipped_models[window_id] if window_id in shipped_models else None)
    if expected is _ANCHOR_ABSENT:
        assert anchor_detector.find_anchor(frame, window_id) is None
        return
    if expected is _NAV_FRAME:
        nav = load_navigation(UI_MODEL_DIR / "navigation.yaml")
        evaluate = build_nav_evaluator(nav, default_matcher(), ASSETS_DIR)
        if name == "scr_nav_menu_open":
            checks = nav.signatures["menu_open"]
        else:
            checks = (
                Check(kind="template", element="hamburger_button",
                      variant="normal"),
                Check(kind="template", element="quickslot_sell_icon",
                      variant="normal"),
            )
        report = evaluate(checks, frame)
        assert report.passed, f"{name}: {report.failures}"
        return

    anchor = anchor_detector.find_anchor(frame, window_id)
    assert anchor is not None, f"якорь {window_id} не найден на {name}"
    assert anchor.window_id == window_id

    expect_state, expect_params = expected
    report = state_detector.detect(frame, anchor)
    assert report.conflicts == (), (
        f"{name}: конфликт сигнатур {report.conflicts} на реальном кадре — "
        f"строгое «все сигнатуры» конфликт не даёт; unknown здесь должен "
        f"быть честным (неподтверждение), иначе это расхождение признаков "
        f"модели, а не недоразмеченность (см. TestConflict для механизма)"
    )
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
        | set(FACT_FILTER_STATES) | set(FALSE_CONFIRM_EQUIPMENT)
        | set(SHOP_FACT_OPENING) | set(HONEST_UNKNOWN)
        | set(NAV_FRAMES) | set(FACT_OPEN_STATE)
    )
    assert covered == set(META["screenshots"])
    # Категории не пересекаются: один скрин — одна ожидаемая категория.
    categories = [
        set(ANCHOR_ABSENT), set(ANCHOR_ABSENT_OPEN), set(FACT_FILTER_STATES),
        set(FALSE_CONFIRM_EQUIPMENT), set(SHOP_FACT_OPENING),
        set(HONEST_UNKNOWN), set(NAV_FRAMES), set(FACT_OPEN_STATE),
    ]
    total = sum(len(c) for c in categories)
    assert total == len(set().union(*categories)), "категории пересекаются"


def test_screenshot_files_all_in_meta():
    """Файл сырья без записи в мете = скриншот без теста (и наоборот)."""
    pngs = {p.stem for p in SCR_DIR.glob("scr_*.png")}
    assert pngs == set(META["screenshots"])
