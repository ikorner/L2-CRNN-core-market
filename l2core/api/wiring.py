"""Сборка провайдеров API из детекторов и адаптеров (задача 7.1,
l2core/api/wiring.py) — ЕДИНАЯ ТОЧКА СБОРКИ wiring: здесь провайдеры
сценариев 6.x и примитивов 5.x (find_anchor / window_status / run_checks /
goal_reached / current_window / close_point / hud_locator / detect_state /
health-провайдеры) собираются из реальных детекторов 3.4/3.5 и адаптеров.

Слои:
- ApiWiring — типизированный контейнер провайдеров, который потребляет
  фасад L2CoreAPI (l2core/api/core.py). Собирается либо build_wiring
  поверх живых детекторов, либо вручную (тесты на фейках — паттерн
  сценариев 6.x: адаптеры/провайдеры передаются готовыми, ядро их
  не создаёт, кроме этой точки сборки);
- build_wiring — сборка поверх AnchorDetector (3.3) / StateDetector (3.4) /
  WindowStatusDetector (3.5) и matcher'а; детекторы можно подменить
  (тесты на скриптованных кадрах);
- create_api — фабрика живых адаптеров: MssScreen, HidDevice, TorchOcr
  (create_ocr_client), PyWin32Gateway (DPI-aware, курсор, activate_game 5.2),
  модели из ui_model/; возвращает готовый L2CoreAPI.

КОНВЕНЦИИ ПРОВАЙДЕРОВ (контракт фасада):
- find_anchor(frame, window_id) — свежий поиск якоря окна (инъекция
  двухаргументная: один детектор на все окна, связывание по id — здесь);
- window_status(frame, window_id) — статус 3.5 окна;
- run_checks(checks, frame, anchor) — оценка verify-наборов переходов той
  же семантикой, что сигнатуры состояний (StateDetector.evaluate — точка
  переиспользования, без неё wiring дублировал бы _evaluate);
- current_window(frame) -> Optional[str] — какое окно торговли открыто;
- close_point(window_id, frame) — точка кнопки закрытия от якоря окна
  (market -> main_market_close_button, shop_sell -> close_sell_button;
  shop_buy кнопки не размечено -> None — fallback 5.8: ESC как закрытие);
- hud_locator(element, variant, frame) — абсолютная точка HUD-иконки:
  template matching в search_area + click_point варианта;
- goal_reached(target) -> Callable[[frame], bool] — цель recover 5.8:
  якорь целевого окна найден (окно открыто);
- health-провайдеры: game_window (находилка 3.2), ocr_health / hid_health
  (живость адаптеров; дефолты честные: локальный OCR жив по построению —
  модель загружена при создании, HID — через is_connected, если есть).

Инвариант «кэша позиции между кадрами нет» (3): провайдеры держат только
детекторы (их внутреннее состояние стабильности 3.5 — монитору разрешено,
design.md решение 8); новых кэшей сборка не вводит.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from l2core.adapters.hid import HidDevice
from l2core.adapters.ocr import create_ocr_client
from l2core.adapters.screen import MssScreen
from l2core.adapters.template import TemplateMatcher, default_matcher
from l2core.adapters.win32 import (
    GameWindow,
    GameWindowFinder,
    PyWin32Gateway,
    Win32Gateway,
    ensure_dpi_aware,
)
from l2core.api.core import L2CoreAPI
from l2core.api.dryrun import DryRunHid
from l2core.config import CoreConfig, load_config
from l2core.detector.anchor import Anchor, AnchorDetector, _to_gray
from l2core.detector.states import StateDetector, StateReport
from l2core.detector.status import WindowStatus, WindowStatusDetector
from l2core.model import ElementSpec, OcrRegionSpec, UiModel, load_model
from l2core.adapters.table_segmentation import TableSegmentation
from l2core.api.actionlog import ActionLogger
from l2core.primitives.activate import ActivateResult, activate_game
from l2core.primitives.click import CheckReport, CheckRunner, JitterRng
from l2core.primitives.outcome import IrreversibleGuard
from l2core.primitives.recover import RecoverResult

__all__ = ["ApiWiring", "CLOSE_ELEMENTS", "build_wiring", "create_api"]

# Кнопки закрытия окон по моделям (конвенция recover 5.8): shop_buy кнопки
# закрытия в модели нет — close_point вернёт None, recover честно падает в
# fallback «ESC как закрытие».
CLOSE_ELEMENTS = {
    "market": "main_market_close_button",
    "shop_sell": "close_sell_button",
}

_MODEL_NAMES = ("market", "shop_buy", "shop_sell")


@dataclass(frozen=True)
class ApiWiring:
    """Провайдеры и адаптеры, из которых фасад L2CoreAPI композирует
    вызовы. Готовится build_wiring (живое) или вручную (тесты на фейках —
    тот же паттерн, что у сценариев 6.x). Поля-регионы (search_region /
    price_region / shop_regions) закрывают KNOWN_GAPS отгруженных моделей:
    OCR-регионы поля поиска, поля цены и показателей лавки не размечены —
    их передаёт стратегия/сборка (R4: ядро не выдумывает константы)."""

    config: CoreConfig
    screen: object
    hid: object
    ocr: object
    market: UiModel
    shop_buy: UiModel
    shop_sell: UiModel
    hud_model: UiModel                       # модель с HUD-иконками (market)
    cursor_pos: Callable[[], Tuple[int, int]]
    activate: Callable[[], ActivateResult]
    find_anchor: Callable[[np.ndarray, str], Optional[Anchor]]
    window_status: Callable[[np.ndarray, str], WindowStatus]
    run_checks: CheckRunner
    current_window: Callable[[np.ndarray], Optional[str]]
    close_point: Callable[[str, np.ndarray], Optional[Tuple[int, int]]]
    hud_locator: Callable[[ElementSpec, Optional[str], np.ndarray],
                          Optional[Tuple[int, int]]]
    goal_reached: Callable[[str], Callable[[np.ndarray], bool]]
    game_window: Callable[[], GameWindow]
    ocr_health: Callable[[], bool]
    hid_health: Callable[[], bool]
    detect_state: Optional[Callable[[np.ndarray, Anchor], StateReport]] = None
    search_region: Optional[OcrRegionSpec] = None
    price_region: Optional[OcrRegionSpec] = None
    shop_regions: Optional[Mapping[str, OcrRegionSpec]] = None
    segmentation: Optional[TableSegmentation] = None
    free_area_point: Optional[Tuple[int, int]] = None
    on_escalation: Optional[Callable[[RecoverResult], None]] = None
    cancelled: Optional[Callable[[], bool]] = None
    note_activity: Optional[Callable[[], None]] = None  # kill switch 8.1
    action_logger: Optional[ActionLogger] = None  # JSONL-лог действий (8.3)
    rng: Optional[JitterRng] = None
    guard: IrreversibleGuard = field(default_factory=IrreversibleGuard)
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    dry_run: bool = False  # режим без HID (8.2): действия только логируются


def _load_hud_templates(models: Mapping[str, UiModel], assets_dir: Path) -> dict:
    """Шаблоны HUD-элементов (element_id, variant) -> (серый шаблон,
    элемент, вариант). HUD-элементы живут в модели market (icon_sample с
    search_area); ошибки ссылок/файлов — падение при сборке (разметчик)."""
    import cv2

    templates = {}
    for model in models.values():
        for element_id, element in model.elements.items():
            if not element.hud:
                continue
            for variant_name, spec in element.variants.items():
                path = Path(assets_dir) / spec.template
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(f"шаблон HUD-иконки не найден: {path}")
                templates[(element_id, variant_name)] = (
                    np.asarray(image, dtype=np.float64), element, spec)
    return templates


def build_wiring(*, config: CoreConfig,
                 models: Mapping[str, UiModel],
                 screen: object, hid: object, ocr: object,
                 matcher: Optional[TemplateMatcher], assets_dir,
                 cursor_pos: Callable[[], Tuple[int, int]],
                 activate: Callable[[], ActivateResult],
                 game_finder: GameWindowFinder,
                 anchor_detector: Optional[AnchorDetector] = None,
                 state_detector: Optional[StateDetector] = None,
                 status_detector: Optional[WindowStatusDetector] = None,
                 search_region: Optional[OcrRegionSpec] = None,
                 price_region: Optional[OcrRegionSpec] = None,
                 shop_regions: Optional[Mapping[str, OcrRegionSpec]] = None,
                 segmentation: Optional[TableSegmentation] = None,
                 hud_threshold: float = 0.9,
                 ocr_health: Optional[Callable[[], bool]] = None,
                 hid_health: Optional[Callable[[], bool]] = None,
                 free_area_point: Optional[Tuple[int, int]] = None,
                 on_escalation: Optional[Callable[[RecoverResult], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None,
                 note_activity: Optional[Callable[[], None]] = None,
                 action_logger: Optional[ActionLogger] = None,
                 rng: Optional[JitterRng] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 dry_run: Optional[bool] = None) -> ApiWiring:
    """Собрать провайдеров фасада из детекторов и адаптеров.

    models — модели трёх окон (market/shop_buy/shop_sell); детекторы по
    умолчанию строятся на них же (matcher/assets_dir), при желании
    подменяются (тесты на скриптованных кадрах). dry_run (8.2): None —
    флаг из config.dry_run, иначе явное значение (create_api передаёт
    эффективный флаг с учётом приоритета параметр > конфиг).
    """
    missing = [name for name in _MODEL_NAMES if name not in models]
    if missing:
        raise ValueError(f"build_wiring: не заданы модели окон {missing} "
                         f"(нужны {_MODEL_NAMES})")
    assets_dir = Path(assets_dir)
    matcher = matcher or default_matcher()
    anchor_detector = anchor_detector or AnchorDetector.for_windows(
        models, matcher, assets_dir)
    state_detector = state_detector or StateDetector(
        models, matcher, assets_dir, ocr_client=ocr)
    status_detector = status_detector or WindowStatusDetector(
        game_finder, anchor_detector, state_detector=state_detector)

    hud_templates = _load_hud_templates(models, assets_dir)

    def hud_locator(element: ElementSpec, variant: Optional[str],
                    frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Точка HUD-иконки: template matching в search_area; порог —
        hud_threshold (семейство порогов template-детекции 3.3/3.4);
        точка клика — click_point варианта (иначе центр шаблона)."""
        key = (element.id, variant or next(iter(element.variants), None))
        loaded = hud_templates.get(key)
        if loaded is None or element.search_area is None:
            return None
        template, _, spec = loaded
        area = element.search_area
        gray = _to_gray(np.asarray(frame))
        h, w = template.shape
        if (area.y < 0 or area.x < 0
                or area.y + area.height > gray.shape[0]
                or area.x + area.width > gray.shape[1]):
            return None
        crop = gray[area.y:area.y + area.height, area.x:area.x + area.width]
        if crop.shape[0] < h or crop.shape[1] < w:
            return None
        match = matcher.match(crop, template)
        if match.score < hud_threshold:
            return None
        click_point = spec.click_point or (w // 2, h // 2)
        return (area.x + match.x + click_point[0],
                area.y + match.y + click_point[1])

    def close_point(window_id: str,
                    frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Точка кнопки закрытия окна: якорь + rect + click_point варианта
        (как _click_point 5.3). Кнопки в модели нет (shop_buy) или якоря
        нет на кадре — None (recover: fallback/честный пропуск)."""
        element_id = CLOSE_ELEMENTS.get(window_id)
        if element_id is None:
            return None
        anchor = anchor_detector.find_anchor(frame, window_id)
        if anchor is None:
            return None
        model = models[window_id]
        element = model.elements.get(element_id)
        if element is None or element.rect is None:
            return None
        rect = element.rect
        spec = next(iter(element.variants.values()), None)
        click_point = spec.click_point if spec is not None else None
        if click_point is None:
            click_point = (rect.width // 2, rect.height // 2)
        return (anchor.x + rect.x + click_point[0],
                anchor.y + rect.y + click_point[1])

    def current_window(frame: np.ndarray) -> Optional[str]:
        """Какое окно торговли открыто на кадре (первый найденный якорь)."""
        for window_id in _MODEL_NAMES:
            if anchor_detector.find_anchor(frame, window_id) is not None:
                return window_id
        return None

    def run_checks(checks, frame: np.ndarray,
                   anchor: Anchor) -> CheckReport:
        """Оценка verify-набора перехода семантикой сигнатур 3.4
        (StateDetector.evaluate — общий matcher/пороги/OCR)."""
        results = state_detector.evaluate(tuple(checks), frame, anchor)
        failures = tuple(r.detail for r in results if r.matched is not True)
        return CheckReport(passed=not failures, failures=failures)

    if ocr_health is None:
        # Локальный OCR жив по построению: модель загружена при создании
        # TorchOcr; отдельного health-протокола OcrClient нет (честный
        # дефолт, не притворство проверкой).
        ocr_health = lambda: True
    if hid_health is None:
        is_connected = getattr(hid, "is_connected", None)
        hid_health = (lambda: bool(is_connected())) if callable(
            is_connected) else (lambda: True)

    return ApiWiring(
        config=config, screen=screen, hid=hid, ocr=ocr,
        market=models["market"], shop_buy=models["shop_buy"],
        shop_sell=models["shop_sell"], hud_model=models["market"],
        cursor_pos=cursor_pos, activate=activate,
        find_anchor=anchor_detector.find_anchor,
        window_status=status_detector.check,
        run_checks=run_checks,
        detect_state=state_detector.detect,
        current_window=current_window,
        close_point=close_point,
        hud_locator=hud_locator,
        goal_reached=lambda target: (
            lambda frame: anchor_detector.find_anchor(frame, target)
            is not None),
        game_window=game_finder.find,
        ocr_health=ocr_health,
        hid_health=hid_health,
        search_region=search_region, price_region=price_region,
        shop_regions=shop_regions, segmentation=segmentation,
        free_area_point=free_area_point, on_escalation=on_escalation,
        cancelled=cancelled, note_activity=note_activity,
        action_logger=action_logger,
        rng=rng, clock=clock, sleep=sleep,
        dry_run=config.dry_run if dry_run is None else dry_run)


def create_api(config, *, models_dir="ui_model", assets_dir="assets",
               model_names: Tuple[str, ...] = _MODEL_NAMES,
               screen=None, hid=None, ocr=None,
               gateway: Optional[Win32Gateway] = None,
               search_region: Optional[OcrRegionSpec] = None,
               price_region: Optional[OcrRegionSpec] = None,
               shop_regions: Optional[Mapping[str, OcrRegionSpec]] = None,
               segmentation: Optional[TableSegmentation] = None,
               owner_thread=None,
               cancelled: Optional[Callable[[], bool]] = None,
               note_activity: Optional[Callable[[], None]] = None,
               action_logger: Optional[ActionLogger] = None,
               rng: Optional[JitterRng] = None,
               clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep,
               dry_run: Optional[bool] = None) -> L2CoreAPI:
    """Фабрика живой сборки: конфиг (путь или CoreConfig), модели из
    ui_model/, живые адаптеры (MssScreen / HidDevice / TorchOcr /
    PyWin32Gateway) и детекторы 3.3–3.5; вернуть готовый L2CoreAPI.

    Адаптеры можно подменить (screen/hid/ocr/gateway) — точки инъекции
    тестов и не-Windows окружений; kill switch (8.1) подключается через
    cancelled. Действия HID выполняются только внутри блокирующих вызовов
    фасада из потока-владельца (owner_thread).

    Dry-run (8.2): эффективный флаг — параметр dry_run, если задан,
    иначе config.dry_run. Включённый режим поднимает DryRunHid вместо
    живого HidDevice (обращений к arduino нет, действия логируются
    как намерения); явно переданный hid побеждает режим — инъекция
    адаптера для тестов. Экран читается по-настоящему: verify работает
    на реальных кадрах, результаты примитивов честно уходят в
    FAILED/UNKNOWN (эффекта нет) — семантика для первых запусков.

    JSONL-лог действий (8.3): при config.log.enabled (дефолт) поднимается
    ActionLogger из config.log (dir/лимиты/политика; ts — от тех же часов
    сборки); явный action_logger побеждает конфиг. Каждый действие-метод
    фасада пишет запись (кадры до/после — только ключевые: не-OK и
    необратимые) — l2core/api/actionlog.py.
    """
    if isinstance(config, (str, Path)):
        config = load_config(config)
    effective_dry_run = config.dry_run if dry_run is None else dry_run
    models_dir = Path(models_dir)
    models = {name: load_model(models_dir / f"{name}.yaml")
              for name in model_names}
    gateway = gateway or PyWin32Gateway()
    ensure_dpi_aware(gateway)
    screen = screen or MssScreen()
    if hid is None:
        if effective_dry_run:
            hid = DryRunHid()
        else:
            hid = HidDevice(config.paths.arduino_port)
            hid.connect()   # порт + проверка прошивки (жизненный цикл — у фасада)
    ocr = ocr or create_ocr_client("torch", model_path=config.paths.model)
    if action_logger is None and config.log.enabled:
        action_logger = ActionLogger(
            config.log.dir, max_bytes=config.log.max_bytes,
            keep_files=config.log.keep_files, policy=config.log.policy,
            clock=clock)
    finder = GameWindowFinder(
        gateway, title=config.window.title,
        class_name=config.window.class_name)
    activate = lambda: activate_game(  # noqa: E731
        gateway, config=config, cancelled=cancelled, clock=clock, sleep=sleep)
    wiring = build_wiring(
        config=config, models=models, screen=screen, hid=hid, ocr=ocr,
        matcher=default_matcher(), assets_dir=Path(assets_dir),
        cursor_pos=lambda: gateway.get_cursor_pos() or (0, 0),
        activate=activate, game_finder=finder,
        search_region=search_region, price_region=price_region,
        shop_regions=shop_regions, segmentation=segmentation,
        cancelled=cancelled, note_activity=note_activity,
        action_logger=action_logger,
        rng=rng, clock=clock, sleep=sleep, dry_run=effective_dry_run)
    return L2CoreAPI(wiring, owner_thread=owner_thread)
