"""Тесты мониторинга своей лавки monitor_shop (задача 6.3, spec
market-scenarios «Сценарий мониторинга своей лавки»): детекция продажи
перечитыванием окна лавки (OCR количества/прибыли со сравнением снимков,
не чатом), базовая линия монитора, событие sold (callback-инъекция).

Монитор — ЯВНОЕ исключение из правила «сценарии без внутреннего состояния»
(spec market-scenarios: внутреннее состояние допускается только мониторам/
наблюдателям — базовая линия для детекции «продано»).

Детерминизм (инвариант 7): ScriptedScreen (кадры-скрипты с цветовыми
маркерами показателей), ColorOcr (текст по маркеру), скриптованный
find_anchor, конфиг с фиксированными порогами.

Guard-семантика: детекция НЕ через чат — монитор вообще не исполняет
действий (ни кликов, ни скролла, ни HID); guard-тесты смотрят исходник.
"""
import ast
import hashlib
import pathlib

import numpy as np
import pytest

from l2core.config import CoreConfig, OcrConfig, TimingConfig
from l2core.detector.anchor import Anchor
from l2core.model import (
    GrammarSpec,
    OcrRegionSpec,
    Rect,
    UiModel,
    WindowSpec,
)
from l2core.scenarios.monitor import (
    MonitorError,
    MonitorPollResult,
    ShopMonitor,
    SoldPayload,
)
from tests.test_rows import ColorOcr  # noqa: F401  (переиспользуем фейк)

ANCHOR = Anchor(window_id="shop_sell", x=100, y=50, score=0.99,
                method="template")
WINDOW_SIZE = (200, 160)
FRAME_SIZE = (400, 300)  # (w, h) — вмещает окно со смещением якоря

# Показатели лавки: количество проданного и прибыль (OCR-регионы НЕ
# размечены в отгруженной модели shop_sell — KNOWN_GAPS; rect'ы передаёт
# стратегия, в координатах окна от якоря).
REGION_SOLD = OcrRegionSpec(
    id="sold_count",
    rect=Rect(40, 60, 30, 12),
    grammar=GrammarSpec(name="digits", kind="digits"),
)
REGION_REVENUE = OcrRegionSpec(
    id="revenue",
    rect=Rect(40, 90, 40, 12),
    grammar=GrammarSpec(name="price", kind="price", min_value=0),
)
REGIONS = {"sold_count": REGION_SOLD, "revenue": REGION_REVENUE}

# Маркеры показателей (цвет -> (текст, conf)); неизвестный цвет -> ("???", 0.1).
C_SOLD_0 = (250, 120, 40)
C_SOLD_3 = (40, 220, 60)
C_SOLD_5 = (60, 80, 230)
C_SOLD_8 = (230, 60, 180)
C_REV_0 = (200, 40, 200)
C_REV_150 = (20, 200, 200)
C_REV_240 = (90, 140, 60)
C_UNKNOWN = (1, 2, 3)
OCR_TEXTS = {
    C_SOLD_0: ("0", 0.99),
    C_SOLD_3: ("3", 0.99),
    C_SOLD_5: ("5", 0.99),
    C_SOLD_8: ("8", 0.99),
    C_REV_0: ("0", 0.99),
    C_REV_150: ("150000", 0.99),
    C_REV_240: ("240000", 0.99),
}


def make_model(ocr_regions=None) -> UiModel:
    return UiModel(
        path="ui_model/shop_sell.yaml",
        window=WindowSpec(id="shop_sell", title="Торговая лавка (Продажа)",
                          size=WINDOW_SIZE),
        elements={},
        states={},
        transitions=(),
        ocr_regions=dict(ocr_regions or {}),
        table_geometry=None,
    )


def make_frame(sold_color=None, revenue_color=None) -> np.ndarray:
    """Кадр с цветовыми маркерами показателей в регионах (коорд. кадра)."""
    frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 30, dtype=np.uint8)
    for region, color in ((REGION_SOLD, sold_color),
                          (REGION_REVENUE, revenue_color)):
        if color is None:
            continue
        r = region.rect
        frame[ANCHOR.y + r.y:ANCHOR.y + r.y + r.height,
              ANCHOR.x + r.x:ANCHOR.x + r.x + r.width] = color
    return frame


class ScriptedScreen:
    """Экран-скрипт: выдаёт текущий кадр из держателя (без действий)."""

    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.grabs = 0

    def grab(self) -> np.ndarray:
        self.grabs += 1
        return self.frame


def make_monitor(screen, ocr=None, *, regions=REGIONS,
                 on_sold=None, cancelled=None, model=None):
    return ShopMonitor(
        model=model if model is not None else make_model(),
        ocr=ocr if ocr is not None else ColorOcr(OCR_TEXTS),
        screen=screen,
        find_anchor=lambda frame: ANCHOR,
        config=CoreConfig(timing=TimingConfig(),
                          ocr=OcrConfig(confidence_threshold=0.85,
                                        reread_count=1)),
        regions=regions,
        counter_region="sold_count",
        revenue_region="revenue",
        on_sold=on_sold,
        cancelled=cancelled,
    )


class TestBaseline:
    def test_capture_baseline_reads_current_values(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        monitor = make_monitor(screen)
        result = monitor.capture_baseline()
        assert isinstance(result, MonitorPollResult)
        assert not result.cancelled and not result.failures
        assert monitor.baseline is not None
        assert monitor.baseline.values == {"sold_count": 0, "revenue": 0}
        assert monitor.baseline.snapshot_id == \
            hashlib.sha1(np.ascontiguousarray(screen.frame).tobytes()).hexdigest()

    def test_poll_without_baseline_is_refused(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        monitor = make_monitor(screen)
        result = monitor.poll()
        assert result.sold is None
        assert result.failures
        assert "baseline" in result.failures[0]

    def test_capture_baseline_anchor_lost_keeps_no_baseline(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        monitor = make_monitor(screen)
        monitor.find_anchor = lambda frame: None
        result = monitor.capture_baseline()
        assert result.failures
        assert monitor.baseline is None


class TestSoldDetection:
    def test_no_change_no_event(self):
        screen = ScriptedScreen(make_frame(C_SOLD_5, C_REV_150))
        monitor = make_monitor(screen)
        assert not monitor.capture_baseline().failures
        result = monitor.poll()
        assert result.sold is None
        assert not result.degraded and not result.failures

    def test_increase_emits_sold_with_delta(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        events = []
        monitor = make_monitor(screen, on_sold=events.append)
        assert not monitor.capture_baseline().failures
        screen.frame = make_frame(C_SOLD_5, C_REV_150)
        result = monitor.poll()
        assert result.sold is not None
        payload = result.sold
        assert isinstance(payload, SoldPayload)
        assert payload.sold_count == 5      # суммарно продано (снимок)
        assert payload.sold_delta == 5      # дельта относительно baseline
        assert payload.revenue == 150000
        assert payload.revenue_delta == 150000
        assert payload.snapshot_id == \
            hashlib.sha1(np.ascontiguousarray(screen.frame).tobytes()).hexdigest()
        assert [payload] == events          # событие — через callback

    def test_baseline_updated_after_sale_next_sale_has_fresh_delta(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        events = []
        monitor = make_monitor(screen, on_sold=events.append)
        monitor.capture_baseline()
        screen.frame = make_frame(C_SOLD_5, C_REV_150)
        monitor.poll()
        assert monitor.baseline.values == {"sold_count": 5, "revenue": 150000}
        # Повторный poll без новых продаж — события нет.
        assert monitor.poll().sold is None
        # Следующая продажа: дельта считается от обновлённой линии.
        screen.frame = make_frame(C_SOLD_8, C_REV_240)
        result = monitor.poll()
        assert result.sold.sold_count == 8
        assert result.sold.sold_delta == 3
        assert result.sold.revenue_delta == 90000
        assert len(events) == 2

    def test_decrease_is_relist_not_sold(self):
        screen = ScriptedScreen(make_frame(C_SOLD_5, C_REV_150))
        events = []
        monitor = make_monitor(screen, on_sold=events.append)
        monitor.capture_baseline()
        # Переразметка лавки: показатели упали (снятие/перевыставление).
        screen.frame = make_frame(C_SOLD_3, C_REV_0)
        result = monitor.poll()
        assert result.sold is None
        assert result.diagnostics           # диагностика об уменьшении
        assert not events
        # Новая линия принята: повторное чтение без изменений — не продажа.
        screen.frame = make_frame(C_SOLD_3, C_REV_0)
        assert monitor.poll().sold is None

    def test_decrease_then_sale_from_new_baseline(self):
        screen = ScriptedScreen(make_frame(C_SOLD_5, C_REV_150))
        events = []
        monitor = make_monitor(screen, on_sold=events.append)
        monitor.capture_baseline()
        screen.frame = make_frame(C_SOLD_3, C_REV_0)
        monitor.poll()                      # переразметка: линия := 3
        screen.frame = make_frame(C_SOLD_8, C_REV_240)
        result = monitor.poll()
        assert result.sold.sold_delta == 5  # 8 - 3, а не 8 - 5

    def test_counter_ocr_failure_degrades_without_event(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        events = []
        monitor = make_monitor(screen, on_sold=events.append)
        monitor.capture_baseline()
        # Счётчик не читается (низкий confidence) — продажу подтвердить нельзя.
        screen.frame = make_frame(C_UNKNOWN, C_REV_150)
        result = monitor.poll()
        assert result.sold is None
        assert result.degraded
        assert result.failures
        assert "sold_count" in result.failures[0]
        assert not events
        # Базовая линия НЕ сдвинута ложным чтением.
        assert monitor.baseline.values == {"sold_count": 0, "revenue": 0}

    def test_revenue_ocr_failure_still_sold_with_empty_revenue(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        events = []
        monitor = make_monitor(screen, on_sold=events.append)
        monitor.capture_baseline()
        # Прибыль не читается, счётчик вырос: продажа есть, прибыль честно None.
        screen.frame = make_frame(C_SOLD_5, C_UNKNOWN)
        result = monitor.poll()
        assert result.sold is not None
        assert result.sold.sold_delta == 5
        assert result.sold.revenue is None
        assert result.sold.revenue_delta is None
        assert result.degraded              # честная деградация зафиксирована
        assert len(events) == 1

    def test_anchor_lost_poll_degrades_without_event(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        events = []
        monitor = make_monitor(screen, on_sold=events.append)
        monitor.capture_baseline()
        monitor.find_anchor = lambda frame: None
        screen.frame = make_frame(C_SOLD_5, C_REV_150)
        result = monitor.poll()
        assert result.sold is None
        assert result.degraded and result.failures
        assert not events


class TestConfiguration:
    def test_counter_region_required(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        with pytest.raises(MonitorError, match="sold_count"):
            make_monitor(screen, regions={})

    def test_counter_region_must_be_numeric(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        regions = {"sold_count": OcrRegionSpec(
            id="sold_count", rect=Rect(0, 0, 10, 10),
            grammar=GrammarSpec(name="text", kind="text"))}
        with pytest.raises(MonitorError, match="грамматик"):
            make_monitor(screen, regions=regions)

    def test_revenue_region_resolved_from_model(self):
        # Регион прибыли из секции ocr_regions модели (параметр не нужен).
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        model = make_model(ocr_regions={"revenue": REGION_REVENUE})
        events = []
        monitor = make_monitor(screen, on_sold=events.append, model=model,
                               regions={"sold_count": REGION_SOLD})
        monitor.capture_baseline()
        screen.frame = make_frame(C_SOLD_5, C_REV_150)
        assert monitor.poll().sold.revenue == 150000

    def test_cancelled_poll_reads_nothing(self):
        screen = ScriptedScreen(make_frame(C_SOLD_0, C_REV_0))
        monitor = make_monitor(screen, cancelled=lambda: True)
        result = monitor.poll()
        assert result.cancelled
        assert screen.grabs == 0            # кадр вообще не читался


class TestNoChatNoActionsGuard:
    """Guard: детекция продажи НЕ через чат — монитор не исполняет действий
    вообще (spec market-scenarios «Сценарий мониторинга своей лавки»: детекция
    перечитыванием окна, не чатом; ни кликов, ни скролла — чтение только
    через OCR-инъекции)."""

    def _source(self):
        return pathlib.Path(__file__).parents[1] / "l2core" / "scenarios" / \
            "monitor.py"

    def test_no_action_imports(self):
        src = self._source().read_text(encoding="utf-8")
        for forbidden in ("adapters.hid", "primitives.click",
                          "primitives.scroll", "primitives.drag",
                          "primitives.enter", "chat", "чат"):
            assert forbidden not in src, \
                f"монитор не должен трогать действия/чат: найдено {forbidden!r}"

    def test_no_action_calls_in_ast(self):
        tree = ast.parse(self._source().read_text(encoding="utf-8"))
        forbidden_attrs = {"click", "scroll", "drag", "key_press", "key_down",
                           "key_up", "move_rel", "mouse_down", "mouse_up",
                           "type_text", "release_all"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attrs, (
                    f"вызов действия {node.func.attr!r} в мониторе "
                    f"(строка {node.lineno})")
