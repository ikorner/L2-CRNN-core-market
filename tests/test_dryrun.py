"""Тесты dry-run режима (задача 8.2, spec safety «Dry-run режим без HID»).

Конфиг-флаг dry_run (схема 1.3; валидация ключа — tests/test_config.py):
create_api поднимает DryRunHid вместо живого HidDevice (явный параметр
побеждает конфиг, явный hid побеждает оба); экран читается по-настоящему,
действия — только намерения в логе, verify честно показывает
FAILED/UNKNOWN: эффекта нет, детекция работает.

Thread-owner дисциплина не дублируется в адаптере: guard владения —
OwnerGuardedHid (7.2), композиция с dry-run проверяется отдельно.
"""
import dataclasses
import logging
import threading
from pathlib import Path

import pytest

from l2core.adapters.hid import HidDevice
from l2core.api.core import L2CoreAPI
from l2core.api.dryrun import DryRunHid, DryRunIntention
from l2core.api.hid_owner import HidOwnershipError, OwnerGuardedHid
from l2core.api.wiring import create_api
from l2core.config import CoreConfig
from l2core.primitives.outcome import ActionOutcome
from tests.test_api import (
    MARKET_ANCHOR, AnchorScript, FakeClock, World, make_wiring,
)


class TestDryRunHid:
    """Каждое действие — структурированное намерение в логе; к железу
    обращений нет (адаптер его вообще не содержит)."""

    def test_actions_recorded_as_intentions(self):
        hid = DryRunHid()
        hid.move_rel(3, -2)
        hid.click(1, 1, 50)
        hid.scroll(1, -1, 10)
        hid.key_press(0x1E)

        assert [i.action for i in hid.intentions] == [
            "move_rel", "click", "scroll", "key_press"]

    def test_intention_keeps_args(self):
        hid = DryRunHid()
        hid.click(1, 2, 40)

        (intention,) = hid.intentions
        assert intention.action == "click"
        assert intention.args == (1, 2, 40)
        assert intention.kwargs == {}

    def test_intention_keeps_kwargs(self):
        hid = DryRunHid()
        hid.click(1, count=2, hold_ms=30)

        (intention,) = hid.intentions
        assert intention.kwargs == {"count": 2, "hold_ms": 30}

    def test_release_all_is_noop_with_log(self):
        hid = DryRunHid()

        hid.release_all()   # не подключён — не бросает, железо не тронуто

        assert [i.action for i in hid.intentions] == ["release_all"]

    def test_all_action_methods_noop(self):
        hid = DryRunHid()
        hid.connect()
        hid.mouse_down()
        hid.mouse_up()
        hid.traj([(1, 2, 8)])
        hid.key_down(0x04)
        hid.key_up(0x04)
        hid.key_combo(0xE0, 0x12)
        hid.type_char("1")
        hid.close()

        assert [i.action for i in hid.intentions] == [
            "connect", "mouse_down", "mouse_up", "traj", "key_down",
            "key_up", "key_combo", "type_char", "close"]

    def test_queries_are_not_intentions(self):
        # Запросы состояния — не действия: в журнал намерений не пишутся,
        # адаптер отчитывается «живым» (dry-run HID-путь доступен по построению).
        hid = DryRunHid()

        assert hid.is_connected() is True
        assert hid.ping() is True
        assert hid.get_version() == (0, 0)
        assert hid.intentions == []

    def test_traj_capacity_matches_vendor(self):
        # Паритет валидации вендора: примитивы полагаются на тот же лимит.
        hid = DryRunHid()
        assert hid.TRAJ_MAX_POINTS == HidDevice.TRAJ_MAX_POINTS
        with pytest.raises(ValueError):
            hid.traj([])
        with pytest.raises(ValueError):
            hid.traj([(1, 1, 5)] * (HidDevice.TRAJ_MAX_POINTS + 1))
        hid.traj([(1, 1, 5)] * HidDevice.TRAJ_MAX_POINTS)
        assert [i.action for i in hid.intentions] == ["traj"]

    def test_every_intention_written_to_logger(self, caplog):
        hid = DryRunHid()
        with caplog.at_level(logging.INFO, logger="l2core.dry_run"):
            hid.click(1, 1, 50)
            hid.release_all()

        records = [r for r in caplog.records
                   if r.name == "l2core.dry_run"]
        assert len(records) == 2
        assert "click" in records[0].getMessage()
        assert "release_all" in records[1].getMessage()

    def test_external_intention_list_accepted(self):
        # Опциональная точка инъекции списка (тесты/стратегия держат свой
        # журнал намерений без чтения адаптера).
        journal = []
        hid = DryRunHid(intentions=journal)
        hid.key_press(0x28)

        assert len(journal) == 1
        assert journal[0].action == "key_press"
        assert hid.intentions is journal

    def test_metadata_attributes(self):
        hid = DryRunHid()
        assert hid.port_name
        assert isinstance(hid.version, tuple)


class TestCommandParity:
    """Guard-тест паритета командного поверхности: публичные методы
    вендорного HidDevice ⊆ DryRunHid (новая команда вендора в dry-run
    не пропадёт молча — та же дисциплина, что TestHidOwnerContract 7.2)."""

    def test_dryrun_covers_vendor_public_methods(self):
        vendor_public = frozenset(
            name for name, value in vars(HidDevice).items()
            if not name.startswith("_") and callable(value))
        dryrun_public = frozenset(
            name for name, value in vars(DryRunHid).items()
            if not name.startswith("_") and callable(value))
        missing = vendor_public - dryrun_public
        assert not missing, (
            f"команды вендорного HidDevice, не реализованные DryRunHid "
            f"(в dry-run пропали бы намерения): {sorted(missing)}")


class TestCreateApiDryRun:
    """Точка включения — create_api: флаг из конфига, явный параметр
    побеждает конфиг, явный hid побеждает режим (инъекция для тестов)."""

    @pytest.fixture
    def patched(self, monkeypatch):
        import l2core.api.wiring as wiring_module

        constructed = []

        class FakeScreen:
            def grab(self):
                import numpy as np
                return np.zeros((4, 4, 3), dtype=np.uint8)

            def close(self):
                pass

        class FakeHidDevice:
            def __init__(self, port=None, *args, **kwargs):
                self.port = port
                constructed.append(port)

            def connect(self):
                pass

            def is_connected(self):
                return True

        class FakeGateway:
            def find_windows(self, title, class_name):
                return [1]

            def get_client_rect(self, hwnd):
                return (0, 0, 1920, 1080)

            def is_iconic(self, hwnd):
                return False

            def get_foreground_window(self):
                return 1

            def set_foreground(self, hwnd):
                return True

            def set_process_dpi_aware(self):
                return True

            def get_cursor_pos(self):
                return (0, 0)

        monkeypatch.setattr(wiring_module, "MssScreen", FakeScreen)
        monkeypatch.setattr(wiring_module, "HidDevice", FakeHidDevice)
        monkeypatch.setattr(wiring_module, "create_ocr_client",
                            lambda *a, **kw: object())
        monkeypatch.setattr(wiring_module, "PyWin32Gateway", FakeGateway)
        repo = Path(__file__).resolve().parent.parent
        return {"constructed": constructed,
                "kwargs": dict(models_dir=repo / "ui_model",
                               assets_dir=repo / "assets")}

    def test_param_enables_dry_run(self, patched):
        api = create_api(CoreConfig(), dry_run=True, **patched["kwargs"])

        assert isinstance(api._w.hid, DryRunHid)
        assert patched["constructed"] == []      # живой HID не создан
        assert api.dry_run is True
        assert api._w.dry_run is True

    def test_config_flag_enables_dry_run(self, patched):
        config = CoreConfig(dry_run=True)
        api = create_api(config, **patched["kwargs"])

        assert isinstance(api._w.hid, DryRunHid)
        assert patched["constructed"] == []

    def test_explicit_param_wins_over_config(self, patched):
        config = CoreConfig(dry_run=True)
        api = create_api(config, dry_run=False, **patched["kwargs"])

        assert not isinstance(api._w.hid, DryRunHid)
        assert patched["constructed"] == ["COM3"]  # живой HID поднят
        assert api.dry_run is False

    def test_default_is_live_hid(self, patched):
        api = create_api(CoreConfig(), **patched["kwargs"])

        assert not isinstance(api._w.hid, DryRunHid)
        assert patched["constructed"] == ["COM3"]

    def test_explicit_hid_wins_over_dry_run(self, patched):
        class ForeignHid:
            def is_connected(self):
                return True

        foreign = ForeignHid()
        api = create_api(CoreConfig(), dry_run=True, hid=foreign,
                         **patched["kwargs"])

        assert api._w.hid is foreign
        assert api.dry_run is True   # флаг режима — про сборку, не про адаптер


class TestThreadOwnerComposition:
    """DryRunHid сам guard владения НЕ проверяет (одна точка дисциплины —
    OwnerGuardedHid 7.2): guard оборачивает dry-run адаптер как любое
    устройство; чужой поток отклоняется ДО адаптера — намерение не пишется."""

    def test_cross_thread_rejected_before_recording(self):
        hid = DryRunHid()
        guarded = OwnerGuardedHid(hid, owner_thread=threading.current_thread())
        outcome = {}

        def worker():
            try:
                guarded.click(1, 1, 50)
            except HidOwnershipError as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert isinstance(outcome.get("error"), HidOwnershipError)
        assert hid.intentions == []

    def test_owner_thread_records_through_guard(self):
        hid = DryRunHid()
        guarded = OwnerGuardedHid(hid, owner_thread=threading.current_thread())

        guarded.click(1, 1, 50)

        assert [i.action for i in hid.intentions] == ["click"]


class TestOpenMarketDryRun:
    """Поведенческий сценарий спеки «Действия логируются без выполнения»:
    полный контур api.open_market в dry-run — verify читает реальный
    (скриптованный) экран, к HID обращений нет, намерения залогированы."""

    def _api(self, world, hid, clock, hud_calls, anchors):
        wiring = make_wiring(world, hid, clock=clock, hud_calls=hud_calls,
                             anchor_script=AnchorScript(anchors))
        wiring = dataclasses.replace(wiring, cursor_pos=lambda: (0, 0),
                                     dry_run=True)
        return L2CoreAPI(wiring)

    def test_intention_logged_verify_reports_failed(self):
        # Экран «до действия»: якорь обзора виден (verify выполняется), но
        # клик не выполнялся — сигнатура не сходится: честный FAILED.
        world, clock = World(), FakeClock()
        hid = DryRunHid()
        hud_calls = []
        api = self._api(world, hid, clock, hud_calls,
                        {"market": MARKET_ANCHOR})

        result = api.open_market()

        assert not result.success
        assert result.outcome is ActionOutcome.FAILED
        assert not world.market_opened            # мир не мутировал
        assert hud_calls == ["market_pic"]        # verify-контур реальный
        assert [i.action for i in hid.intentions] == ["move_rel", "click"]

    def test_no_anchor_on_frame_is_unknown(self):
        # Окна на кадре нет: verify ни разу не выполнился — UNKNOWN
        # (эффект «клика» не подтверждён; в dry-run он и не посылался).
        world, clock = World(), FakeClock()
        hid = DryRunHid()
        api = self._api(world, hid, clock, [], {})

        result = api.open_market()

        assert not result.success
        assert result.outcome is ActionOutcome.UNKNOWN
        assert result.unconfirmed
        assert [i.action for i in hid.intentions] == ["move_rel", "click"]

    def test_already_open_window_reports_ok(self):
        # Окно уже открыто (мир выставлен снаружи): verify сходится по
        # реальному кадру — OK; намерение всё равно залогировано.
        world, clock = World(), FakeClock()
        world.market_opened = True
        hid = DryRunHid()
        api = self._api(world, hid, clock, [],
                        {"market": MARKET_ANCHOR})

        result = api.open_market()

        assert result.success
        assert result.outcome is ActionOutcome.OK
        assert [i.action for i in hid.intentions] == ["move_rel", "click"]

    def test_repeated_calls_accumulate_intentions(self):
        world, clock = World(), FakeClock()
        hid = DryRunHid()
        api = self._api(world, hid, clock, [],
                        {"market": MARKET_ANCHOR})

        api.open_market()
        api.open_market()

        actions = [i.action for i in hid.intentions]
        assert actions == ["move_rel", "click", "move_rel", "click"]


def test_dry_run_intention_is_frozen():
    intention = DryRunIntention(action="click", args=(1,), kwargs={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        intention.action = "move_rel"
