"""Тесты JSONL-лога действий (задача 8.3, spec replay-logging:
«JSONL-лог каждого действия»).

Покрытие:
- запись: поля записи (seq/ts/action/key/frames/verify), seq по счётчику,
  ts из инъецированных часов, нормализация action, attempts/failures;
- политика ключевых кадров: не-OK и необратимые — с кадрами, обычный OK —
  без кадров; policy.all_failures / policy.irreversible выключают кадры;
  отмена — не ключевая;
- ротация: превышение max_bytes → новый actions-NNNN.jsonl; keep_files —
  старшие файлы удаляются; ключевые записи старших файлов мигрируют в
  новейший (кадры сохранены); сиротские PNG чистятся;
- возобновление: logger на существующей директории продолжает счётчики;
- конфиг: секция log (enabled/dir/max_bytes/keep_files/policy) по схеме 1.3;
- интеграция: wiring.action_logger (по умолчанию None — логирование off),
  записи методов фасада (open_market FAILED/OK, start_sell необратимый),
  create_api поднимает logger из конфига.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from l2core.api.actionlog import ActionLogger, ActionLogError
from l2core.api.core import L2CoreAPI
from l2core.api.wiring import create_api
from l2core.config import ConfigError, CoreConfig, LogConfig, LogPolicy, \
    load_config
from l2core.primitives.click import VerifySpec
from l2core.primitives.outcome import ActionOutcome
from tests.test_api import (
    SELL_ANCHOR,
    AnchorScript,
    FakeClock,
    NavWorld,
    ScriptedHid,
    World,
    make_nav_evaluate,
    make_nav_hud_locator,
    make_nav_model,
    make_wiring,
)


def frame(seed):
    """Маленький детерминированный кадр BGR uint8."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[0, 0] = seed
    return img


def make_logger(log_dir, *, clock=None, **kwargs):
    return ActionLogger(log_dir, clock=clock or (lambda: 0.0), **kwargs)


def read_records(log_dir):
    """Все записи всех файлов лога, в порядке следования."""
    records = []
    for path in sorted(Path(log_dir).glob("actions-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def record_kwargs(**overrides):
    base = dict(action={"type": "click", "target": "market",
                        "args": {"element": "tab"}},
                success=True, outcome=ActionOutcome.OK)
    base.update(overrides)
    return base


class TestRecord:
    """Формат записи: seq/ts/action/key/frames/verify."""

    def test_record_fields(self, tmp_path):
        clock = FakeClock()
        alog = make_logger(tmp_path / "a", clock=clock.monotonic)
        clock.now = 1.5

        record = alog.record(**record_kwargs(attempts=2,
                                             failures=("не сошлось",)))

        assert record["seq"] == 1
        assert record["ts"] == 1.5            # ts — из инъецированных часов
        assert record["key"] is False
        assert record["action"] == {"type": "click", "target": "market",
                                    "args": {"element": "tab"}}
        assert record["frames"] == {"before": None, "after": None}
        assert record["verify"] == {
            "success": True, "outcome": "OK", "unconfirmed": False,
            "cancelled": False, "attempts": 2, "failures": ["не сошлось"]}

    def test_seq_increments_and_ts_follows_clock(self, tmp_path):
        clock = FakeClock()
        alog = make_logger(tmp_path / "a", clock=clock.monotonic)

        r1 = alog.record(**record_kwargs())
        clock.sleep(0.25)
        r2 = alog.record(**record_kwargs())

        assert (r1["seq"], r2["seq"]) == (1, 2)
        assert r2["ts"] - r1["ts"] == pytest.approx(0.25)

    def test_jsonl_line_written_to_file(self, tmp_path):
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir)
        alog.record(**record_kwargs())

        (path,) = log_dir.glob("actions-*.jsonl")
        line = json.loads(path.read_text(encoding="utf-8"))
        assert line["seq"] == 1
        assert line["action"]["type"] == "click"

    def test_action_normalized(self, tmp_path):
        alog = make_logger(tmp_path / "a")
        record = alog.record(
            action={"type": "drag", "args": {"points": ((1, 2), (3, 4)),
                                             "count": np.int64(2)}},
            success=True)

        assert record["action"]["args"] == {"points": [[1, 2], [3, 4]],
                                            "count": 2}

    def test_frame_rejected_if_not_image(self, tmp_path):
        alog = make_logger(tmp_path / "a")
        with pytest.raises(ActionLogError):
            alog.record(**record_kwargs(success=False, before=np.zeros(4)))


class TestKeyFramesPolicy:
    """Ключевые кадры: только не-OK и необратимые записи несут PNG."""

    def test_ok_record_has_no_frames(self, tmp_path):
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir)
        record = alog.record(**record_kwargs(before=frame(1), after=frame(2)))

        assert record["key"] is False
        assert record["frames"] == {"before": None, "after": None}
        assert not list((log_dir / "frames").glob("*.png"))  # PNG не пишутся

    def test_failed_record_keeps_frames(self, tmp_path):
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir)
        record = alog.record(**record_kwargs(
            success=False, outcome=ActionOutcome.FAILED,
            before=frame(1), after=frame(2)))

        assert record["key"] is True
        before_ref = record["frames"]["before"]
        after_ref = record["frames"]["after"]
        assert before_ref.startswith("frames/") and before_ref.endswith(".png")
        assert after_ref.startswith("frames/")
        assert (log_dir / before_ref).is_file()
        assert (log_dir / after_ref).is_file()

    def test_unknown_unconfirmed_is_key(self, tmp_path):
        alog = make_logger(tmp_path / "a")
        record = alog.record(**record_kwargs(
            success=False, outcome=ActionOutcome.UNKNOWN, unconfirmed=True,
            before=frame(1)))

        assert record["key"] is True
        assert record["frames"]["before"] is not None
        assert record["frames"]["after"] is None   # кадр-после не передан

    def test_cancelled_record_is_not_key(self, tmp_path):
        alog = make_logger(tmp_path / "a")
        record = alog.record(**record_kwargs(
            success=False, outcome=None, cancelled=True, before=frame(1)))

        assert record["key"] is False
        assert record["frames"] == {"before": None, "after": None}

    def test_irreversible_success_keeps_frames(self, tmp_path):
        alog = make_logger(tmp_path / "a")
        record = alog.record(**record_kwargs(irreversible=True,
                                             before=frame(1), after=frame(2)))

        assert record["key"] is True
        assert record["frames"]["before"] is not None

    def test_policy_all_failures_off(self, tmp_path):
        alog = make_logger(tmp_path / "a",
                           policy=LogPolicy(all_failures=False))
        record = alog.record(**record_kwargs(
            success=False, outcome=ActionOutcome.FAILED, before=frame(1)))

        assert record["key"] is False
        assert record["frames"]["before"] is None

    def test_policy_irreversible_off(self, tmp_path):
        alog = make_logger(tmp_path / "a",
                           policy=LogPolicy(irreversible=False))
        record = alog.record(**record_kwargs(irreversible=True,
                                             before=frame(1)))

        assert record["key"] is False
        assert record["frames"]["before"] is None

    def test_frames_deduplicated_by_content(self, tmp_path):
        # Одинаковые кадры (по содержимому) — один PNG (конвенция сессий 3.1:
        # дедуп по хэшу, имя файла — содержимое).
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir)
        record = alog.record(**record_kwargs(
            success=False, before=frame(7), after=frame(7)))

        assert record["frames"]["before"] == record["frames"]["after"]
        assert len(list((log_dir / "frames").glob("*.png"))) == 1


class TestRotation:
    """Ротация по размеру, keep_files, миграция ключевых записей, чистка."""

    def test_rotate_on_max_bytes(self, tmp_path):
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir)
        alog.record(**record_kwargs())
        size_one = (log_dir / "actions-0001.jsonl").stat().st_size

        alog = make_logger(log_dir, max_bytes=size_one + 10)
        alog.record(**record_kwargs())   # влезает в текущий файл
        alog.record(**record_kwargs())   # превышение -> новый файл

        files = sorted(p.name for p in log_dir.glob("actions-*.jsonl"))
        assert files == ["actions-0001.jsonl", "actions-0002.jsonl"]
        assert [r["seq"] for r in read_records(log_dir)] == [1, 2, 3]

    def test_keep_files_deletes_oldest(self, tmp_path):
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir, max_bytes=1, keep_files=2)
        for i in range(3):
            alog.record(**record_kwargs(action={"type": f"a{i}"}))

        files = sorted(p.name for p in log_dir.glob("actions-*.jsonl"))
        assert files == ["actions-0002.jsonl", "actions-0003.jsonl"]
        types = [r["action"]["type"] for r in read_records(log_dir)]
        assert types == ["a1", "a2"]   # запись a0 удалена со старшим файлом

    def test_key_records_migrated_with_frames(self, tmp_path):
        # Ключевая запись старшего файла при удалении мигрирует в новейший
        # файл (вместе со ссылками на кадры — ключевые кадры сохраняются).
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir, max_bytes=1, keep_files=1)
        alog.record(**record_kwargs(action={"type": "fail"},
                                    success=False,
                                    outcome=ActionOutcome.FAILED,
                                    before=frame(1), after=frame(2)))
        frame_ref = read_records(log_dir)[0]["frames"]["before"]

        alog.record(**record_kwargs(action={"type": "ok"}))
        alog.record(**record_kwargs(action={"type": "ok2"}))

        files = sorted(p.name for p in log_dir.glob("actions-*.jsonl"))
        assert files == ["actions-0003.jsonl"]
        records = read_records(log_dir)
        assert [r["action"]["type"] for r in records] == ["fail", "ok2"]
        assert records[0]["seq"] == 1                 # оригинальные seq/ts
        assert records[0]["frames"]["before"] == frame_ref
        assert (log_dir / frame_ref).is_file()       # ключевой кадр сохранён

    def test_key_frames_survive_repeated_rotation(self, tmp_path):
        # Миграция сохраняет ключевые записи и при повторных ротациях;
        # кадры ключевых записей не становятся сиротами.
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir, max_bytes=1, keep_files=1)
        alog.record(**record_kwargs(action={"type": "k"},
                                    success=False, before=frame(9)))
        kept = read_records(log_dir)[0]["frames"]["before"]

        alog.record(**record_kwargs(action={"type": "ok"}))

        assert (log_dir / kept).is_file()

    def test_orphan_frames_cleaned(self, tmp_path):
        # Сиротский PNG (ссылок из сохранённых записей нет — остаток
        # прерванной сессии) чистится при ротации.
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir, max_bytes=1, keep_files=5)
        orphan = log_dir / "frames" / "deadbeef.png"
        orphan.write_bytes(b"png")

        alog.record(**record_kwargs())

        assert not orphan.exists()

    def test_resume_continues_counters(self, tmp_path):
        # Logger на существующей директории: тот же текущий файл, seq
        # продолжается (перезапуск процесса не обнуляет журнал).
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir)
        alog.record(**record_kwargs())
        alog.record(**record_kwargs())

        alog2 = make_logger(log_dir)
        record = alog2.record(**record_kwargs())

        assert record["seq"] == 3
        files = sorted(p.name for p in log_dir.glob("actions-*.jsonl"))
        assert files == ["actions-0001.jsonl"]
        assert len(read_records(log_dir)) == 3

    def test_names_are_counter_based(self, tmp_path):
        # Детерминизм: имена файлов — по счётчику, без меток времени.
        log_dir = tmp_path / "a"
        alog = make_logger(log_dir, max_bytes=1)
        alog.record(**record_kwargs())
        alog.record(**record_kwargs())

        assert [p.name for p in log_dir.glob("actions-*.jsonl")] == \
            ["actions-0001.jsonl", "actions-0002.jsonl"]


class TestLogConfig:
    """Секция log конфигурации (схема 1.3): пути, лимиты, политика."""

    def test_defaults(self):
        cfg = CoreConfig()
        assert cfg.log.enabled is True
        assert cfg.log.dir == "logs/actions"
        assert cfg.log.max_bytes == 20_000_000
        assert cfg.log.keep_files == 5
        assert cfg.log.policy.all_failures is True
        assert cfg.log.policy.irreversible is True

    def test_yaml_section(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "log:\n"
            "  enabled: false\n"
            "  dir: var/actions\n"
            "  max_bytes: 1000\n"
            "  keep_files: 2\n"
            "  policy:\n"
            "    all_failures: false\n"
            "    irreversible: false\n",
            encoding="utf-8")

        cfg = load_config(cfg_path)

        assert cfg.log.enabled is False
        assert cfg.log.dir == "var/actions"
        assert cfg.log.max_bytes == 1000
        assert cfg.log.keep_files == 2
        assert cfg.log.policy == LogPolicy(all_failures=False,
                                           irreversible=False)

    def test_unknown_key_rejected(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("log:\n  verbosity: high\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(cfg_path)

    def test_unknown_policy_key_rejected(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("log:\n  policy:\n    everything: true\n",
                            encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(cfg_path)

    @pytest.mark.parametrize("section", [
        "log:\n  max_bytes: 0\n",
        "log:\n  keep_files: 0\n",
        "log:\n  dir: ''\n",
        "log:\n  enabled: 1\n",
        "log:\n  policy: все\n",
    ])
    def test_invalid_values_rejected(self, tmp_path, section):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(section, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(cfg_path)

    def test_shipped_config_yaml_matches_defaults(self):
        repo = Path(__file__).resolve().parent.parent
        cfg = load_config(repo / "config.yaml")
        assert cfg.log == LogConfig()


class TestFacadeIntegration:
    """Точка интеграции — методы фасада (граница: кадры до/после на входе
    и выходе ВЫЗОВА API, не вокруг отдельной HID-команды)."""

    def _api(self, tmp_path, world, *, clock=None, anchors=None, **wiring_kw):
        clock = clock or FakeClock()
        hid = ScriptedHid(clock, world)
        log_dir = tmp_path / "actions"
        alog = make_logger(log_dir)
        wiring = make_wiring(world, hid, clock=clock,
                             anchor_script=AnchorScript(anchors or {}),
                             **wiring_kw)
        wiring = dataclasses.replace(wiring, action_logger=alog)
        return L2CoreAPI(wiring), log_dir

    def _nav_api(self, tmp_path, nav_world, **wiring_kw):
        clock = FakeClock()
        hid = ScriptedHid(clock, nav_world)
        log_dir = tmp_path / "actions"
        alog = make_logger(log_dir)
        wiring = make_wiring(
            nav_world, hid, clock=clock,
            navigation=make_nav_model(),
            nav_evaluate=make_nav_evaluate(nav_world, []),
            hud_locator=make_nav_hud_locator(nav_world, []),
            **wiring_kw)
        wiring = dataclasses.replace(wiring, action_logger=alog)
        return L2CoreAPI(wiring), log_dir

    def test_wiring_default_has_no_logger(self, tmp_path):
        nav_world = NavWorld()
        nav_world.menu_open = True
        nav_world.market_opened = True
        clock = FakeClock()
        hid = ScriptedHid(clock, nav_world)
        wiring = make_wiring(
            nav_world, hid, clock=clock,
            navigation=make_nav_model(),
            nav_evaluate=make_nav_evaluate(nav_world, []),
            hud_locator=make_nav_hud_locator(nav_world, []))
        api = L2CoreAPI(wiring)
        assert api._w.action_logger is None
        result = api.open_market()      # логирование off — работает без него
        assert result.outcome is not None

    def test_open_market_failed_logged_with_frames(self, tmp_path):
        # FAILED (эффекта нет — клик не открывает обзор): ключевая запись
        # с кадрами.
        nav_world = NavWorld()
        nav_world.open_click_works = False
        api, log_dir = self._nav_api(tmp_path, nav_world)

        result = api.open_market()

        assert result.outcome is ActionOutcome.FAILED
        (record,) = read_records(log_dir)
        assert record["action"]["type"] == "open_market"
        assert record["action"]["target"] == "market"
        assert record["verify"]["outcome"] == "FAILED"
        assert record["key"] is True
        assert record["frames"]["before"] is not None
        assert (log_dir / record["frames"]["before"]).is_file()

    def test_open_market_ok_not_key(self, tmp_path):
        nav_world = NavWorld()
        nav_world.menu_open = True
        nav_world.market_opened = True
        api, log_dir = self._nav_api(tmp_path, nav_world)

        result = api.open_market()

        assert result.outcome is ActionOutcome.OK
        (record,) = read_records(log_dir)
        assert record["key"] is False
        assert record["frames"] == {"before": None, "after": None}

    def test_start_sell_irreversible_logged(self, tmp_path):
        # Необратимое действие: ключевая запись даже при OK.
        world = World()
        api, log_dir = self._api(tmp_path, world,
                                 anchors={"shop_sell": SELL_ANCHOR})

        result = api.start_sell(confirm_checks=VerifySpec(checks=()),
                                verify=VerifySpec(checks=()))

        assert result.outcome is ActionOutcome.OK
        (record,) = read_records(log_dir)
        assert record["action"]["type"] == "start_sell"
        assert record["key"] is True
        assert record["frames"]["before"] is not None
        assert record["frames"]["after"] is not None

    def test_cancelled_call_logged_without_frames(self, tmp_path):
        world = World()
        api, log_dir = self._api(tmp_path, world, cancelled=lambda: True)

        result = api.open_market()

        assert result.cancelled
        (record,) = read_records(log_dir)
        assert record["verify"]["cancelled"] is True
        assert record["key"] is False


class TestCreateApiLogger:
    """create_api поднимает ActionLogger из config.log (enabled/dir/лимиты/
    политика); выключенная секция — logger None (логирование off)."""

    @pytest.fixture
    def patched(self, monkeypatch):
        import l2core.api.wiring as wiring_module

        class FakeScreen:
            def grab(self):
                return np.zeros((4, 4, 3), dtype=np.uint8)

            def close(self):
                pass

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
        monkeypatch.setattr(wiring_module, "HidDevice",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                AssertionError("живой HID не нужен")))
        monkeypatch.setattr(wiring_module, "create_ocr_client",
                            lambda *a, **kw: object())
        monkeypatch.setattr(wiring_module, "PyWin32Gateway", FakeGateway)
        repo = Path(__file__).resolve().parent.parent
        return {"kwargs": dict(models_dir=repo / "ui_model",
                               assets_dir=repo / "assets")}

    def test_config_enabled_builds_logger(self, patched, tmp_path):
        config = dataclasses.replace(
            CoreConfig(dry_run=True),
            log=LogConfig(dir=str(tmp_path / "acts")))
        api = create_api(config, **patched["kwargs"])

        assert isinstance(api._w.action_logger, ActionLogger)

    def test_config_disabled_no_logger(self, patched):
        config = dataclasses.replace(
            CoreConfig(dry_run=True), log=LogConfig(enabled=False))
        api = create_api(config, **patched["kwargs"])

        assert api._w.action_logger is None

    def test_explicit_logger_wins(self, patched, tmp_path):
        alog = make_logger(tmp_path / "explicit")
        api = create_api(CoreConfig(dry_run=True),
                         action_logger=alog, **patched["kwargs"])

        assert api._w.action_logger is alog
