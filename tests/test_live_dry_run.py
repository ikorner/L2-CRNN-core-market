"""Живой dry-run прогон (9.4): проба run_probe на stub-API.

Раннер tools/live_dry_run.py — ручной инструмент приёмки 9.4 (вне CI):
run_probe — чистая композиция вызовов фасада L2CoreAPI, тестируется на
stub-API без экрана/HID/Win32 (детерминизм, инвариант 7). Живую сборку
main() (Win32, mss, torch) тесты не поднимают — она покрывается самим
прогоном на живой игре.

Покрытие:
- последовательность шагов и сводка при открытом обзоре;
- пропуск табличных шагов, когда обзор не открылся (типичный исход dry-run);
- фиксация исключения шага с продолжением пробы;
- прерывание по cancelled (kill switch) между шагами.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from l2core.api.core import CoreState, HealthResult, NavigationResult
from l2core.model.rows import RowsResult
from l2core.primitives.outcome import ActionOutcome
from l2core.primitives.scroll import ScrollResult

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "live_dry_run.py"


@pytest.fixture()
def probe():
    """Модуль раннера, загруженный из tools/ (прецедент: test_ocr_golden)."""
    if not SCRIPT.exists():
        pytest.fail(f"не найден {SCRIPT}")
    spec = importlib.util.spec_from_file_location("live_dry_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass ищет модуль по __module__
    spec.loader.exec_module(module)
    return module


class StubApi:
    """Минимальный двойник фасада: фиксирует вызовы, отдаёт к canned-итоги."""

    def __init__(self, *, state_after_open=None, fail_on=None,
                 fail_state_on_call=None):
        self.calls = []
        self._state_after_open = state_after_open
        self._fail_on = fail_on
        self._fail_state_on_call = fail_state_on_call
        self._state_calls = 0

    def health(self, **kwargs):
        self.calls.append("health")
        return HealthResult(ok=True, game_window=True, game_detail="ok",
                            trading_window=None, ocr=True, hid=True)

    def get_state(self, **kwargs):
        self.calls.append("get_state")
        self._state_calls += 1
        if self._fail_state_on_call == self._state_calls:
            raise RuntimeError("экран недоступен")
        if self._state_after_open is not None and "open_market" in self.calls:
            return self._state_after_open
        return CoreState(window=None, status="not_found", state=None)

    def open_market(self, **kwargs):
        self.calls.append("open_market")
        if self._fail_on == "open_market":
            raise RuntimeError("нет окна игры")
        return NavigationResult(success=False, outcome=ActionOutcome.FAILED)

    def read_rows(self, **kwargs):
        self.calls.append("read_rows")
        return RowsResult(rows=(), snapshot_id="s1")

    def scroll_list(self, direction="down", n=1, **kwargs):
        self.calls.append("scroll_list")
        return ScrollResult(requested=n, scrolled=0, reached_end=False,
                            position=None, wheels=n)

    def close_market(self, **kwargs):
        self.calls.append("close_market")
        return NavigationResult(success=True, outcome=ActionOutcome.OK)


def test_probe_sequence_and_summary(probe):
    api = StubApi(state_after_open=CoreState(
        window="market", status="found", state="list__tab_home"))
    lines = []
    report = probe.run_probe(api, out=lines.append)
    assert api.calls == ["health", "get_state", "open_market", "get_state",
                         "read_rows", "scroll_list", "read_rows",
                         "close_market"]
    assert report.killed is False
    assert [s.name for s in report.steps] == api.calls
    assert all(s.ok and s.detail for s in report.steps)
    assert lines  # сводка напечатана через out


def test_probe_skips_table_when_not_found(probe):
    api = StubApi()  # обзор не открылся — типичный исход dry-run
    report = probe.run_probe(api, out=lambda line: None)
    assert api.calls == ["health", "get_state", "open_market", "get_state",
                         "close_market"]
    assert "read_rows" not in api.calls
    assert "scroll_list" not in api.calls
    assert report.killed is False


def test_probe_continues_after_step_error(probe):
    api = StubApi(fail_on="open_market")
    lines = []
    report = probe.run_probe(api, out=lines.append)
    by_name = {s.name: s for s in report.steps}
    assert by_name["open_market"].error is not None
    assert "RuntimeError" in by_name["open_market"].error
    assert by_name["close_market"].ok
    assert "close_market" in api.calls  # проба продолжилась после ошибки
    assert lines


def test_probe_survives_second_state_failure(probe):
    # Второй get_state (после open_market) бросает: ошибка фиксируется
    # шагом, проба продолжается (close_market), раннер не падает.
    api = StubApi(state_after_open=CoreState(
        window="market", status="found", state="list__tab_home"),
        fail_state_on_call=2)
    lines = []
    report = probe.run_probe(api, out=lines.append)
    states = [s for s in report.steps if s.name == "get_state"]
    assert states[-1].error is not None
    assert "RuntimeError" in states[-1].error
    assert "close_market" in api.calls
    assert lines


def test_probe_cancelled_between_steps(probe):
    api = StubApi()
    checks = {"n": 0}

    def cancelled():
        checks["n"] += 1
        return checks["n"] > 1  # kill перед вторым шагом

    lines = []
    report = probe.run_probe(api, out=lines.append, cancelled=cancelled)
    assert report.killed is True
    assert api.calls == ["health"]
    assert lines  # сводка выполненных шагов напечатана


def test_resolve_model_override_wins(tmp_path, probe):
    existing = tmp_path / "a.pt"
    existing.write_bytes(b"fake")
    missing = str(tmp_path / "missing.pt")
    assert probe._resolve_model(missing, str(existing)) == existing


def test_resolve_model_missing_returns_none(tmp_path, probe):
    missing = str(tmp_path / "missing.pt")
    assert probe._resolve_model(missing, None) is None
    assert probe._resolve_model(missing, str(tmp_path / "also-missing.pt")) is None
