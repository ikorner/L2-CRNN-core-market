"""Живой dry-run прогон на живой игре (приёмка задачи 9.4, spec safety).

Ручной инструмент (вне CI), запуск из корня репо при запущенной и видимой
игре:

    python tools/live_dry_run.py --model E:\\L2-CRNN-MODEL-TRAIN\\runs\\<run>\\crnn_best.pt

Экран читается по-настоящему (MssScreen), детекция состояний работает на
живых кадрах, а все HID-действия идут в DryRunHid как намерения (лог
l2core.dry_run + JSONL ActionLogger с ключевыми кадрами, 8.3). Навигация
честно падает в FAILED/UNKNOWN — эффекта без HID нет; OK возможен, только
если окно уже открыто. Остановка — хоткей kill switch (ctrl+shift+k).

Чекпоинт OCR: флаг --model побеждает paths.model конфига; без файла —
понятная ошибка, код выхода 2 (конвенция benchmark_ocr.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

__all__ = ["StepReport", "ProbeReport", "run_probe"]


@dataclass(frozen=True)
class StepReport:
    """Итог одного шага пробы: деталь (одна строка) или ошибка."""

    name: str
    detail: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Шаг выполнился без исключения (FAILED-итог — тоже честный исход)."""
        return self.error is None


@dataclass(frozen=True)
class ProbeReport:
    """Сводка пробы: шаги в порядке выполнения + факт прерывания kill'ом."""

    steps: Tuple[StepReport, ...]
    killed: bool


# --- Шаги пробы: каждый принимает фасад и возвращает строку-деталь ---

def _step_health(api) -> str:
    h = api.health()
    return (f"ok={h.ok} окно_игры={h.game_window} ({h.game_detail}) "
            f"окно_торговли={h.trading_window} ocr={h.ocr} hid={h.hid}")


def _state_detail(state) -> str:
    return (f"window={state.window} status={state.status} "
            f"state={state.state} params={dict(state.params)}")


def _step_state(api) -> str:
    return _state_detail(api.get_state())


def _step_open_market(api) -> str:
    r = api.open_market()
    return (f"success={r.success} outcome={r.outcome} "
            f"cancelled={r.cancelled} needs_recover={r.needs_recover}")


def _step_read_rows(api) -> str:
    r = api.read_rows()
    return f"строк={len(r.rows)} snapshot_id={r.snapshot_id}"


def _make_step_scroll(n: int):
    def _step_scroll(api) -> str:
        r = api.scroll_list("down", n)
        return (f"requested={r.requested} scrolled={r.scrolled} "
                f"reached_end={r.reached_end} wheels={r.wheels}")
    return _step_scroll


def _step_close_market(api) -> str:
    r = api.close_market()
    return (f"success={r.success} outcome={r.outcome} "
            f"cancelled={r.cancelled} needs_recover={r.needs_recover}")


def run_probe(api, *, out: Callable[[str], None] = print,
              cancelled: Optional[Callable[[], bool]] = None,
              scroll_n: int = 2) -> ProbeReport:
    """Прога-зонд по фасаду: health → get_state → open_market → get_state →
    (read_rows → scroll → read_rows, если обзор found) → close_market.

    Исключение шага фиксируется в StepReport.error, проба продолжается;
    cancelled (kill switch 8.1) проверяется между шагами — прерывает пробу.
    Каждый шаг и итоговая сводка печатаются через out.
    """
    reports = []
    killed = False

    def cancelled_now() -> bool:
        return cancelled is not None and cancelled()

    for name, call in (("health", _step_health),
                       ("get_state", _step_state),
                       ("open_market", _step_open_market)):
        if cancelled_now():
            killed = True
            break
        reports.append(_run_step(name, call, api, out))
    if not killed and not cancelled_now():
        # Второе чтение состояния — самим шагом, объект нужен для ветвления.
        try:
            state = api.get_state()
            detail = _state_detail(state)
            reports.append(StepReport(name="get_state", detail=detail))
            out(f"  [get_state] {detail}")
            opened = state.window == "market" and state.status == "found"
        except Exception as exc:
            reports.append(StepReport(
                name="get_state", detail="",
                error=f"{type(exc).__name__}: {exc}"))
            out(f"  [get_state] ОШИБКА: {reports[-1].error}")
            opened = state.window == "market" and state.status == "found"
        following = (("read_rows", _step_read_rows),
                     ("scroll_list", _make_step_scroll(scroll_n)),
                     ("read_rows", _step_read_rows)) if opened else ()
        for name, call in (*following, ("close_market", _step_close_market)):
            if cancelled_now():
                killed = True
                break
            reports.append(_run_step(name, call, api, out))
    elif not killed:
        killed = True
    _print_summary(reports, killed=killed, out=out)
    return ProbeReport(steps=tuple(reports), killed=killed)


def _run_step(name: str, call: Callable, api,
              out: Callable[[str], None]) -> StepReport:
    try:
        detail = call(api)
    except Exception as exc:  # ошибка шага не роняет пробу
        report = StepReport(name=name, detail="", error=f"{type(exc).__name__}: {exc}")
        out(f"  [{name}] ОШИБКА: {report.error}")
        return report
    report = StepReport(name=name, detail=detail)
    out(f"  [{name}] {detail}")
    return report


def _print_summary(reports, *, killed: bool,
                   out: Callable[[str], None]) -> None:
    ok = sum(1 for s in reports if s.ok)
    out(f"Сводка: шагов={len(reports)} без_ошибок={ok} killed={killed}")
