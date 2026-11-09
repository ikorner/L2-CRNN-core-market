"""Kill switch: локальный хоткей + глобальный таймаут неактивности
(задача 8.1, spec safety: «Аварийная остановка хоткеем», «Таймаут
неактивности», решение 16 design.md).

ПРИНЦИП РАБОТЫ. KillSwitch — слушатель в собственном потоке: опрашивает
источник хоткея и часы; при срабатывании (хоткей ИЛИ таймаут тишины) —
kill(reason), ИДЕМПОТЕНТНЫЙ и необратимый в пределах жизни объекта:

1. Флаг killed выставляется; `is_killed()` — cancelled-callable: его
   передают в сборку wiring (wiring.cancelled — дефолт всех блокирующих
   вызовов API 7.1), каждый wait_until/примитив проверяет его на poll'ах —
   блокирующий вызов прерывается. СЕМАНТИКА ПРЕРЫВАНИЯ — CANCELLED-домен:
   результат примитива/фасада с cancelled=True (outcome=None, эффект до
   прерывания фиксируется флагом unconfirmed, если ушёл), НЕ UNKNOWN:
   UNKNOWN — эффект ушёл и подтвердить нельзя (инвариант 4/5.7), kill —
   внешняя остановка, это другой домен.
2. `hid.release_all()` — best-effort: залипшая кнопка мыши/клавиши из
   прерванного drag/ввода отпускается. ВЫБОР АРХИТЕКТУРЫ (guard vs kill):
   слушатель живёт В СВОЕМ потоке, а владелец HID — поток стратегии,
   поэтому release_all при kill — САНКЦИОНИРОВАННОЕ ИСКЛЮЧЕНИЕ guard'а
   владельца: OwnerGuardedHid(hid, kill_switch=ks.is_killed) пропускает
   ТОЛЬКО release_all при поднятом kill-флаге, любая другая HID-команда
   из чужого потока по-прежнему отклоняется (fail closed, hid_owner.py).
   Обоснование: release_all идемпотентен и только отпускает кнопки —
   безопасен из любого потока; альтернатива (kill-слушатель в executor-
   потоке) невозможна — поток стратегии блокирован в вызове API, из
   которого его же и надо прерывать. HID-сбой при release_all глушится
   (фиксация в лог) — kill не должен умереть от транспорта.
3. Курсор уводится в безопасную точку: `set_cursor_pos(*safe_point)`
   (Win32-шлюз, протокол дополнен set_cursor_pos) — увод от кнопок игры,
   чтобы курсор не остался над «Начать продажу» и т.п. БЕЗОПАСНАЯ ТОЧКА:
   параметр safe_point побеждает config.kill_switch.safe_point; оба не
   заданы — курсор не трогается + диагностика в лог (честная деградация,
   как revenue=None у монитора 6.3; эталонный config.yaml задаёт точку).
4. Событие стратегии `killed`: push-канал 7.3 через on_kill-callback
   (движок подключает: L2CoreBot(kill_switch=ks) -> queue.push(
   EVENT_KILLED, {"reason": reason})) — payload по справочнику docs/API.md.
   Сбой callback глушится с фиксацией в лог.

ПОЛИТИКА ОСТАНОВКИ ДВИЖКА. Kill НЕ останавливает L2CoreBot сам: observer
получает is_killed (явным cancelled или по умолчанию от kill_switch) и
завершает свой цикл сам; dispatcher должен остаться живым — он доставляет
само событие killed подписчикам; join потоков — за стратегией (получила
killed -> bot.stop(), идиемпотентен, безопасен и из callback'а dispatcher'а).
«Мгновенный стоп цикла» из спеки = мгновенное прерывание блокирующих
вызовов + выход наблюдателя, а не принудительный kill потоков.

ТАЙМАУТ НЕАКТИВНОСТИ. Линия — время последнего вызова note_activity();
зовёт его executor/API: каждый публичный метод L2CoreAPI проверяет поток-
владелец и через wiring.note_activity фиксирует активность (поток
стратегии жив и работает). Превышение config.kill_switch.
inactivity_timeout секунд тишины -> kill("inactivity") — та же остановка,
что по хоткею (spec, сценарий «Таймаут неактивности»).

КАНАЛ ХОТКЕЯ. Инъекция hotkey_source (контракт: start/stop/pressed);
прод-реализация — Win32HotkeySource (adapters/win32.py): низкоуровневый
хук WH_KEYBOARD_LL в собственном потоке — нажатие фиксируется самим
хуком (не опросом), короткое нажатие не пропадает между poll'ами. Слушает
локальную клавиатуру, НЕ arduino (spec safety).

ОЖИДАНИЕ В ЦИКЛЕ — wait_until (инвариант 1: прямых sleep в ядре нет;
guard tests/test_sleep_contract.py), часы/сон — инъекции. poll_once() —
одна детерминированная итерация проверок (хоткей? таймаут?), публичная
точка для тестов без потоков.

ЭТАЛОННАЯ СБОРКА (живой запуск):

    ks = KillSwitch(config.kill_switch,
                    hotkey_source=Win32HotkeySource(config.kill_switch.hotkey),
                    set_cursor_pos=lambda x, y: gateway.set_cursor_pos(x, y))
    api = create_api(config, cancelled=ks.is_killed)      # wiring.cancelled
    bot = L2CoreBot(wiring_or_create, kill_switch=ks)     # hid + killed
    ks.start(); bot.start()
    ...                                                   # стратегия работает
    # по killed: bot.stop()

YAGNI (задача 8.1): dry-run (8.2) и JSONL-лог (8.3) сюда не входят.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional, Tuple

from l2core.config import KillSwitchConfig
from l2core.primitives.wait import wait_until

__all__ = [
    "REASON_HOTKEY",
    "REASON_INACTIVITY",
    "KillSwitch",
    "KillSwitchError",
]

logger = logging.getLogger(__name__)

# Причины kill — контракт payload события killed (docs/API.md).
REASON_HOTKEY = "hotkey"
REASON_INACTIVITY = "inactivity"

# Дефолт интервала опроса слушателя (сек): хоткей фиксирует сам хук в
# момент нажатия (флаг), слушатель замечает его не позже этого интервала;
# таймаут неактивности отслеживается с этой же гранулярностью.
DEFAULT_POLL_INTERVAL = 0.1


class KillSwitchError(Exception):
    """Нарушение жизненного цикла слушателя (повторный start и пр.)."""

    def __init__(self, detail: str):
        super().__init__(f"KillSwitch: {detail}")
        self.detail = detail


class KillSwitch:
    """Слушатель аварийной остановки: хоткей + таймаут неактивности.

    Конструктор принимает провайдеров инъекциями (тесты на фейках,
    прод — адаптеры). hid/on_kill можно подключить и позже через attach()
    (движок 7.2 так подключает guard-сборку, появляющуюся после него) —
    до первого kill; последний attach побеждает.
    """

    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        *,
        hotkey_source: Optional[Any] = None,
        inactivity_timeout: Optional[float] = None,
        safe_point: Optional[Tuple[int, int]] = None,
        set_cursor_pos: Optional[Callable[[int, int], Any]] = None,
        on_kill: Optional[Callable[[str], None]] = None,
        hid: Optional[Any] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if config is not None and not isinstance(config, KillSwitchConfig):
            raise TypeError(f"config: ожидается KillSwitchConfig "
                            f"(получено {type(config).__name__})")
        if inactivity_timeout is None and config is not None:
            inactivity_timeout = config.inactivity_timeout
        if inactivity_timeout is not None and inactivity_timeout <= 0:
            raise ValueError(f"inactivity_timeout должен быть > 0 "
                             f"(получено {inactivity_timeout!r})")
        if safe_point is None and config is not None:
            safe_point = config.safe_point
        if safe_point is not None:
            if (len(safe_point) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                           for v in safe_point)):
                raise ValueError(f"safe_point: ожидается пара неотрицательных "
                                 f"int (получено {safe_point!r})")
            safe_point = (int(safe_point[0]), int(safe_point[1]))
        if poll_interval <= 0:
            raise ValueError(f"poll_interval должен быть > 0 "
                             f"(получено {poll_interval!r})")
        self._hotkey_source = hotkey_source
        self._inactivity_timeout = inactivity_timeout
        self._safe_point = safe_point
        self._set_cursor_pos = set_cursor_pos
        self._on_kill = on_kill
        self._hid = hid
        self._poll_interval = poll_interval
        self._clock = clock
        self._sleep = sleep

        self._lock = threading.Lock()
        self._killed = False
        self._reason: Optional[str] = None
        self._last_activity = clock()
        self._stop = threading.Event()
        self._started = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Контракт cancelled-callable (wiring.cancelled, точки прерывания 5.x)

    def is_killed(self) -> bool:
        """Флаг аварийной остановки (точка инъекции отмены wait_until)."""
        with self._lock:
            return self._killed

    def cancelled(self) -> bool:
        """Алиас is_killed() — единый контракт cancelled-callable."""
        return self.is_killed()

    @property
    def reason(self) -> Optional[str]:
        """Причина сработавшей остановки (None — не срабатывал)."""
        with self._lock:
            return self._reason

    @property
    def safe_point(self) -> Optional[Tuple[int, int]]:
        """Безопасная точка курсора (None — курсор при kill не уводится)."""
        return self._safe_point

    # ------------------------------------------------------------------
    # Активность executor'а (линия таймаута неактивности)

    def note_activity(self) -> None:
        """Зафиксировать активность: линия таймаута неактивности сдвигается
        на «сейчас» (вызывается из потока стратегии — API через
        wiring.note_activity)."""
        with self._lock:
            if not self._killed:
                self._last_activity = self._clock()

    # ------------------------------------------------------------------
    # Подключение провайдеров (движок 7.2 подключает после конструктора)

    def attach(self, *, hid: Optional[Any] = None,
               on_kill: Optional[Callable[[str], None]] = None) -> None:
        """Подключить hid/on_kill (до первого kill; последний attach
        побеждает). Движок так получает guard-сборку HID, созданную уже
        после KillSwitch."""
        with self._lock:
            if hid is not None:
                self._hid = hid
            if on_kill is not None:
                self._on_kill = on_kill

    # ------------------------------------------------------------------
    # Срабатывание

    def poll_once(self) -> bool:
        """Одна итерация проверок (детерминированная точка тестов):
        хоткей нажат? таймаут неактивности превышен? -> kill, True."""
        if self.is_killed():
            return True
        source = self._hotkey_source
        if source is not None and source.pressed():
            self.kill(REASON_HOTKEY)
            return True
        timeout = self._inactivity_timeout
        if timeout is not None:
            with self._lock:
                idle = self._clock() - self._last_activity
            if idle >= timeout:
                self.kill(REASON_INACTIVITY)
                return True
        return False

    def kill(self, reason: str) -> bool:
        """Аварийная остановка: флаг + release_all + курсор + on_kill.
        ИДЕМПОТЕНТНА: повторный вызов — False, эффекты выполняются ровно
        один раз (первый reason — контрактный)."""
        with self._lock:
            if self._killed:
                return False
            self._killed = True
            self._reason = reason
        logger.warning("kill switch: аварийная остановка (%s)", reason)
        # Эффекты — вне lock: HID/курсор/callback чужие, сбои глушим.
        if self._hid is not None:
            try:
                self._hid.release_all()
            except Exception:
                logger.exception(
                    "kill switch: release_all не удалось (HID-сбой): "
                    "состояние кнопок неизвестно — проверьте arduino")
        else:
            logger.warning("kill switch: hid не подключён (attach до kill) — "
                           "release_all пропущен")
        if self._set_cursor_pos is not None and self._safe_point is not None:
            try:
                self._set_cursor_pos(*self._safe_point)
            except Exception:
                logger.exception(
                    "kill switch: увод курсора в безопасную точку %s не "
                    "удался", self._safe_point)
        elif self._safe_point is None:
            logger.warning("kill switch: safe_point не задан — курсор не "
                           "уводится (задайте kill_switch.safe_point в "
                           "конфиге)")
        if self._on_kill is not None:
            try:
                self._on_kill(reason)
            except Exception:
                logger.exception(
                    "kill switch: on_kill-callback бросил исключение")
        return True

    # ------------------------------------------------------------------
    # Жизненный цикл слушателя

    def start(self) -> None:
        """Запустить поток слушателя (повторный start — KillSwitchError;
        restart после stop не поддерживается — жизненный цикл один)."""
        if self._started:
            raise KillSwitchError(
                "повторный запуск: слушатель уже запущен" if self._thread
                else "повторный запуск после stop не поддерживается "
                     "(жизненный цикл один)")
        self._started = True
        if self._hotkey_source is not None:
            self._hotkey_source.start()
        self._thread = threading.Thread(
            target=self.run, name="l2core-killswitch")
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> bool:
        """Остановить слушатель: stop-флаг + join (idempotent; без start —
        no-op). Возвращает факт join'а; kill-флаг НЕ снимается — kill
        необратим, повторный start после stop не поддерживается."""
        self._stop.set()
        joined = True
        thread = self._thread
        if thread is not None:
            self._thread = None
            if thread is not threading.current_thread():
                thread.join(timeout=timeout)
                joined = not thread.is_alive()
        if self._hotkey_source is not None:
            self._hotkey_source.stop()
        return joined

    def run(self) -> None:
        """Тело потока слушателя: poll_once + wait_until до wake/таймаута
        (инвариант 1: прямых sleep нет; часы/сон — инъекции)."""
        source = self._hotkey_source

        def _wake() -> bool:
            if self._stop.is_set():
                return True
            return source is not None and source.pressed()

        while not self._stop.is_set():
            if self.poll_once():
                break
            wait_until(
                _wake,
                timeout=self._poll_interval,
                interval=min(self._poll_interval, 0.05),
                clock=self._clock, sleep=self._sleep)
