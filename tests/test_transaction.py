"""Транзакция кадра: grab → analyze once; refresh/re-verify (задача 3.6).

Спека action-primitives («Кэш кадра в рамках транзакции», design.md
решение 18): один grab анализируется не более одного раза внутри шага;
перед необратимым действием — свежий grab и re-verify цели.

Контракт (l2core/detector/transaction.py):

- FrameTransaction(screen) — явный контекстный менеджер; внутри контекста
  frame возвращает один и тот же кадр (один grab до refresh);
- refresh() — инвалидация кэша и свежий grab (следующий анализ по новому
  кадру);
- reverify(predicate) — свежий grab + проверка predicate по новому кадру
  (точка входа «re-verify перед необратимым», исполнитель — примитивы 5.x);
- вложенный вход — RuntimeError (определённое поведение, не молчаливый
  общий кэш); повторный вход после выхода — свежая транзакция;
- обращение к frame вне контекста — RuntimeError.

Детерминизм: счётчик grab'ов — обёртка-шпион поверх FakeScreen.
"""

import numpy as np
import pytest

from l2core.detector.transaction import FrameTransaction

FRAME_A = np.full((1080, 1920, 3), 40, dtype=np.uint8)
FRAME_B = np.full((1080, 1920, 3), 80, dtype=np.uint8)


class CountingScreen:
    """Обёртка-шпион: каждый grab — следующий скриптованный кадр."""

    def __init__(self, frames):
        if not frames:
            raise ValueError("нужен хотя бы один кадр")
        self._frames = list(frames)
        self.grabs = 0

    def grab(self):
        self.grabs += 1
        if self.grabs > len(self._frames):
            return self._frames[-1]
        return self._frames[self.grabs - 1]


def test_two_analyses_one_grab_inside_transaction():
    """Два анализа внутри транзакции — один grab (analyze once)."""
    screen = CountingScreen([FRAME_A, FRAME_B])
    with FrameTransaction(screen) as tx:
        first = tx.frame
        second = tx.frame
    assert screen.grabs == 1
    assert first is FRAME_A
    assert second is FRAME_A


def test_refresh_invalidates_and_grabs_fresh():
    """refresh() — новый grab; последующие анализы по новому кадру."""
    screen = CountingScreen([FRAME_A, FRAME_B, FRAME_B])
    with FrameTransaction(screen) as tx:
        assert tx.frame is FRAME_A
        fresh = tx.refresh()
        assert screen.grabs == 2
        assert fresh is FRAME_B
        assert tx.frame is FRAME_B


def test_reverify_grabs_fresh_and_checks_predicate():
    """re-verify перед необратимым: свежий grab + проверка по новому кадру."""
    screen = CountingScreen([FRAME_A, FRAME_B])
    seen = []

    def predicate(frame):
        seen.append(frame)
        return frame is FRAME_B

    with FrameTransaction(screen) as tx:
        tx.frame  # кэшированный анализ
        assert tx.reverify(predicate) is True
    assert screen.grabs == 2
    assert seen == [FRAME_B]


def test_reverify_false_when_predicate_rejects():
    screen = CountingScreen([FRAME_A, FRAME_B])
    with FrameTransaction(screen) as tx:
        tx.frame  # кэшированный анализ
        assert tx.reverify(lambda frame: False) is False
    assert screen.grabs == 2


def test_nested_enter_is_rejected():
    """Вложенная транзакция — определённая ошибка, не общий кэш."""
    screen = CountingScreen([FRAME_A])
    with FrameTransaction(screen) as outer:
        with pytest.raises(RuntimeError):
            with FrameTransaction(screen):
                pass  # pragma: no cover — не достигается
        assert outer.frame is FRAME_A


def test_reenter_after_exit_is_fresh_transaction():
    """Повторный вход после выхода — свежая транзакция (кэш не протекает)."""
    screen = CountingScreen([FRAME_A, FRAME_B])
    with FrameTransaction(screen) as tx:
        assert tx.frame is FRAME_A
    with FrameTransaction(screen) as tx:
        assert tx.frame is FRAME_B
    assert screen.grabs == 2


def test_frame_outside_context_is_error():
    screen = CountingScreen([FRAME_A])
    tx = FrameTransaction(screen)
    with pytest.raises(RuntimeError):
        _ = tx.frame


def test_refresh_outside_context_is_error():
    screen = CountingScreen([FRAME_A])
    tx = FrameTransaction(screen)
    with pytest.raises(RuntimeError):
        tx.refresh()


def test_reverify_replaces_cached_frame_for_further_analyses():
    """После reverify дальнейшие анализы идут уже по свежему кадру."""
    screen = CountingScreen([FRAME_A, FRAME_B])
    with FrameTransaction(screen) as tx:
        assert tx.frame is FRAME_A
        tx.reverify(lambda frame: True)
        assert tx.frame is FRAME_B
    assert screen.grabs == 2
