"""Юнит-тесты абстракции Screen: MssScreen, FakeScreen, ReplayScreen, RecordingScreen."""

import numpy as np
import pytest

from l2core.adapters.screen import (
    ActionMismatchError,
    FakeScreen,
    MssScreen,
    MssError,
    RecordingScreen,
    ReplayScreen,
    Screen,
    ScreenClosedError,
    SessionExhaustedError,
    _bgra_to_bgr,
)
from l2core.adapters.session import SessionFormatError, SessionWriter, load_session


def frame(fill: int, h: int = 4, w: int = 6) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = fill
    return arr


class TestMssScreen:
    def test_grab_returns_numpy_frame_from_injected_grabber(self):
        expected = frame(42, h=8, w=10)
        screen = MssScreen(grabber=lambda: expected)
        got = screen.grab()
        assert isinstance(got, np.ndarray)
        assert got.dtype == np.uint8
        assert got.shape == (8, 10, 3)
        assert np.array_equal(got, expected)

    def test_creation_does_not_touch_mss(self, monkeypatch):
        """Создание без grabber не требует mss/монитора — захват ленивый."""
        monkeypatch.setitem(__import__("sys").modules, "mss", None)
        screen = MssScreen()  # не падает: mss не импортирован
        with pytest.raises(MssError):
            screen.grab()

    def test_bgra_to_bgr_conversion(self):
        bgra = np.zeros((2, 2, 4), dtype=np.uint8)
        bgra[..., 0] = 1  # B
        bgra[..., 1] = 2  # G
        bgra[..., 2] = 3  # R
        bgra[..., 3] = 255  # A
        bgr = _bgra_to_bgr(bgra)
        assert bgr.shape == (2, 2, 3)
        assert np.array_equal(bgr[0, 0], [1, 2, 3])
        assert bgr.flags["C_CONTIGUOUS"]


class TestFakeScreen:
    def test_grab_returns_current_frame_and_actions_advance(self):
        screen = FakeScreen([frame(1), frame(2), frame(3)])
        assert np.array_equal(screen.grab(), frame(1))
        screen.notify_action({"kind": "click"})
        assert np.array_equal(screen.grab(), frame(2))
        screen.notify_action({"kind": "click"})
        assert np.array_equal(screen.grab(), frame(3))

    def test_exhaustion_raises(self):
        screen = FakeScreen([frame(1), frame(2)])
        screen.notify_action({"kind": "click"})
        with pytest.raises(SessionExhaustedError):
            screen.notify_action({"kind": "click"})

    def test_grab_is_stable_between_actions(self):
        screen = FakeScreen([frame(1), frame(2)])
        assert np.array_equal(screen.grab(), screen.grab())


class TestReplayScreen:
    def _build_session(self, tmp_path, events_builder, meta=None):
        path = tmp_path / "session"
        with SessionWriter(path, meta=meta) as writer:
            events_builder(writer)
        return path

    def test_grab_returns_initial_frame(self, tmp_path):
        path = self._build_session(
            tmp_path, lambda w: w.record_frame(frame(11))
        )
        screen = ReplayScreen(path)
        assert np.array_equal(screen.grab(), frame(11))

    def test_action_advances_to_post_action_frame(self, tmp_path):
        def build(w):
            w.record_frame(frame(1))
            w.record_action(
                {"kind": "click", "target": "market_pic"}, frame_after=frame(2)
            )
            w.record_action({"kind": "click", "target": "tab"}, frame_after=frame(3))

        screen = ReplayScreen(self._build_session(tmp_path, build))
        assert np.array_equal(screen.grab(), frame(1))
        screen.notify_action({"kind": "click", "target": "market_pic"})
        assert np.array_equal(screen.grab(), frame(2))
        screen.notify_action({"kind": "click", "target": "tab"})
        assert np.array_equal(screen.grab(), frame(3))

    def test_repeated_grabs_do_not_advance(self, tmp_path):
        path = self._build_session(
            tmp_path,
            lambda w: (
                w.record_frame(frame(1)),
                w.record_action({"kind": "click"}, frame_after=frame(2)),
            ),
        )
        screen = ReplayScreen(path)
        assert np.array_equal(screen.grab(), frame(1))
        assert np.array_equal(screen.grab(), frame(1))

    def test_mismatched_action_raises(self, tmp_path):
        path = self._build_session(
            tmp_path,
            lambda w: (
                w.record_frame(frame(1)),
                w.record_action({"kind": "click", "target": "a"}, frame_after=frame(2)),
            ),
        )
        screen = ReplayScreen(path)
        with pytest.raises(ActionMismatchError):
            screen.notify_action({"kind": "click", "target": "b"})

    def test_exhaustion_raises(self, tmp_path):
        path = self._build_session(
            tmp_path,
            lambda w: (
                w.record_frame(frame(1)),
                w.record_action({"kind": "click"}, frame_after=frame(2)),
            ),
        )
        screen = ReplayScreen(path)
        screen.notify_action({"kind": "click"})
        with pytest.raises(SessionExhaustedError):
            screen.notify_action({"kind": "click"})
        # после исчерпания grab возвращает последний кадр (не падает)
        assert np.array_equal(screen.grab(), frame(2))

    def test_tuple_action_matches_recorded_list(self, tmp_path):
        """Действие с кортежами нормализовано — replay сопоставляет структурно."""
        path = self._build_session(
            tmp_path,
            lambda w: (
                w.record_frame(frame(1)),
                w.record_action({"kind": "click", "pos": [5, 6]}, frame_after=frame(2)),
            ),
        )
        screen = ReplayScreen(path)
        screen.notify_action({"kind": "click", "pos": (5, 6)})  # не поднимает
        assert np.array_equal(screen.grab(), frame(2))

    def test_session_built_programmatically_replays(self, tmp_path):
        """R7: сессия построена программно (без живого экрана) и проигрывается."""
        path = tmp_path / "session"
        with SessionWriter(path, meta={"source": "synthetic"}) as writer:
            writer.record_frame(frame(100))
            writer.record_action({"kind": "click"}, frame_after=frame(200))
            writer.set_result("ok")
        screen = ReplayScreen(path)
        screen.notify_action({"kind": "click"})
        assert np.array_equal(screen.grab(), frame(200))

    def test_invalid_session_raises(self, tmp_path):
        path = tmp_path / "session"
        path.mkdir()
        with pytest.raises(SessionFormatError):
            ReplayScreen(path)


class TestRecordingScreen:
    def test_records_frames_actions_results(self, tmp_path):
        inner = FakeScreen([frame(1), frame(2)])
        path = tmp_path / "session"
        screen = RecordingScreen(inner, path)
        assert np.array_equal(screen.grab(), frame(1))
        screen.notify_action({"kind": "click", "target": "market_pic", "pos": (1, 2)})
        assert np.array_equal(screen.grab(), frame(2))
        screen.record_result("ok", {"verify": "list__tab_home"})
        screen.close()

        session = load_session(path)
        assert [e["type"] for e in session.events] == ["frame", "action"]
        action = session.events[1]
        assert action["action"]["pos"] == [1, 2]
        assert action["result"] == {"status": "ok", "detail": {"verify": "list__tab_home"}}
        assert np.array_equal(session.read_image(action["image"]), frame(2))

    def test_duplicate_grabs_not_recorded_twice(self, tmp_path):
        inner = FakeScreen([frame(1), frame(2)])
        path = tmp_path / "session"
        with RecordingScreen(inner, path) as screen:
            screen.grab()
            screen.grab()
            screen.grab()
            screen.notify_action({"kind": "click"})
            screen.grab()
            screen.grab()

        session = load_session(path)
        assert [e["type"] for e in session.events] == ["frame", "action"]
        assert len(list((path / "frames").iterdir())) == 2

    def test_frame_change_without_action_recorded(self, tmp_path):
        inner = FakeScreen([frame(1), frame(2), frame(3)])
        path = tmp_path / "session"
        with RecordingScreen(inner, path) as screen:
            screen.grab()                       # frame(1)
            screen.notify_action({"kind": "a"})
            screen.grab()                       # frame(2) — кадр после действия
            screen.notify_action({"kind": "b"})
            screen.grab()                       # frame(3)

        session = load_session(path)
        assert [e["type"] for e in session.events] == ["frame", "action", "action"]
        # все три кадра различны — три PNG
        assert len(list((path / "frames").iterdir())) == 3

    def test_pending_action_without_grab_flushed_on_close(self, tmp_path):
        inner = FakeScreen([frame(1), frame(1)])
        path = tmp_path / "session"
        with RecordingScreen(inner, path) as screen:
            screen.grab()
            screen.notify_action({"kind": "click"})

        session = load_session(path)
        assert session.events[1]["image"] is None

    def test_close_is_idempotent(self, tmp_path):
        inner = FakeScreen([frame(1)])
        screen = RecordingScreen(inner, tmp_path / "session")
        screen.grab()
        screen.close()
        first = (tmp_path / "session" / "session.json").read_bytes()
        screen.close()  # повторный close не портит файл
        second = (tmp_path / "session" / "session.json").read_bytes()
        assert first == second

    def test_grab_after_close_rejected(self, tmp_path):
        screen = RecordingScreen(FakeScreen([frame(1)]), tmp_path / "session")
        screen.close()
        with pytest.raises(ScreenClosedError, match="закрыт"):
            screen.grab()

    def test_replay_of_recorded_session(self, tmp_path):
        """Полный цикл: живой прогон (на FakeScreen) → запись → replay."""
        frames = [frame(i) for i in (1, 2, 3)]
        actions = [
            {"kind": "click", "target": "market_pic"},
            {"kind": "click", "target": "tab_home"},
        ]
        path = tmp_path / "session"
        with RecordingScreen(FakeScreen(frames), path) as recording:
            for action in actions:
                recording.grab()
                recording.notify_action(action)
                recording.grab()
                recording.record_result("ok")

        replay = ReplayScreen(path)
        for i, action in enumerate(actions):
            assert np.array_equal(replay.grab(), frames[i])
            replay.notify_action(action)
            assert np.array_equal(replay.grab(), frames[i + 1])


class TestScreenProtocol:
    def test_all_implementations_are_screens(self):
        assert isinstance(MssScreen(grabber=lambda: frame(1)), Screen)
        assert isinstance(FakeScreen([frame(1)]), Screen)
