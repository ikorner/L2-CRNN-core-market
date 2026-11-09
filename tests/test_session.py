"""Юнит-тесты формата файла сессии (запись/чтение, детерминизм, ошибки формата)."""

import json

import numpy as np
import pytest

from l2core.adapters.session import (
    SessionFormatError,
    SessionWriter,
    load_session,
    normalize_action,
)


def frame(fill: int, h: int = 4, w: int = 6) -> np.ndarray:
    """Детерминированный тестовый кадр (BGR uint8)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = fill
    return arr


class TestWriterRoundTrip:
    def test_frame_action_result_roundtrip(self, tmp_path):
        path = tmp_path / "session"
        before = frame(10)
        after = frame(20)
        with SessionWriter(path, meta={"source": "synthetic"}) as writer:
            writer.record_frame(before)
            writer.record_action(
                {"kind": "click", "target": "market_pic", "pos": (12, 34)},
                frame_after=after,
            )
            writer.set_result("ok", {"verify": "list__tab_home"})

        session = load_session(path)
        assert session.meta == {"source": "synthetic"}
        assert len(session.events) == 2

        first, second = session.events
        assert first["type"] == "frame"
        assert second["type"] == "action"
        assert second["action"] == {
            "kind": "click",
            "target": "market_pic",
            "pos": [12, 34],  # кортеж нормализован в список
        }
        assert second["result"] == {"status": "ok", "detail": {"verify": "list__tab_home"}}
        # кадры читаются без потерь
        assert np.array_equal(session.read_image(first["image"]), before)
        assert np.array_equal(session.read_image(second["image"]), after)

    def test_action_without_result_has_null_result(self, tmp_path):
        path = tmp_path / "session"
        with SessionWriter(path) as writer:
            writer.record_frame(frame(1))
            writer.record_action({"kind": "scroll"}, frame_after=frame(2))

        session = load_session(path)
        assert session.events[1]["result"] is None

    def test_action_without_frame_after(self, tmp_path):
        path = tmp_path / "session"
        with SessionWriter(path) as writer:
            writer.record_frame(frame(1))
            writer.record_action({"kind": "click"}, frame_after=None)

        session = load_session(path)
        assert session.events[1]["image"] is None
        assert "image" in session.events[1]  # ключ присутствует (null), формат явный

    def test_frame_event_between_actions(self, tmp_path):
        """Изменение кадра без действия ядра — отдельное frame-событие."""
        path = tmp_path / "session"
        with SessionWriter(path) as writer:
            writer.record_frame(frame(1))
            writer.record_frame(frame(5))

        session = load_session(path)
        assert [e["type"] for e in session.events] == ["frame", "frame"]
        # разные кадры — разные PNG
        assert session.events[0]["image"] != session.events[1]["image"]
        assert len(list((path / "frames").iterdir())) == 2

    def test_identical_frames_share_png(self, tmp_path):
        path = tmp_path / "session"
        with SessionWriter(path) as writer:
            writer.record_frame(frame(7))
            writer.record_frame(frame(7))

        session = load_session(path)
        assert session.events[0]["image"] == session.events[1]["image"]
        assert len(list((path / "frames").iterdir())) == 1


class TestDeterminism:
    def test_same_calls_produce_identical_bytes(self, tmp_path):
        dirs = []
        for name in ("a", "b"):
            path = tmp_path / name
            dirs.append(path)
            with SessionWriter(path, meta={"source": "synthetic"}) as writer:
                writer.record_frame(frame(3))
                writer.record_action({"kind": "click", "pos": [1, 2]}, frame_after=frame(4))
                writer.set_result("failed", {"reason": "таймаут"})

        manifests = [
            (d / "session.json").read_bytes() for d in dirs
        ]
        assert manifests[0] == manifests[1]
        pngs_a = sorted((dirs[0] / "frames").iterdir())
        pngs_b = sorted((dirs[1] / "frames").iterdir())
        assert [p.name for p in pngs_a] == [p.name for p in pngs_b]
        for pa, pb in zip(pngs_a, pngs_b):
            assert pa.read_bytes() == pb.read_bytes()

    def test_manifest_is_human_readable_json(self, tmp_path):
        path = tmp_path / "session"
        with SessionWriter(path, meta={"source": "synthetic"}) as writer:
            writer.record_frame(frame(7))

        raw = (path / "session.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert data["format"] == "l2core-session"
        assert data["version"] == 1
        assert "\n" in raw  # форматировано с отступами, читается человеком


class TestWriterValidation:
    def test_first_event_must_be_frame(self, tmp_path):
        writer = SessionWriter(tmp_path / "session")
        with pytest.raises(SessionFormatError, match="первым событием"):
            writer.record_action({"kind": "click"}, frame_after=frame(1))

    def test_result_without_action_rejected(self, tmp_path):
        writer = SessionWriter(tmp_path / "session")
        writer.record_frame(frame(1))
        with pytest.raises(SessionFormatError, match="после action"):
            writer.set_result("ok")

    def test_existing_nonempty_directory_rejected(self, tmp_path):
        path = tmp_path / "session"
        path.mkdir()
        (path / "junk.txt").write_text("x", encoding="utf-8")
        with pytest.raises(SessionFormatError, match="уже существует"):
            SessionWriter(path)


class TestLoadValidation:
    def _write_manifest(self, tmp_path, data, mkdir_frames=False):
        path = tmp_path / "session"
        path.mkdir()
        (path / "session.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        if mkdir_frames:
            (path / "frames").mkdir()
        return path

    def test_wrong_format_rejected(self, tmp_path):
        path = self._write_manifest(tmp_path, {"format": "other", "version": 1})
        with pytest.raises(SessionFormatError, match="format"):
            load_session(path)

    def test_unsupported_version_rejected(self, tmp_path):
        path = self._write_manifest(
            tmp_path,
            {"format": "l2core-session", "version": 99, "events": []},
        )
        with pytest.raises(SessionFormatError, match="версию"):
            load_session(path)

    def test_empty_events_rejected(self, tmp_path):
        path = self._write_manifest(
            tmp_path,
            {"format": "l2core-session", "version": 1, "events": []},
        )
        with pytest.raises(SessionFormatError, match="событий нет"):

            load_session(path)

    def test_missing_image_file_rejected(self, tmp_path):
        path = self._write_manifest(
            tmp_path,
            {
                "format": "l2core-session",
                "version": 1,
                "events": [{"type": "frame", "image": "frames/000000.png"}],
            },
            mkdir_frames=True,
        )
        session = load_session(path)
        with pytest.raises(SessionFormatError, match="не найден"):
            session.read_image("frames/000000.png")


class TestNormalizeAction:
    def test_numpy_and_tuples_normalized(self):
        action = {
            "kind": "click",
            "pos": (np.int64(5), np.int32(6)),
            "flags": (True,),
        }
        assert normalize_action(action) == {
            "kind": "click",
            "pos": [5, 6],
            "flags": [True],
        }

    def test_normalize_is_stable(self):
        action = {"kind": "scroll", "dir": "down", "n": 3}
        assert normalize_action(normalize_action(action)) == normalize_action(action)
