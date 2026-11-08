"""
Формат файла сессии (версия 1) и программный API записи/чтения.

Сессия — директория:

    <сессия>/
      session.json    # манифест: формат, версия, мета-информация, события
      frames/*.png    # кадры (PNG; массив numpy BGR uint8, сериализация без потерь)

session.json (читаем человеком: JSON с отступами, без меток времени):

    {
      "format": "l2core-session",
      "version": 1,
      "meta": {...},               # произвольные данные записавшего
      "events": [
        {"type": "frame",  "image": "frames/000000.png"},
        {"type": "action", "action": {...}, "image": "frames/000001.png",
         "result": {"status": "ok", "detail": {...}}}
      ]
    }

Семантика событий (как сессия моделирует «эффекты действий»):

- frame — текущий кадр изменился без действия ядра (анимация, чужие
  изменения экрана). Первое событие сессии обязано быть frame: это
  начальное состояние, которое ReplayScreen отдаёт до первого действия.
- action — действие ядра. "action" — нормализованный JSON-словарь
  (структурированная запись, пригодная для assert'ов); "image" — кадр,
  ставший текущим сразу после действия (эффект действия в replay:
  ReplayScreen не кликает, а переходит на этот кадр). "image" может быть
  null — действие записано, но кадр-после не зафиксирован (сессия
  оборвана до следующего grab).
- result — verify-результат действия (заполняется set_result после
  выполнения и проверки), null пока не записан.

Replay-семантика (курсор — текущая позиция в событиях):

- grab() отдаёт кадр текущей позиции; повторные grab между действиями
  возвращают один и тот же кадр (прогон детерминирован);
- notify_action(a) ищет СЛЕДУЮЩЕЕ событие action, сопоставляет
  нормализованное действие (несовпадение — ActionMismatchError: сценарий
  отклонился от записи) и переносит курсор на него; событий action дальше
  нет — SessionExhaustedError.

Программная запись без живого экрана (R7, путь задачи 9.2): сессия
собирается через SessionWriter — record_frame/record_action/set_result,
save. Действия и кадры-после задаются скриптово.

Детерминизм: меток времени в формате нет (только если переданы извне в
meta); байты session.json и PNG полностью определены последовательностью
вызовов записи.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FORMAT_NAME = "l2core-session"
FORMAT_VERSION = 1
MANIFEST_NAME = "session.json"
FRAMES_DIR = "frames"


class SessionError(Exception):
    """Базовое исключение модуля сессий."""


class SessionFormatError(SessionError):
    """Нарушение формата сессии (манифест, события, отсутствующие кадры)."""


# --- Нормализация действий ---


def _normalize(value: Any) -> Any:
    """Преобразовать значение в JSON-совместимый канонический вид.

    Кортежи -> списки, скаляры numpy -> python-типы, ключи -> строки.
    """
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_normalize(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_action(action: dict) -> dict:
    """Каноническая форма действия для хранения и сопоставления в replay."""
    if not isinstance(action, dict):
        raise SessionFormatError(f"действие должно быть словарём, получено {action!r}")
    return _normalize(action)


def actions_equal(a: Any, b: Any) -> bool:
    """Структурное равенство двух нормализованных действий."""
    return _normalize(a) == _normalize(b)


# --- Запись сессии ---


class SessionWriter:
    """Потоковая запись сессии: кадры (PNG) пишутся сразу, манифест — в save.

    Идемпотентность: повторный save не меняет результат; одинаковые кадры
    (по содержимому) не дублируются на диске — события ссылаются на один PNG.
    """

    def __init__(self, path: str | Path, meta: dict | None = None):
        self._path = Path(path)
        if self._path.exists() and any(self._path.iterdir()):
            raise SessionFormatError(
                f"директория сессии уже существует и не пуста: {self._path}"
            )
        self._frames_dir = self._path / FRAMES_DIR
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._meta = _normalize(meta or {})
        self._events: list[dict] = []
        self._image_by_hash: dict[str, str] = {}
        self._counter = 0
        self._saved = False

    def record_frame(self, frame: np.ndarray) -> str:
        """Записать frame-событие (текущий кадр). Возвращает ref на PNG."""
        ref = self._store_image(frame)
        self._events.append({"type": "frame", "image": ref})
        return ref

    def record_action(self, action: dict, frame_after: np.ndarray | None) -> None:
        """Записать action-событие; frame_after — кадр после действия (или None)."""
        if not self._events:
            raise SessionFormatError(
                "первым событием сессии должен быть frame (начальное состояние)"
            )
        ref = self._store_image(frame_after) if frame_after is not None else None
        self._events.append(
            {"type": "action", "action": normalize_action(action), "image": ref,
             "result": None}
        )

    def set_result(self, status: str, detail: dict | None = None) -> None:
        """Привязать verify-результат к последнему action-событию."""
        if not self._events or self._events[-1]["type"] != "action":
            raise SessionFormatError("set_result допустим только после action-события")
        result: dict[str, Any] = {"status": status}
        if detail is not None:
            result["detail"] = _normalize(detail)
        self._events[-1]["result"] = result

    def save(self) -> None:
        """Записать session.json (атомарно). Повторный вызов — no-op."""
        if self._saved:
            return
        manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "meta": self._meta,
            "events": self._events,
        }
        text = json.dumps(manifest, ensure_ascii=False, indent=2)
        tmp = self._path / (MANIFEST_NAME + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(self._path / MANIFEST_NAME)
        self._saved = True

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.save()

    def _store_image(self, frame: np.ndarray) -> str:
        _require_frame(frame)
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise SessionError("cv2.imencode не смог закодировать кадр")
        data = encoded.tobytes()
        digest = hashlib.sha1(data).hexdigest()
        existing = self._image_by_hash.get(digest)
        if existing is not None:
            return existing
        ref = f"{FRAMES_DIR}/{self._counter:06d}.png"
        self._counter += 1
        (self._path / ref).write_bytes(data)
        self._image_by_hash[digest] = ref
        return ref


def _require_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3:
        raise SessionFormatError(
            f"кадр должен быть numpy.ndarray BGR uint8 (H, W, 3), получено "
            f"{type(frame).__name__} {getattr(frame, 'dtype', None)} "
            f"{getattr(frame, 'shape', None)}"
        )


# --- Чтение сессии ---


class Session:
    """Загруженная сессия: мета, события, чтение кадров по ref."""

    def __init__(self, path: Path, meta: dict, events: list[dict]):
        self.path = path
        self.meta = meta
        self.events = events
        self._cache: dict[str, np.ndarray] = {}

    def read_image(self, ref: str) -> np.ndarray:
        """Декодировать кадр по ref ("frames/000000.png"); кэш в рамках сессии."""
        if ref in self._cache:
            return self._cache[ref]
        file_path = self.path / ref
        if not file_path.is_file():
            raise SessionFormatError(f"файл кадра не найден: {file_path}")
        img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if img is None:
            raise SessionFormatError(f"не удалось декодировать кадр: {file_path}")
        self._cache[ref] = img
        return img


def load_session(path: str | Path) -> Session:
    """Загрузить и валидировать сессию (формат, версия, структура событий)."""
    session_path = Path(path)
    manifest_path = session_path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SessionFormatError(f"нет манифеста сессии: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SessionFormatError(f"манифест не является JSON: {e}") from e

    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise SessionFormatError(
            f"это не сессия {FORMAT_NAME!r}: format={data.get('format')!r}"
        )
    version = data.get("version")
    if version != FORMAT_VERSION:
        raise SessionFormatError(
            f"неподдерживаемую версию формата: {version!r} (ожидается {FORMAT_VERSION})"
        )
    events = data.get("events")
    if not isinstance(events, list) or not events:
        raise SessionFormatError("в манифесте событий нет или список пуст")
    if events[0].get("type") != "frame":
        raise SessionFormatError("первым событием сессии должен быть frame")
    for event in events:
        if not isinstance(event, dict) or event.get("type") not in ("frame", "action"):
            raise SessionFormatError(f"неизвестное событие сессии: {event!r}")

    return Session(session_path, data.get("meta") or {}, events)
