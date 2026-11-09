"""JSONL-лог действий (задача 8.3, spec replay-logging «JSONL-лог каждого
действия»).

Каждое действие ядра (вызов действия-метода фасада 7.1 — навигация, табы/
фильтры, поиск, скролл, выставление/старт/отмена лавки, recover) фиксируется
JSONL-записью: действие (тип + цель + аргументы), verify-результат
(success/outcome/unconfirmed/cancelled/attempts/failures), ссылки на
кадры-снимки до/после и ts из инъецированных часов. Метод API = одно
действие ядра с одним verify-итогом (кадры — на границе вызова: «до» = кадр
на входе, «после» = кадр на выходе; НЕ кадры вокруг отдельной HID-команды —
граница зафиксирована в отчёте 8.3; точечная прибавка action_logger в каждый
примитив 5.x отвергнута по минимальной инвазивности).

Ключевые кадры (спека: «сохраняются только ключевые кадры»): запись всегда
пишется, PNG сохраняются только для ключевых записей — не-OK исходов
(FAILED/UNKNOWN — для разбора) и необратимых действий (start_sell — контроль
точки невозврата). Обычные успешные записи идут без кадров. Политика
конфигурируема: log.policy {all_failures, irreversible}; отмена (kill switch
8.1) — не ключевая (диагностика отмены живёт в событии killed и dry-run
намерениях). Пустые frames — {"before": null, "after": null}.

Устройство хранилища (детерминизм, конвенции сессий 3.1 и модалок 3.6):

    log_dir/
      actions-NNNN.jsonl   # по строке на запись; NNNN — счётчик ротации
      frames/<sha1>.png    # снимки по хэшу содержимого (дедуп бесплатно:
                           # одинаковый кадр — один файл, ссылка общая)

- seq — сквозной счётчик записей (возобновляется по существующим файлам:
  перезапуск процесса журнал не обнуляет); ts — инъецированные часы; имён
  с метками времени нет.
- Ротация: запись дописана, размер текущего jsonl превысил max_bytes →
  следующий счётчик (новый файл; запись не разрывается между файлами).
- keep_files: файлов старше лимента хранения не держим. Ключевые записи
  удаляемого файла МИГРИРУЮТ в новейший (оригинальные seq/ts/frames) — так
  ключевые кадры переживают ротацию вместе со своими записями; неключевые
  записи старших файлов удаляются вместе с файлами («удаление старых
  записей»).
- Чистка сирот: PNG, на которые не ссылается ни одна сохранённая запись
  (остатки прерванных сессий, кадры удалённых неключевых записей), удаляются
  при ротации.

Место в слоях — l2core/api (рядом с dryrun.py 8.2), не adapters: это часть
сборки API (поднимается create_api из config.log, живёт в ApiWiring), а не
драйвер внешнего устройства. Запись синхронна и идёт в потоке вызова
(поток стратегии-владельца, thread-owner 7.1) — конкурентности нет по
построению сборки. YAGNI: live-записи вне тестов и прогона 9.4 здесь нет.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from l2core.adapters.session import normalize_action
from l2core.config import LogPolicy

__all__ = ["ActionLogger", "ActionLogError"]

logger = logging.getLogger(__name__)

# actions-NNNN.jsonl: NNNN — счётчик ротации (4+ цифры, детерминизм).
_FILE_RE = re.compile(r"^actions-(\d+)\.jsonl$")
FILE_TEMPLATE = "actions-{:04d}.jsonl"
FRAMES_DIR = "frames"


class ActionLogError(Exception):
    """Ошибка записи лога действий (некорректный кадр, сбой кодирования)."""


class ActionLogger:
    """JSONL-журнал действий с ротацией и ключевыми кадрами (см. модуль).

    Args:
        log_dir: директория лога (actions-NNNN.jsonl + frames/).
        max_bytes: порог ротации текущего jsonl (превышение → новый файл).
        keep_files: сколько файлов хранить (старшие удаляются, ключевые
            записи старших файлов мигрируют в новейший).
        policy: политика ключевых кадров (LogPolicy из l2core.config:
            all_failures — кадры для всех не-OK исходов, irreversible —
            кадры для необратимых действий).
        clock: часы для ts записи (инъекция, в проде time.monotonic).

    Возобновление: на существующей директории продолжает текущий файл
    (максимальный счётчик) и seq (максимальный в файлах + 1).
    """

    def __init__(self, log_dir: str | Path, *,
                 max_bytes: int = 20_000_000,
                 keep_files: int = 5,
                 policy: Optional[LogPolicy] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._dir = Path(log_dir)
        self._max_bytes = max_bytes
        self._keep_files = keep_files
        self._policy = policy if policy is not None else LogPolicy()
        self._clock = clock
        self._dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir = self._dir / FRAMES_DIR
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._counter, self._seq = self._scan_existing()

    # ------------------------------------------------------------------
    # Запись

    def record(self, action: dict, *, success: bool,
               outcome: Any = None,
               unconfirmed: bool = False,
               cancelled: bool = False,
               failures: tuple = (),
               attempts: Optional[int] = None,
               irreversible: bool = False,
               before: Optional[np.ndarray] = None,
               after: Optional[np.ndarray] = None) -> dict:
        """Зафиксировать действие: JSONL-запись (+PNG для ключевых).

        Возвращает записанный словарь (seq/ts присвоены). Ключевость:
        необратимое (policy.irreversible) или не-OK без отмены
        (policy.all_failures) — такие записи несут кадры до/после.
        """
        key = self._is_key(success=success, cancelled=cancelled,
                           irreversible=irreversible)
        frames = {"before": None, "after": None}
        if key:
            if before is not None:
                frames["before"] = self._store_frame(before)
            if after is not None:
                frames["after"] = self._store_frame(after)
        record = {
            "seq": self._seq,
            "ts": self._clock(),
            "key": key,
            "action": normalize_action(action),
            "frames": frames,
            "verify": {
                "success": success,
                # str-Enum (5.7): JSON-сериализация — по значению ("OK");
                # str(outcome) дал бы "ActionOutcome.OK" — не используем.
                "outcome": outcome,
                "unconfirmed": unconfirmed,
                "cancelled": cancelled,
                "attempts": attempts,
                "failures": list(failures),
            },
        }
        self._append_line(self._current_path(), record)
        self._seq += 1
        logger.info("действие #%s %s: success=%s outcome=%s key=%s",
                    record["seq"], record["action"].get("type"),
                    success, record["verify"]["outcome"], key)
        if self._current_path().stat().st_size > self._max_bytes:
            self._rotate()
        return record

    # ------------------------------------------------------------------
    # Ротация и чистка

    def _rotate(self) -> None:
        """Текущий файл превысил лимит: следующий счётчик + удержание
        keep_files (миграция ключевых записей, удаление старших файлов,
        чистка сиротских PNG)."""
        self._counter += 1
        files = self._all_files()
        while len(files) > self._keep_files:
            oldest = files.pop(0)
            self._migrate_key_records(oldest)
            oldest.unlink()
            logger.info("ротация: удалён старый файл лога %s", oldest.name)
        self._cleanup_orphans()

    def _migrate_key_records(self, path: Path) -> None:
        """Ключевые записи удаляемого файла — в новейший (как есть: исходные
        seq/ts/frames) — ключевые кадры переживают ротацию со своими
        записями."""
        key_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # битая строка — не мигрируем
            if record.get("key"):
                key_lines.append(line)
        if key_lines:
            self._append_raw(self._current_path(), key_lines)

    def _cleanup_orphans(self) -> None:
        """Удалить PNG, на которые не ссылается ни одна сохранённая запись."""
        referenced = set()
        for path in self._all_files():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for ref in (record.get("frames") or {}).values():
                    if ref:
                        referenced.add(Path(ref).name)
        for png in self._frames_dir.glob("*.png"):
            if png.name not in referenced:
                png.unlink()

    # ------------------------------------------------------------------
    # Внутреннее

    def _is_key(self, *, success: bool, cancelled: bool,
                irreversible: bool) -> bool:
        if cancelled:
            return False   # отмена (kill switch 8.1) — не ключевая запись
        if irreversible and self._policy.irreversible:
            return True
        return bool(not success and self._policy.all_failures)

    def _store_frame(self, frame: np.ndarray) -> str:
        """PNG-снимок в frames/<sha1>.png; имя — хэш содержимого (дедуп:
        одинаковые кадры — один файл, конвенция сессий 3.1)."""
        import cv2

        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3):
            raise ActionLogError(
                f"кадр должен быть numpy.ndarray BGR uint8 (H, W, 3), "
                f"получено {type(frame).__name__} "
                f"{getattr(frame, 'dtype', None)} "
                f"{getattr(frame, 'shape', None)}")
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise ActionLogError("cv2.imencode не смог закодировать кадр")
        data = encoded.tobytes()
        name = hashlib.sha1(data).hexdigest() + ".png"
        path = self._frames_dir / name
        if not path.exists():
            path.write_bytes(data)
        return f"{FRAMES_DIR}/{name}"

    def _current_path(self) -> Path:
        return self._dir / FILE_TEMPLATE.format(self._counter)

    def _all_files(self) -> list:
        """Все файлы лога, от старшего к новейшему (по счётчику в имени)."""
        files = []
        for path in self._dir.glob("actions-*.jsonl"):
            match = _FILE_RE.match(path.name)
            if match:
                files.append((int(match.group(1)), path))
        return [path for _, path in sorted(files)]

    def _scan_existing(self) -> tuple[int, int]:
        """Возобновление: текущий счётчик = максимум в именах файлов,
        seq = максимальный среди записей + 1 (пусто — (1, 1))."""
        files = self._all_files()
        counter = 1
        seq = 0
        for path in files:
            match = _FILE_RE.match(path.name)
            if match:
                counter = max(counter, int(match.group(1)))
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record.get("seq"), int):
                    seq = max(seq, record["seq"])
        return counter, seq + 1

    @staticmethod
    def _append_line(path: Path, record: dict) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _append_raw(path: Path, lines: list) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
