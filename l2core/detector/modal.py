"""Перехватчик неизвестных модалок `modal.unknown` (задача 3.6).

Спека window-state-detection («Перехватчик неизвестных модалок»): контур/
текст вне известных окон → снимок в лог, состояние modal.unknown, переход
в recover; известные типы модалок добавляются в модель по мере досъёмки.

R4 (честность): надёжная детекция «контура/текста вне известных окон» из
одного кадра из сырья не выводится. Реализован честный минимум — механизм
перехватчика (интерфейс, снимок PNG + JSONL-строка + запись в лог, событие
modal.unknown, точка входа recover) плюс консервативная эвристика кандидата:

- окно игры доступно (иначе модалка неотличима от закрытого клиента);
- ни один якорь известных окон на кадре не найден (вызывающий код передаёт
  результаты find_anchor по всем окнам модели);
- в центральной области клиентской области окна игры есть крупный яркий
  прямоугольный регион: связный компонент ярких пикселей (порог яркости как
  у первого этапа якоря, anchor.py), занимающий долю ширины/высоты клиента
  и заполненный яркими пикселями выше доли заполнения.

Ограничения (документировано, см. отчёт 3.6 и KNOWN_GAPS/OWNER_QUESTIONS):
эвристика видит «крупное яркое пятно», а не «контур/текст»: часть игровых
кадров без окна торговли (открытые лавки «<Покупка>/<Продажа>», яркие
HUD-экраны) даёт кандидата — это зафиксировано characterization-тестами
как поведение эвристики, а не ложный позитив известного окна. Парковка:
известные типы модалок по мере досъёмки переходят в модель (явные
сигнатуры состояний, детектор 3.4), эвристика остаётся последней линией.

Граница (YAGNI): исполнителя recover (5.8) здесь нет — on_unknown callback
получает отчёт, решение о recover за вызывающим кодом.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from l2core.adapters.win32 import GameWindow
from l2core.detector.anchor import Anchor

logger = logging.getLogger(__name__)

# Состояние перехватчика (спека window-state-detection).
MODAL_UNKNOWN = "modal.unknown"


@dataclass(frozen=True)
class ModalReport:
    """Результат проверки кадра на неизвестную модалку.

    detected — эвристика нашла кандидата; state — MODAL_UNKNOWN при
    срабатывании, иначе None; region — (x, y, w, h) в пикселях кадра;
    reason — диагностическая причина срабатывания/отказа; snapshot_path —
    путь к PNG-снимку (None без срабатывания).
    """

    detected: bool
    state: Optional[str]
    region: Optional[Tuple[int, int, int, int]]
    reason: Optional[str]
    snapshot_path: Optional[Path]


class ModalInterceptor:
    """Эвристический перехватчик неизвестных модалок (см. докстринг модуля).

    Args:
        log_dir: директория логов: снимки modal_unknown_NNNN.png + файл
            modal.jsonl (по строке на срабатывание: событие, координаты
            региона, причина, имя снимка).
        on_unknown: опциональный callback(ModalReport) — точка входа
            recover (5.8); вызывается после записи снимка и лога.
        brightness: порог яркости серого, пиксель «яркий».
        min_size_fraction: минимальные доли ширины/высоты клиента окна
            игры для кандидата (регион меньше — кнопка/иконка, не модалка).
        min_fill_fraction: минимальная доля ярких пикселей внутри рамки
            региона (защита от разрозненных бликов).
        central_fraction: доля клиента по каждой оси, образующая
            центральную область поиска (модалки встают по центру экрана).
        downscale: шаг прореживания маски перед поиском компонент
            (ускорение; координаты пересчитываются обратно в пиксели).
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        on_unknown: Optional[Callable[[ModalReport], None]] = None,
        brightness: int = 140,
        min_size_fraction: float = 0.15,
        min_fill_fraction: float = 0.5,
        central_fraction: Tuple[float, float] = (0.5, 0.6),
        downscale: int = 4,
    ):
        self._log_dir = Path(log_dir)
        self._on_unknown = on_unknown
        self._brightness = brightness
        self._min_size_fraction = min_size_fraction
        self._min_fill_fraction = min_fill_fraction
        self._central_fraction = central_fraction
        self._downscale = downscale
        self._counter = 0

    def check(
        self,
        frame: np.ndarray,
        game: GameWindow,
        anchors: Mapping[str, Optional[Anchor]],
    ) -> ModalReport:
        """Проверка кадра; при срабатывании — снимок, лог, событие."""
        if not game.ok or game.rect is None:
            return ModalReport(
                detected=False, state=None, region=None,
                reason=f"окно игры недоступно ({game.reason}): "
                       f"модалка неотличима, эвристика не применяется",
                snapshot_path=None,
            )
        found = [window_id for window_id, anchor in anchors.items()
                 if anchor is not None]
        if found:
            return ModalReport(
                detected=False, state=None, region=None,
                reason=f"якорь известного окна найден ({', '.join(found)}): "
                       f"эвристика не применяется",
                snapshot_path=None,
            )

        region = self._find_region(frame, game.rect)
        if region is None:
            return ModalReport(
                detected=False, state=None, region=None,
                reason="крупный яркий прямоугольный регион в центральной "
                       "области не найден",
                snapshot_path=None,
            )

        x, y, w, h = region
        reason = (f"якорей известных окон нет, в центральной области "
                  f"крупный яркий регион ({x},{y}) {w}x{h}")
        snapshot_path = self._save_snapshot(frame)
        self._append_jsonl(region, reason, snapshot_path)
        logger.warning("%s: %s; причина: %s; снимок: %s",
                       MODAL_UNKNOWN, region, reason, snapshot_path.name)
        report = ModalReport(
            detected=True, state=MODAL_UNKNOWN, region=region,
            reason=reason, snapshot_path=snapshot_path,
        )
        if self._on_unknown is not None:
            self._on_unknown(report)
        return report

    # --- эвристика ---

    def _find_region(
        self, frame: np.ndarray, game_rect: Tuple[int, int, int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Кандидат: связный яркий компонент центральной области клиента."""
        gx, gy, gw, gh = game_rect
        fh, fw = frame.shape[:2]
        x0 = gx + int(gw * (1 - self._central_fraction[0]) / 2)
        x1 = gx + int(gw * (1 + self._central_fraction[0]) / 2)
        y0 = gy + int(gh * (1 - self._central_fraction[1]) / 2)
        y1 = gy + int(gh * (1 + self._central_fraction[1]) / 2)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(fw, x1), min(fh, y1)
        if x1 - x0 < self._downscale * 2 or y1 - y0 < self._downscale * 2:
            return None

        crop = frame[y0:y1, x0:x1]
        gray = crop[..., 0] * 0.114 + crop[..., 1] * 0.587 + crop[..., 2] * 0.299
        mask = gray > self._brightness

        step = self._downscale
        h, w = mask.shape
        small_h, small_w = h // step, w // step
        if small_h == 0 or small_w == 0:
            return None
        # Прореживание: ячейка яркая, если в ней есть яркие пиксели.
        small = mask[: small_h * step, : small_w * step].reshape(
            small_h, step, small_w, step).any(axis=(1, 3))

        component = self._largest_component(small)
        if component is None:
            return None
        sx, sy, sw, sh = component
        rx = x0 + sx * step
        ry = y0 + sy * step
        rw, rh = sw * step, sh * step

        if rw < gw * self._min_size_fraction or rh < gh * self._min_size_fraction:
            return None
        fill = float(mask[sy * step: sy * step + rh, sx * step: sx * step + rw].mean())
        if fill < self._min_fill_fraction:
            return None
        return (rx, ry, rw, rh)

    @staticmethod
    def _largest_component(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Самый крупный 4-связный компонент True: рамка (x, y, w, h)."""
        seen = np.zeros(mask.shape, dtype=bool)
        best: Optional[Tuple[int, int, int, int]] = None
        best_area = 0
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if not mask[y, x] or seen[y, x]:
                    continue
                # BFS по компоненту.
                queue = deque([(y, x)])
                seen[y, x] = True
                min_x = max_x = x
                min_y = max_y = y
                area = 0
                while queue:
                    cy, cx = queue.popleft()
                    area += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                                   (cy, cx - 1), (cy, cx + 1)):
                        if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                                and mask[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            queue.append((ny, nx))
                if area > best_area:
                    best_area = area
                    best = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        return best

    # --- журналирование ---

    def _save_snapshot(self, frame: np.ndarray) -> Path:
        """PNG-снимок кадра; имя — счётчик (детерминизм, без меток времени)."""
        import cv2

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        path = self._log_dir / f"modal_unknown_{self._counter:04d}.png"
        if not cv2.imwrite(str(path), frame):
            raise IOError(f"не удалось записать снимок модалки: {path}")
        return path

    def _append_jsonl(
        self, region: Tuple[int, int, int, int], reason: str, path: Path
    ) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "event": MODAL_UNKNOWN,
            "region": list(region),
            "reason": reason,
            "snapshot": path.name,
        }
        with open(self._log_dir / "modal.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
