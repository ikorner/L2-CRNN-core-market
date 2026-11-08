"""Детектор состояний окна торговли (задача 3.4).

Классификация параметризованных состояний (window, params) по кадру и якорю
(сам якорь ищет AnchorDetector, задача 3.3; кэша позиции между кадрами нет —
инвариант 3). Модель принятия решений (spec window-state-detection):

- состояние подтверждается, когда совпали ВСЕ его сигнатуры; валидатор 2.3
  гарантирует минимум две независимые сигнатуры на состояние, состояние с
  одной сигнатурой не подтверждается никогда (защита в глубину);
- сигнатуры оцениваются в иерархии «пиксель-пробы -> template -> OCR»
  (дешёвые первыми) с остановкой состояния на первой несовпавшей: дальнейшие
  признаки отклонённого состояния не вычисляются;
- OCR-сигнатуры оцениваются только при наличии OCR-клиента (контракт
  ``read(region_img) -> (text, confidence)``, design.md раздел адаптеров);
  без клиента состояние с OCR-сигнатурами не подтверждается — деградация с
  диагностикой «нет OCR-клиента», а не исключение;
- ни одно состояние не подтверждено -> unknown с перечнем несовпавших
  признаков; подтверждено несколько -> unknown с журналированием
  конфликтующих состояний и имён их признаков (структурированный logging);
- состояния с префиксом recover_ — глобальный recover (рантайм 5.8,
  соглашение о префиксе совпадает с валидатором 2.3), классификацией
  состояния окна не занимаются.

Сигнатуры отделены от значений params: params возвращаются из размеченного
состояния модели (связка «active-вариант элемента -> params» фиксируется
разметчиком в модели, детектор её только воспроизводит).

Пороги (происхождение):
- template_threshold=0.9: замер 3.4 на скринах base_screenshot/scr/ с
  якорем: позитивы подтверждённых состояний >= 0.998 (на размеченных
  скринах), максимум дискриминирующего ложного совпадения 0.829 (active-кроп
  чужого таба). Часть шаблонов модели не дискриминирует (фильтр-ряд
  «Все» одинаков во всех табах) — строгое «все сигнатуры» это перекрывает;
- color_tolerance=16 на канал: допущение (в отгруженных моделях
  пиксель-проб нет, замера нет; параметр открыт).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from l2core.adapters.template import TemplateMatcher
from l2core.detector.anchor import Anchor, _to_gray
from l2core.model import Check, UiModel

logger = logging.getLogger(__name__)

# Иерархия признаков: дешёвые пиксель-пробы первыми, OCR последним.
KIND_ORDER = {"pixel_probe": 0, "template": 1, "ocr": 2}

# Префикс состояний глобального recover (соглашение совпадает с валидатором
# 2.3: путь в recover-состояние по графу переходов).
RECOVER_PREFIX = "recover_"


@dataclass(frozen=True)
class SignatureResult:
    """Результат оценки одной сигнатуры.

    matched: True — совпала, False — не совпала, None — недоступна
    (OCR без клиента). detail — человекочитаемая диагностика (score,
    причина отказа, текст OCR).
    """

    state_id: str
    kind: str
    name: str
    matched: Optional[bool]
    detail: str


@dataclass(frozen=True)
class StateReport:
    """Результат классификации состояния окна на одном кадре.

    state: None — unknown (params тогда пуст); conflicts — id подтверждённых
    состояний при конфликте; matched/failed — оценённые сигнатуры
    (failed включает недоступные OCR-признаки, ocr_degraded=True).
    """

    window_id: str
    state: Optional[str]
    params: dict
    matched: Tuple[SignatureResult, ...]
    failed: Tuple[SignatureResult, ...]
    conflicts: Tuple[str, ...]
    ocr_degraded: bool


def _signature_name(check: Check) -> str:
    if check.kind == "pixel_probe":
        pts = "+".join(f"{p.at[0]},{p.at[1]}" for p in check.points)
        return f"pixel_probe[{pts}]"
    if check.kind == "template":
        return f"template:{check.element}/{check.variant}"
    return f"ocr:{check.region}"


class StateDetector:
    """Классификатор состояний окон торговли по кадру и якорю.

    Args:
        models: модели окон (id -> UiModel).
        matcher: реализация template matching (см. l2core.adapters.template).
        assets_dir: каталог assets/ с шаблонами модели.
        ocr_client: опциональный OCR-клиент (контракт
            ``read(region_img) -> (text, confidence)``); без него состояния
            с OCR-сигнатурами не подтверждаются (деградация с диагностикой).
        template_threshold: порог NCC template-сигнатур (замер 3.4).
        color_tolerance: допуск пиксель-пробы на канал (допущение).
    """

    def __init__(
        self,
        models: Mapping[str, UiModel],
        matcher: TemplateMatcher,
        assets_dir: Path,
        *,
        ocr_client=None,
        template_threshold: float = 0.9,
        color_tolerance: int = 16,
    ):
        self._models = dict(models)
        self._matcher = matcher
        self._assets_dir = Path(assets_dir)
        self._ocr_client = ocr_client
        self._template_threshold = template_threshold
        self._color_tolerance = color_tolerance
        self._templates = self._load_templates()

    def _load_templates(self) -> dict:
        """Шаблоны template-сигнатур (окно, элемент, вариант) -> серый массив.

        Ссылки модели проверяются здесь же: несуществующий элемент/вариант
        или отсутствующий файл — падение при создании (ошибка разметчика,
        не рантайма).
        """
        import cv2

        templates = {}
        for window_id, model in self._models.items():
            for state_id, state in model.states.items():
                for check in state.signatures:
                    if check.kind == "ocr":
                        if check.region not in model.ocr_regions:
                            raise KeyError(
                                f"OCR-сигнатура состояния {state_id!r} ссылается "
                                f"на несуществующий регион {check.region!r}"
                            )
                        continue
                    if check.kind != "template":
                        continue
                    key = (window_id, check.element, check.variant)
                    if key in templates:
                        continue
                    try:
                        element = model.elements[check.element]
                    except KeyError:
                        raise KeyError(
                            f"сигнатура состояния {state_id!r} ссылается на "
                            f"несуществующий элемент {check.element!r}"
                        ) from None
                    try:
                        spec = element.variants[check.variant]
                    except KeyError:
                        raise KeyError(
                            f"элемент {check.element!r} состояния {state_id!r} "
                            f"не имеет варианта {check.variant!r}"
                        ) from None
                    path = self._assets_dir / spec.template
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        raise FileNotFoundError(f"шаблон сигнатуры не найден: {path}")
                    templates[key] = (np.asarray(image, dtype=np.float64), element)
        return templates

    def detect(self, frame: np.ndarray, anchor: Anchor) -> StateReport:
        """Классификация состояния окна по кадру и свежему якорю."""
        model = self._models[anchor.window_id]
        gray = _to_gray(frame)
        frame_h, frame_w = gray.shape

        confirmed = []
        matched_all = []
        failed_all = []
        for state_id in sorted(model.states):
            if state_id.startswith(RECOVER_PREFIX):
                continue
            state = model.states[state_id]
            if len(state.signatures) < 2:
                failed_all.append(SignatureResult(
                    state_id, "model", f"state:{state_id}", False,
                    "у состояния одна сигнатура — минимум две (валидатор 2.3)",
                ))
                continue
            # Иерархия: пиксель-пробы -> template -> OCR, остановка на провале.
            ordered = sorted(
                enumerate(state.signatures),
                key=lambda p: (KIND_ORDER[p[1].kind], p[0]),
            )
            state_results = []
            rejected = False
            for _, check in ordered:
                result = self._evaluate(
                    check, model, anchor, gray, frame, frame_w, frame_h, state_id
                )
                state_results.append(result)
                if check.kind == "ocr" and result.matched is None:
                    # OCR-клиента нет: деградация, дальнейшие признаки
                    # состояния не вычисляются — состояние не подтвердить.
                    failed_all.append(result)
                    rejected = True
                    break
                if result.matched is not True:
                    failed_all.append(result)
                    rejected = True
                    break
            if not rejected:
                confirmed.append(state_id)
                matched_all.extend(state_results)

        if len(confirmed) == 1:
            state_id = confirmed[0]
            return StateReport(
                window_id=anchor.window_id,
                state=state_id,
                params=dict(model.states[state_id].params),
                matched=tuple(matched_all),
                failed=tuple(failed_all),
                conflicts=(),
                ocr_degraded=False,
            )
        if len(confirmed) > 1:
            feature_names = {
                sid: sorted(r.name for r in matched_all if r.state_id == sid)
                for sid in confirmed
            }
            logger.warning(
                "конфликт сигнатур состояний окна %s: %s — признаки %s",
                anchor.window_id, sorted(confirmed), feature_names,
            )
            return StateReport(
                window_id=anchor.window_id,
                state=None,
                params={},
                matched=tuple(matched_all),
                failed=tuple(failed_all),
                conflicts=tuple(sorted(confirmed)),
                ocr_degraded=False,
            )
        logger.info(
            "состояние окна %s не подтверждено: %s",
            anchor.window_id,
            [f"{r.state_id}:{r.name} ({r.detail})" for r in failed_all],
        )
        return StateReport(
            window_id=anchor.window_id,
            state=None,
            params={},
            matched=(),
            failed=tuple(failed_all),
            conflicts=(),
            ocr_degraded=any(r.matched is None for r in failed_all),
        )

    def _evaluate(self, check: Check, model: UiModel, anchor: Anchor,
                  gray: np.ndarray, frame: np.ndarray,
                  frame_w: int, frame_h: int, state_id: str) -> SignatureResult:
        name = _signature_name(check)
        if check.kind == "pixel_probe":
            return self._evaluate_probe(check, anchor, frame, state_id, name)
        if check.kind == "template":
            return self._evaluate_template(
                check, anchor, gray, frame_w, frame_h, state_id, name)
        return self._evaluate_ocr(check, model, anchor, frame, state_id, name)

    def _evaluate_probe(self, check: Check, anchor: Anchor, frame: np.ndarray,
                        state_id: str, name: str) -> SignatureResult:
        tolerance = self._color_tolerance
        frame_h, frame_w = frame.shape[:2]
        for point in check.points:
            # Точки сигнатуры — от якоря окна, как rect'ы элементов
            # (инвариант 3: абсолютных координат вне слоя якоря нет).
            x, y = anchor.x + point.at[0], anchor.y + point.at[1]
            if not (0 <= x < frame_w and 0 <= y < frame_h):
                return SignatureResult(
                    state_id, check.kind, name, False,
                    f"точка ({x},{y}) вне кадра {frame_w}x{frame_h}")
            actual = frame[y, x].astype(int)
            expected = np.array(point.color, dtype=int)  # модель в RGB
            expected = expected[::-1]  # кадр BGR
            if np.abs(actual - expected).max() > tolerance:
                return SignatureResult(
                    state_id, check.kind, name, False,
                    f"точка {point.at}: цвет {tuple(actual)[::-1]} "
                    f"вне допуска {tolerance} от {point.color}")
        return SignatureResult(
            state_id, check.kind, name, True,
            f"{len(check.points)}/{len(check.points)} точек в допуске")

    def _evaluate_template(self, check: Check, anchor: Anchor, gray: np.ndarray,
                           frame_w: int, frame_h: int,
                           state_id: str, name: str) -> SignatureResult:
        template, element = self._templates[(anchor.window_id, check.element, check.variant)]
        h, w = template.shape
        if element.hud:
            # HUD-элемент: search_area — допустимая область поиска в кадре.
            area = element.search_area
            x0, y0 = area.x, area.y
            region_w, region_h = area.width, area.height
        else:
            rect = element.rect
            x0, y0 = anchor.x + rect.x, anchor.y + rect.y
            region_w, region_h = rect.width, rect.height
        if x0 < 0 or y0 < 0 or x0 + region_w > frame_w or y0 + region_h > frame_h:
            return SignatureResult(
                state_id, check.kind, name, False,
                f"регион [{x0},{y0},{region_w},{region_h}] вне кадра "
                f"{frame_w}x{frame_h}")
        region = gray[y0:y0 + region_h, x0:x0 + region_w]
        if region.shape[0] < h or region.shape[1] < w:
            return SignatureResult(
                state_id, check.kind, name, False,
                f"шаблон ({w}x{h}) больше региона ({region_w}x{region_h})")
        score = self._matcher.match(region, template).score
        matched = score >= self._template_threshold
        return SignatureResult(
            state_id, check.kind, name, matched,
            f"ncc={score:.3f} (порог {self._template_threshold})")

    def _evaluate_ocr(self, check: Check, model: UiModel, anchor: Anchor,
                      frame: np.ndarray, state_id: str,
                      name: str) -> SignatureResult:
        if self._ocr_client is None:
            return SignatureResult(
                state_id, check.kind, name, None,
                "нет OCR-клиента: сигнатура недоступна")
        region = model.ocr_regions[check.region]
        rect = region.rect
        x0, y0 = anchor.x + rect.x, anchor.y + rect.y
        crop = frame[y0:y0 + rect.height, x0:x0 + rect.width]
        if crop.size == 0:
            return SignatureResult(
                state_id, check.kind, name, False,
                f"регион {check.region} вне кадра")
        text, confidence = self._ocr_client.read(crop)
        matched = text.strip() == check.expect.strip()
        return SignatureResult(
            state_id, check.kind, name, matched,
            f"text={text.strip()!r}, confidence={confidence:.2f}, "
            f"ожидалось {check.expect.strip()!r}")
