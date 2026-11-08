"""Семантический валидатор UI-модели (задача 2.3).

Парсер (l2core.model.loader) проверяет формат: структуру, типы, словари.
Валидатор проверяет инварианты спеки ui-model поверх объектной модели:

- state_signatures — у каждого состояния минимум два признака-сигнатуры.
  «Независимость» на уровне модели трактуем как наличие не менее двух
  признаков; требование разных видов детекции — уровень детектора (3.4)
  и трассировки use-cases (6.5).
- state_recover_path — из каждого состояния есть путь в recover-состояние
  по графу переходов модели (рёбра from -> expected). Спека точного
  определения не даёт; интерпретация (зафиксирована здесь и в отчёте 2.3):
  recover-состояние — состояние, id которого начинается с префикса
  «recover» (соглашение модели). Правило применяется, только если в модели
  есть переходы: пустая секция transitions — не ошибка (ruling R3),
  семантику переходов размечают задачи 2.5/6.5. Полный обход графа
  (unknown достижим отовсюду и имеет выход) — задача 6.5.
- state_signature_refs / transition_verify_refs — проверки сигнатур и
  verify семантически корректны: template ссылается на существующий
  элемент и его вариант, ocr — на существующий регион. Непустота verify
  и on_fail — уровень формата, парсер уже требует её (R3: валидатор
  семантику не дублирует).
- transition_states — from/expected перехода ссылаются на известные
  состояния модели.
- rect_in_window — rect'ы элементов (не HUD) и OCR-регионов вписываются
  в окно; точки pixel_probe (координаты от якоря окна) — в окне.
  Этим же правилом отклоняются абсолютные экранные координаты вне слоя
  якоря (спека ui-model, сценарий «Валидатор отклоняет абсолютные
  координаты»): они не вписываются в окно. search_area HUD-элементов
  заданы в клиентской области игры — в окно не вписываются и здесь
  не проверяются.
- template_missing — все кроп-шаблоны вариантов существуют в assets/.
- table_geometry_required — окно с текстовой таблицей списка
  (WINDOWS_WITH_LISTS, состав зафиксирован в 2.4 по мете) обязано иметь
  table_geometry.

Нарушения — структурированные записи Violation (правило, файл, субъект,
деталь), а не строки: allowlist пробелов отгруженных моделей (ruling R10,
ui_model/KNOWN_GAPS.yaml) сопоставляет их машинно.

CLI для людей: python -m l2core.model.validator <файлы...> [--assets DIR]
(0 — валидно, 1 — есть нарушения, 2 — файл не читается).
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import yaml

from .loader import ModelFormatError, load_model
from .types import Check, TransitionSpec, UiModel

# Окна с текстовой таблицей списка: обязательная table_geometry (FORMAT.md).
# Зафиксировано в 2.4 по мете: у shop_buy/shop_sell «списки» — иконные сетки
# («Снаряжение»/«Список покупок»/«Продажа» окна «Торговая лавка»), текстовых
# колонок нет. Таблицы «<Продажа>»/«<Покупка>» (история торговли) принадлежат
# другим окнам, в моделях не описаны — см. комментарий в KNOWN_GAPS.yaml.
WINDOWS_WITH_LISTS = ("market",)

# Соглашение: recover-состояние — id с этим префиксом (см. модульный docstring).
RECOVER_STATE_PREFIX = "recover"


@dataclass(frozen=True)
class Violation:
    """Структурированная запись нарушения инварианта модели."""

    rule: str  # имя правила (state_signatures, rect_in_window, ...)
    file: str  # файл модели
    subject: str  # состояние / элемент / переход / путь шаблона
    detail: str  # человекочитаемая деталь на русском


@dataclass(frozen=True)
class Gap:
    """Запись allowlist известного пробела (ruling R10).

    subject равен субъекту нарушения; "*" покрывает любой субъект.
    """

    file: str  # имя файла модели (без пути)
    rule: str
    subject: str


class GapCheckResult(NamedTuple):
    unmatched: Tuple[Violation, ...]  # нарушения вне allowlist
    stale: Tuple[Gap, ...]  # записи allowlist без нарушения


def validate_model(model: UiModel, assets_dir: Union[str, Path]) -> List[Violation]:
    """Проверить инварианты модели; вернуть список нарушений (пусто — ок).

    assets_dir — каталог assets/ ядра: шаблоны вариантов задаются
    путями относительно него.
    """
    file = model.path
    width, height = model.window.size
    assets = Path(assets_dir)
    violations: List[Violation] = []

    def rect_fits(rect) -> bool:
        return rect.x + rect.width <= width and rect.y + rect.height <= height

    # --- table_geometry у окна со списком ---
    if model.window.id in WINDOWS_WITH_LISTS and model.table_geometry is None:
        violations.append(
            Violation(
                rule="table_geometry_required",
                file=file,
                subject="table_geometry",
                detail=(
                    f"окно «{model.window.id}» со списком обязано иметь "
                    "секцию table_geometry"
                ),
            )
        )

    # --- Состояния: сигнатуры, ссылки, точки проб ---
    for state in model.states.values():
        if len(state.signatures) < 2:
            violations.append(
                Violation(
                    rule="state_signatures",
                    file=file,
                    subject=state.id,
                    detail=(
                        f"у состояния {state.id} {len(state.signatures)} "
                        "признак(а)-сигнатур, требуется минимум два"
                    ),
                )
            )
        for sig in state.signatures:
            violations.extend(_check_refs(sig, model, file, "state_signature_refs", state.id))
            violations.extend(
                _check_probe_points(sig, width, height, file, state.id)
            )

    # --- Переходы: состояния, verify, точки проб ---
    for transition in model.transitions:
        label = _transition_label(transition)
        if transition.source not in model.states:
            violations.append(
                Violation(
                    rule="transition_states",
                    file=file,
                    subject=label,
                    detail=f"исходное состояние {transition.source!r} не описано в states",
                )
            )
        if transition.expected not in model.states:
            violations.append(
                Violation(
                    rule="transition_states",
                    file=file,
                    subject=label,
                    detail=f"целевое состояние {transition.expected!r} не описано в states",
                )
            )
        for check in transition.verify:
            violations.extend(
                _check_refs(check, model, file, "transition_verify_refs", label)
            )
            violations.extend(_check_probe_points(check, width, height, file, label))

    # --- Путь в recover (только если переходы описаны, R3) ---
    if model.transitions:
        violations.extend(_check_recover_paths(model, file))

    # --- rect'ы в окно ---
    for element in model.elements.values():
        if element.hud:
            continue  # search_area — в клиентской области игры, не окна
        if not rect_fits(element.rect):
            violations.append(
                Violation(
                    rule="rect_in_window",
                    file=file,
                    subject=element.id,
                    detail=(
                        f"rect {list(element.rect)} элемента {element.id} "
                        f"не вписывается в окно {width}x{height}"
                    ),
                )
            )
    for region in model.ocr_regions.values():
        if not rect_fits(region.rect):
            violations.append(
                Violation(
                    rule="rect_in_window",
                    file=file,
                    subject=region.id,
                    detail=(
                        f"rect {list(region.rect)} региона {region.id} "
                        f"не вписывается в окно {width}x{height}"
                    ),
                )
            )

    # --- Шаблоны в assets/ ---
    for element in model.elements.values():
        for variant in element.variants.values():
            if not (assets / variant.template).is_file():
                violations.append(
                    Violation(
                        rule="template_missing",
                        file=file,
                        subject=variant.template,
                        detail=(
                            f"элемент {element.id}, вариант {variant.name}: "
                            f"шаблона нет в {assets}/"
                        ),
                    )
                )

    return violations


def _check_refs(
    check: Check, model: UiModel, file: str, rule: str, subject: str
) -> List[Violation]:
    """Проверка ссылок проверки: template -> элемент+вариант, ocr -> регион."""
    violations = []
    if check.kind == "template":
        element = model.elements.get(check.element)
        if element is None:
            violations.append(
                Violation(rule, file, subject, f"неизвестный элемент {check.element!r}")
            )
        elif check.variant not in element.variants:
            violations.append(
                Violation(
                    rule,
                    file,
                    subject,
                    f"у элемента {check.element!r} нет варианта {check.variant!r}",
                )
            )
    elif check.kind == "ocr":
        if check.region not in model.ocr_regions:
            violations.append(
                Violation(rule, file, subject, f"неизвестный OCR-регион {check.region!r}")
            )
    return violations


def _check_probe_points(
    check: Check, width: int, height: int, file: str, subject: str
) -> List[Violation]:
    """Точки pixel_probe заданы от якоря окна и обязаны быть в окне."""
    if check.kind != "pixel_probe":
        return []
    violations = []
    for point in check.points:
        x, y = point.at
        if x >= width or y >= height:
            violations.append(
                Violation(
                    rule="rect_in_window",
                    file=file,
                    subject=subject,
                    detail=(
                        f"точка пробы {list(point.at)} за пределами "
                        f"окна {width}x{height}"
                    ),
                )
            )
    return violations


def _transition_label(transition: TransitionSpec) -> str:
    return f"{transition.source} -> {transition.expected}"


def _check_recover_paths(model: UiModel, file: str) -> List[Violation]:
    """Достижимость recover-состояния из каждого состояния по рёбрам from -> expected.

    Интерпретация «пути в recover» — в модульном docstring. Обход BFS по
    графу переходов; переходы с неизвестными состояниями рёбер не дают
    (они уже отмечены правилом transition_states).
    """
    recover_states = {
        state_id
        for state_id in model.states
        if state_id.startswith(RECOVER_STATE_PREFIX)
    }
    edges = {}
    for transition in model.transitions:
        if (
            transition.source in model.states
            and transition.expected in model.states
        ):
            edges.setdefault(transition.source, set()).add(transition.expected)

    violations = []
    for state_id in model.states:
        if state_id in recover_states:
            continue
        reachable = set()
        stack = [state_id]
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(edges.get(current, ()))
        if not (reachable & recover_states):
            violations.append(
                Violation(
                    rule="state_recover_path",
                    file=file,
                    subject=state_id,
                    detail=(
                        f"из состояния {state_id} нет пути в recover-состояние "
                        "по графу переходов модели"
                    ),
                )
            )
    return violations


def format_report(violations: Sequence[Violation]) -> str:
    """Человекочитаемый отчёт для CLI и диагностики."""
    if not violations:
        return "Нарушений не найдено."
    lines = [f"Найдено нарушений: {len(violations)}"]
    for violation in violations:
        name = Path(violation.file).name
        lines.append(
            f"[{violation.rule}] {name}: {violation.subject} — {violation.detail}"
        )
    return "\n".join(lines)


# --- Allowlist известных пробелов (ruling R10) ---


def load_gaps(path: Union[str, Path]) -> Tuple[Gap, ...]:
    """Загрузить allowlist пробелов отгруженных моделей (YAML).

    Формат: gaps: [{file, rule, subject}]; subject "*" покрывает любой.
    """
    file = Path(path)
    with open(file, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("gaps"), list):
        raise ValueError(f"{file.name}: ожидается список gaps")
    gaps = []
    for index, item in enumerate(data["gaps"]):
        if not isinstance(item, dict):
            raise ValueError(f"{file.name}: gaps[{index}] ожидается mapping")
        try:
            gap_file = item["file"]
            gap_rule = item["rule"]
            gap_subject = item["subject"]
        except KeyError as exc:
            raise ValueError(f"{file.name}: gaps[{index}] нет ключа {exc}") from exc
        if not all(isinstance(v, str) and v for v in (gap_file, gap_rule, gap_subject)):
            raise ValueError(f"{file.name}: gaps[{index}] поля — непустые строки")
        gaps.append(Gap(file=gap_file, rule=gap_rule, subject=gap_subject))
    return tuple(gaps)


def check_against_gaps(
    violations: Sequence[Violation], gaps: Sequence[Gap]
) -> GapCheckResult:
    """Сопоставить нарушения валидатора с allowlist.

    Нарушение покрыто записью с тем же файлом (имя) и правилом и равным
    субъектом либо "*". unmatched — нарушения вне allowlist (CI падает);
    stale — записи без нарушения (пробел закрыт, запись забыли убрать).
    """
    unmatched = []
    covered: List[Optional[Gap]] = [None] * len(gaps)
    for violation in violations:
        name = Path(violation.file).name
        gap = _find_gap(violation, name, gaps)
        if gap is None:
            unmatched.append(violation)
        else:
            covered[gaps.index(gap)] = gap
    stale = [gap for gap, hit in zip(gaps, covered) if hit is None]
    return GapCheckResult(unmatched=tuple(unmatched), stale=tuple(stale))


def _find_gap(
    violation: Violation, name: str, gaps: Sequence[Gap]
) -> Optional[Gap]:
    for gap in gaps:
        if gap.file != name or gap.rule != violation.rule:
            continue
        if gap.subject == "*" or gap.subject == violation.subject:
            return gap
    return None


# --- CLI ---


def _default_assets_dir() -> Path:
    # l2core/model/validator.py -> корень репозитория -> assets/
    return Path(__file__).resolve().parents[2] / "assets"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: валидировать файлы моделей. 0 — ок, 1 — нарушения, 2 — файл не читается."""
    parser = argparse.ArgumentParser(
        prog="python -m l2core.model.validator",
        description="Семантическая валидация UI-моделей (инварианты спеки ui-model)",
    )
    parser.add_argument("models", nargs="+", help="YAML-файлы моделей окон")
    parser.add_argument(
        "--assets",
        default=str(_default_assets_dir()),
        help="каталог assets/ (по умолчанию — assets/ репозитория)",
    )
    args = parser.parse_args(argv)

    all_violations: List[Violation] = []
    for model_path in args.models:
        try:
            model = load_model(model_path)
        except (ModelFormatError, OSError) as exc:
            print(f"ОШИБКА ЧТЕНИЯ {model_path}: {exc}", file=sys.stderr)
            return 2
        violations = validate_model(model, args.assets)
        if violations:
            print(format_report(violations))
            all_violations.extend(violations)
        else:
            print(f"{Path(model_path).name}: модель валидна")

    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
