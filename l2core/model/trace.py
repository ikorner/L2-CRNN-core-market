"""Трассировка use-cases по графу переходов модели (задача 6.5).

Статический анализ графа переходов UI-модели (spec market-scenarios,
«Трассировка use-cases по графу переходов»). Три требования спеки и их
исполнение:

- «Ни одного перехода без verify/on_fail» — здесь НЕ дублируется: непустые
  verify/on_fail требует парсер (l2core.model.loader, R3 — уровень формата),
  семантику ссылок verify проверяет валидатор 2.3 (transition_verify_refs).
  Трассировка строится поверх загруженной объектной модели, куда такой
  переход не доходит; тест-делегирование — tests/test_trace.py.
- «unknown достижим отовсюду и имеет выход» — формализация (unknown —
  runtime-концепт детектора состояний 3.4, явного состояния в модели нет).
  Трассировка синтезирует узел UNKNOWN и три вида рёбер:
    рёбра переходов: from -> expected (как размечены в модели);
    рёбра деградации: из КАЖДОГО состояния -> UNKNOWN (детектор в любой
      момент может вернуть unknown — runtime-факт, в YAML не моделируется);
    рёбра восстановления: UNKNOWN -> каждое recover_-состояние
      (recover-цепочка 5.8 из unknown приводит окно в его recover_-состояние;
      соглашение о префиксе «recover» — из валидатора 2.3).
  «Достижим отовсюду» — аксиома графа (рёбра деградации есть всегда),
  фальсифицировать её моделью нельзя; проверяется тестом на построенном
  графе. Фальсифицируемые правила (возврат в работу трактуется по рёбрам
  ПЕРЕХОДОВ модели: синтезированные рёбра — телепорты деградации/
  восстановления, по ним «возврат» из любого узла тривиален):
    unknown_no_exit — ни одно recover_-состояние не имеет пути в рабочее
      (не-recover) состояние по переходам модели: recover-цепочка не
      возвращает окно в работу;
    recover_no_return — из recover_-состояния нет пути в рабочее состояние
      по переходам модели (падение с указанием состояния — сценарий спеки).
  Путь из каждого состояния В recover_-состояние по модельным переходам —
  правило state_recover_path валидатора 2.3; здесь не дублируется.
- «Ни одной абсолютной координаты вне слоя якоря» — в графе переходов
  координат нет по построению: парсер отклоняет неизвестные ключи перехода
  (допустимы только from/action/element/key/expected/verify/timeout/on_fail),
  действие — ссылка на элемент или клавиша; типизированная объектная модель
  не имеет координатных полей в TransitionSpec/ActionSpec (исполняемая
  граница — парсер, тест-делегирование). Исполняемый остаток трассировки:
  шаги on_fail — только именованные шаги исполнителя (KNOWN_ON_FAIL_STEPS);
  шаг с координатным литералом («click_at_500_300») отклоняется. Паритет
  со словарём исполнителя (l2core.primitives.click) следит guard-тест.
  Координатные поля самой модели (rect/click_point/точки проб) заданы от
  якоря окна и вписываются в окно — правило rect_in_window валидатора 2.3;
  здесь не дублируется.

Пустая секция transitions — трассировка вакуентна (ruling R3, симметрия с
валидатором 2.3: семантику переходов размечают задачи 2.5/6.5).

Запуск — тесты (CI), CLI не требуется (YAGNI, никакого API — 7.1).
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .types import UiModel
from .validator import (  # noqa: F401  (KNOWN_ON_FAIL_STEPS — реэкспорт)
    KNOWN_ON_FAIL_STEPS,
    RECOVER_STATE_PREFIX,
    Violation,
)

# Синтезированный узел трассировки: runtime-unknown детектора 3.4.
UNKNOWN_NODE = "unknown"

# Именованные шаги on_fail исполняют примитивы (l2core.primitives.click
# _run_on_fail); источник единого словаря — валидатор (см. импорт выше):
# трассировка и валидатор навигационной модели ссылаются на одно множество,
# координатные действия в шагах недопустимы (точки кликов — из rect'ов
# элементов, а не литералов графа).


@dataclass(frozen=True)
class TraceGraph:
    """Граф трассировки: узлы = состояния модели + синтезированный unknown.

    transition_edges — только рёбра переходов модели (from -> expected);
    edges — полный граф (переходы + деградация + восстановление, формализация
    — в docstring модуля). Возврат в работу проверяется по transition_edges:
    синтезированные рёбра — телепорты, по ним достижимость тривиальна.
    """

    nodes: frozenset
    transition_edges: Dict[str, frozenset]
    edges: Dict[str, frozenset]


def build_trace_graph(model: UiModel) -> TraceGraph:
    """Построить граф трассировки по модели (формализация unknown — в
    docstring модуля)."""
    nodes: Set[str] = set(model.states)
    nodes.add(UNKNOWN_NODE)
    edges: Dict[str, Set[str]] = {node: set() for node in nodes}
    transition_edges: Dict[str, Set[str]] = {node: set() for node in nodes}

    recover_states = {
        state_id
        for state_id in model.states
        if state_id.startswith(RECOVER_STATE_PREFIX)
    }

    for transition in model.transitions:
        # Рёбра переходов: только между известными состояниями; ссылки на
        # неизвестные состояния отмечает правило transition_states валидатора.
        if (
            transition.source in model.states
            and transition.expected in model.states
        ):
            edges[transition.source].add(transition.expected)
            transition_edges[transition.source].add(transition.expected)

    for state_id in model.states:
        # Рёбра деградации: unknown достижим отовсюду (runtime-факт).
        edges[state_id].add(UNKNOWN_NODE)
    for recover_id in recover_states:
        # Рёбра восстановления: recover-цепочка 5.8 из unknown приводит окно
        # в его recover_-состояние.
        edges[UNKNOWN_NODE].add(recover_id)

    return TraceGraph(
        nodes=frozenset(nodes),
        transition_edges={
            node: frozenset(targets) for node, targets in transition_edges.items()
        },
        edges={node: frozenset(targets) for node, targets in edges.items()},
    )


def reachable(
    graph: TraceGraph, start: str, *, transition_only: bool = False
) -> frozenset:
    """Множество узлов, достижимых из start (обход в глубину).

    transition_only=True — обход только по рёбрам переходов модели (без
    синтезированных рёбер деградации/восстановления): проверка «возврата
    в работу» без телепортов трассировки.
    """
    if transition_only:
        edges = graph.transition_edges
    else:
        edges = graph.edges
    seen: Set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edges.get(current, ()))
    return frozenset(seen)


def _transition_label(transition) -> str:
    return f"{transition.source} -> {transition.expected}"


def trace_navigation(models: Dict[str, UiModel], nav) -> List[Violation]:
    """Трассировка навигационной модели (рецепты открытия окон).

    Связка с окнами — соглашение имён: рецепт `open_<window>` открывает
    окно из `recover_closed` своей модели (поле opens). Проверяется:
    - nav_window_unreachable — на каждое окно обязан существовать рецепт
      open_<window> с opens == <window>;
    - nav_recover_state_missing — в модели окна нет recover_closed;
    - nav_recipe_no_exit — неуспех рецепта не имеет выхода в recover
      (ни on_fail, ни alternate);
    - nav_on_fail_steps — шаги on_fail рецептов только именованные
      (паритет KNOWN_ON_FAIL_STEPS, как для переходов окон).
    """
    file = nav.path
    violations: List[Violation] = []

    def check_on_fail(steps: Tuple[str, ...], subject: str) -> None:
        for step in steps:
            if step not in KNOWN_ON_FAIL_STEPS:
                violations.append(Violation(
                    rule="nav_on_fail_steps",
                    file=file,
                    subject=subject,
                    detail=(
                        f"неизвестный шаг on_fail {step!r}: шаги — только "
                        "именованные шаги исполнителя "
                        f"{sorted(KNOWN_ON_FAIL_STEPS)}"
                    ),
                ))

    for window_id, model in models.items():
        recipe_id = f"open_{window_id}"
        recipe = nav.recipes.get(recipe_id)
        if recipe is None:
            violations.append(Violation(
                rule="nav_window_unreachable",
                file=file,
                subject=window_id,
                detail=(
                    f"нет рецепта {recipe_id!r}: окно недостижимо из "
                    "recover_closed через навигационную модель"
                ),
            ))
            continue
        if recipe.opens != window_id:
            violations.append(Violation(
                rule="nav_window_unreachable",
                file=file,
                subject=window_id,
                detail=(
                    f"рецепт {recipe_id!r} открывает {recipe.opens!r} "
                    f"вместо {window_id!r}"
                ),
            ))
        if "recover_closed" not in model.states:
            violations.append(Violation(
                rule="nav_recover_state_missing",
                file=file,
                subject=window_id,
                detail=(
                    "в модели окна нет состояния recover_closed — точки "
                    "входа рецепта open_<window> не существует"
                ),
            ))
        if not recipe.on_fail and recipe.alternate is None:
            violations.append(Violation(
                rule="nav_recipe_no_exit",
                file=file,
                subject=f"recipes.{recipe_id}",
                detail=(
                    "неуспех рецепта не имеет выхода в recover: ни on_fail, "
                    "ни alternate"
                ),
            ))
        check_on_fail(recipe.on_fail, f"recipes.{recipe_id}")
        for i, step in enumerate(recipe.steps):
            check_on_fail(step.on_fail, f"recipes.{recipe_id}.steps[{i}]")
    return violations


def trace_model(model: UiModel) -> List[Violation]:
    """Трассировка модели; вернуть нарушения (пусто — граф трассируется).

    Пустая секция transitions — вакуум (ruling R3), как у валидатора 2.3.
    """
    if not model.transitions:
        return []

    file = model.path
    graph = build_trace_graph(model)
    recover_states = {
        state_id
        for state_id in model.states
        if state_id.startswith(RECOVER_STATE_PREFIX)
    }
    working_states = set(model.states) - recover_states
    violations: List[Violation] = []

    for transition in model.transitions:
        label = _transition_label(transition)
        for step in transition.on_fail:
            if step not in KNOWN_ON_FAIL_STEPS:
                violations.append(
                    Violation(
                        rule="transition_on_fail_steps",
                        file=file,
                        subject=label,
                        detail=(
                            f"неизвестный шаг on_fail {step!r}: шаги модели — "
                            "только именованные шаги исполнителя "
                            f"{sorted(KNOWN_ON_FAIL_STEPS)}; координатные "
                            "действия в переходах недопустимы"
                        ),
                    )
                )

    recover_returns = {
        recover_id
        for recover_id in recover_states
        if reachable(graph, recover_id, transition_only=True) & working_states
    }
    if not recover_returns:
        violations.append(
            Violation(
                rule="unknown_no_exit",
                file=file,
                subject=UNKNOWN_NODE,
                detail=(
                    "из unknown нет пути в рабочее состояние: ни одно "
                    f"recover_-состояние ({sorted(recover_states) or 'их нет'}) "
                    "не возвращается в работу по переходам модели"
                ),
            )
        )

    for recover_id in sorted(recover_states):
        if recover_id not in recover_returns:
            violations.append(
                Violation(
                    rule="recover_no_return",
                    file=file,
                    subject=recover_id,
                    detail=(
                        f"из recover_-состояния {recover_id} нет пути в "
                        "рабочее состояние по графу переходов: восстановление "
                        "из unknown через это состояние не возвращает окно в работу"
                    ),
                )
            )

    return violations
