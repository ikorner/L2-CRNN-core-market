"""Трассировка use-cases по графу переходов модели (l2core.model.trace,
задача 6.5, spec market-scenarios «Трассировка use-cases по графу
переходов»).

Проверки на синтетических графах: unknown достижим отовсюду (аксиома
рёбер деградации) и имеет выход (unknown_no_exit, recover_no_return с
указанием состояния); шаги on_fail — только именованные шаги
исполнителя (нет координатных литералов в переходах); валидный граф и
отгруженные модели проходят. Правила парсера/валидатора 2.3 трассировка
не дублирует — фиксируется тестом-делегированием.
"""

from pathlib import Path

import pytest

from l2core.model import load_model, load_navigation
from l2core.model.trace import (
    KNOWN_ON_FAIL_STEPS,
    UNKNOWN_NODE,
    TraceGraph,
    build_trace_graph,
    reachable,
    trace_model,
)
from l2core.model.types import (
    ActionSpec,
    Check,
    ProbePoint,
    StateSpec,
    TransitionSpec,
    UiModel,
    WindowSpec,
)
from l2core.model.validator import RECOVER_STATE_PREFIX, Violation, format_report
from l2core.model.loader import ModelFormatError

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_MODEL = REPO_ROOT / "ui_model"

# Два признака-сигнатуры (минимум валидатора 2.3) — pixel_probe'ы.
_SIG_A = Check(kind="pixel_probe", points=(ProbePoint(at=(1, 1), color=(0, 0, 0)),))
_SIG_B = Check(kind="pixel_probe", points=(ProbePoint(at=(2, 2), color=(9, 9, 9)),))
_VERIFY = Check(kind="pixel_probe", points=(ProbePoint(at=(3, 3), color=(1, 2, 3)),))


def _state(state_id):
    return StateSpec(id=state_id, params={}, signatures=(_SIG_A, _SIG_B))


def _tr(source, expected, on_fail=("press_esc",)):
    return TransitionSpec(
        source=source,
        action=ActionSpec(type="click", element="element_x"),
        expected=expected,
        verify=(_VERIFY,),
        timeout=None,
        on_fail=on_fail,
    )


def _model(states, transitions, path="synthetic.yaml"):
    return UiModel(
        path=path,
        window=WindowSpec(id="market", title="Синтетика", size=(100, 100)),
        elements={},
        states={state.id: state for state in states},
        transitions=tuple(transitions),
        ocr_regions={},
        table_geometry=None,
    )


# --- Позитивный сценарий: валидный граф проходит ---


def test_valid_graph_passes():
    """Рабочий цикл + recover_-состояние с возвратом в работу."""
    model = _model(
        [_state("a"), _state("b"), _state("recover_r")],
        [_tr("a", "b"), _tr("b", "recover_r"), _tr("recover_r", "a")],
    )
    assert trace_model(model) == []


# --- unknown достижим отовсюду (аксиома рёбер деградации) ---


def test_unknown_reachable_from_every_state():
    """Из любого состояния есть ребро деградации в синтезированный unknown."""
    model = _model(
        [_state("a"), _state("b"), _state("recover_r")],
        [_tr("a", "b"), _tr("b", "recover_r"), _tr("recover_r", "a")],
    )
    graph = build_trace_graph(model)
    assert isinstance(graph, TraceGraph)
    assert UNKNOWN_NODE in graph.nodes
    for state_id in model.states:
        assert UNKNOWN_NODE in reachable(graph, state_id)


def test_unknown_node_not_a_state_of_model():
    """Синтезированный узел не конфликтует с состояниями модели."""
    model = _model([_state("a")], [])
    graph = build_trace_graph(model)
    assert UNKNOWN_NODE not in model.states
    assert UNKNOWN_NODE in graph.nodes


# --- unknown без выхода / recover без возврата ---


def test_unknown_without_exit_fails():
    """recover_-состояний нет — из unknown нет пути в рабочее состояние."""
    model = _model([_state("a"), _state("b")], [_tr("a", "b"), _tr("b", "a")])
    violations = trace_model(model)
    hits = [v for v in violations if v.rule == "unknown_no_exit"]
    assert len(hits) == 1
    assert hits[0].subject == UNKNOWN_NODE
    assert "unknown" in format_report(violations)


def test_recover_state_without_return_fails_with_state_name():
    """recover_-состояние без пути обратно — падение с указанием состояния
    (spec: «трассировка падает с указанием состояния»)."""
    model = _model(
        [_state("a"), _state("b"), _state("recover_ok"), _state("recover_orphan")],
        [_tr("a", "b"), _tr("b", "a"), _tr("a", "recover_ok"), _tr("recover_ok", "a")],
    )
    violations = trace_model(model)
    hits = [v for v in violations if v.rule == "recover_no_return"]
    assert [h.subject for h in hits] == ["recover_orphan"]
    # Выход из unknown есть (через recover_ok) — unknown_no_exit не стреляет.
    assert not [v for v in violations if v.rule == "unknown_no_exit"]


def test_dead_end_recover_closes_unknown_exit():
    """Единственное recover_-состояние — тупик: выхода из unknown нет."""
    model = _model(
        [_state("a"), _state("b"), _state("recover_dead")],
        [_tr("a", "b"), _tr("b", "a"), _tr("a", "recover_dead")],
    )
    violations = trace_model(model)
    assert sorted(v.rule for v in violations) == ["recover_no_return", "unknown_no_exit"]


def test_self_loop_recover_is_not_an_exit():
    """recover_-состояние с петлёй на себя, но без пути в работу."""
    model = _model(
        [_state("a"), _state("recover_loop")],
        [_tr("a", "recover_loop"), _tr("recover_loop", "recover_loop")],
    )
    violations = trace_model(model)
    assert "recover_no_return" in [v.rule for v in violations]
    assert "unknown_no_exit" in [v.rule for v in violations]


# --- Координатные литералы в переходах ---


def test_coordinate_like_on_fail_step_fails():
    """Шаг on_fail вне словаря именованных шагов (координатный литерал)
    отклоняется."""
    model = _model(
        [_state("a"), _state("recover_r")],
        [_tr("a", "recover_r"), _tr("recover_r", "a", on_fail=("click_at_500_300",))],
    )
    violations = trace_model(model)
    hits = [v for v in violations if v.rule == "transition_on_fail_steps"]
    assert len(hits) == 1
    assert "click_at_500_300" in hits[0].detail
    assert "recover_r -> a" in hits[0].subject


def test_known_on_fail_steps_pass():
    model = _model(
        [_state("a"), _state("recover_r")],
        [
            _tr("a", "recover_r", on_fail=("press_esc", "click_free_area")),
            _tr("recover_r", "a"),
        ],
    )
    assert trace_model(model) == []


# --- Пустой граф переходов: вакуум (ruling R3, симметрия с валидатором) ---


def test_empty_transitions_is_vacuous_pass():
    model = _model([_state("a")], [])
    assert trace_model(model) == []


# --- Делегирование правил парсеру/валидатору 2.3 (без дублирования) ---


def test_verify_on_fail_enforced_by_parser(tmp_path):
    """Непустые verify/on_fail — правило формата (парсер, R3): файл без
    verify не загружается; трассировка правило не дублирует."""
    raw = """\
window: {id: market, title: Обзор лавок, size: [100, 100]}
elements:
  el:
    rect: [10, 10, 20, 20]
    role: click_target
states:
  a:
    signatures:
      - {kind: pixel_probe, points: [{at: [1, 1], color: [0, 0, 0]}]}
      - {kind: pixel_probe, points: [{at: [2, 2], color: [9, 9, 9]}]}
  b:
    signatures:
      - {kind: pixel_probe, points: [{at: [3, 3], color: [1, 1, 1]}]}
      - {kind: pixel_probe, points: [{at: [4, 4], color: [2, 2, 2]}]}
transitions:
  - from: a
    action: click
    element: el
    expected: b
    on_fail: [press_esc]
ocr_regions: {}
"""
    file = tmp_path / "no_verify.yaml"
    file.write_text(raw, encoding="utf-8")
    with pytest.raises(ModelFormatError):
        load_model(file)


def test_transition_coordinate_keys_rejected_by_parser(tmp_path):
    """Координатные ключи в переходе отклоняются парсером (unknown keys) —
    исполняемая граница «нет координат в переходах», трассировка её
    не дублирует."""
    raw = """\
window: {id: market, title: Обзор лавок, size: [100, 100]}
elements:
  el:
    rect: [10, 10, 20, 20]
    role: click_target
states:
  a:
    signatures:
      - {kind: pixel_probe, points: [{at: [1, 1], color: [0, 0, 0]}]}
      - {kind: pixel_probe, points: [{at: [2, 2], color: [9, 9, 9]}]}
  b:
    signatures:
      - {kind: pixel_probe, points: [{at: [3, 3], color: [1, 1, 1]}]}
      - {kind: pixel_probe, points: [{at: [4, 4], color: [2, 2, 2]}]}
transitions:
  - from: a
    action: click
    element: el
    expected: b
    click_at: [500, 300]
    verify:
      - {kind: pixel_probe, points: [{at: [3, 3], color: [1, 2, 3]}]}
    on_fail: [press_esc]
ocr_regions: {}
"""
    file = tmp_path / "coord_key.yaml"
    file.write_text(raw, encoding="utf-8")
    with pytest.raises(ModelFormatError):
        load_model(file)


# --- Словарь шагов on_fail — паритет с исполнителем (guard от рассинхрона) ---


def test_on_fail_steps_parity_with_executor():
    """KNOWN_ON_FAIL_STEPS совпадает со словарём исполнителя on_fail
    (l2core.primitives.click): рассинхрон — падение теста."""
    from l2core.primitives import click as click_module

    executor_steps = frozenset(
        {click_module._ON_FAIL_PRESS_ESC, click_module._ON_FAIL_CLICK_FREE_AREA}
    )
    assert KNOWN_ON_FAIL_STEPS == executor_steps


# --- Отгруженные модели проходят трассировку ---


@pytest.mark.parametrize("window", ["market", "shop_buy", "shop_sell"])
def test_shipped_models_pass_trace(window):
    model = load_model(UI_MODEL / f"{window}.yaml")
    nav = load_navigation(UI_MODEL / "navigation.yaml")
    violations = trace_model(model, nav=nav)
    assert violations == [], format_report(violations)


# --- Violation-записи структурированы (совместимы с экосистемой 2.3) ---


def test_violations_are_structured():
    model = _model([_state("a"), _state("b")], [_tr("a", "b")])
    violations = trace_model(model)
    assert violations
    for violation in violations:
        assert isinstance(violation, Violation)
        assert violation.file == "synthetic.yaml"
        assert violation.rule
        assert violation.subject
        assert violation.detail
