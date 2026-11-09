"""Тесты валидатора UI-модели (l2core.model.validator, задача 2.3).

Каждое правило спеки ui-model имеет позитивный и негативный сценарий
на синтетических фикстурах (YAML пишется в tmp_path, ruling R10):
сигнатуры состояний, путь в recover, семантика переходов, вписывание
rect'ов в окно, существование шаблонов в assets/, allowlist пробелов
для отгруженных моделей.
"""

from pathlib import Path

import pytest
import yaml

from l2core.model import load_model
from l2core.model.validator import (
    WINDOWS_WITH_LISTS,
    check_against_gaps,
    format_report,
    load_gaps,
    main,
    validate_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_MODEL = REPO_ROOT / "ui_model"
ASSETS = REPO_ROOT / "assets"
KNOWN_GAPS = UI_MODEL / "KNOWN_GAPS.yaml"

# Минимальная валидная модель (market): два состояния, переход с verify,
# rect'ы в окне, шаблоны на месте, table_geometry присутствует.
MINIMAL_MODEL = """\
window: {id: market, title: Обзор лавок, size: [100, 100]}
elements:
  tab_home:
    rect: [10, 10, 20, 20]
    role: click_target
    variants:
      normal: {template: market/tab_home_normal.png, click_point: [20, 20]}
states:
  list_home:
    params: {tab: home}
    signatures:
      - {kind: pixel_probe, points: [{at: [5, 5], color: [255, 220, 120]}]}
      - {kind: template, element: tab_home, variant: normal}
  recover_closed:
    params: {}
    signatures:
      - {kind: pixel_probe, points: [{at: [1, 1], color: [0, 0, 0]}]}
      - {kind: pixel_probe, points: [{at: [3, 3], color: [10, 10, 10]}]}
transitions:
  - from: list_home
    action: click
    element: tab_home
    expected: recover_closed
    verify:
      - {kind: template, element: tab_home, variant: normal}
    on_fail: [press_esc]
ocr_regions:
  window_title: {rect: [0, 0, 60, 12], grammar: text}
table_geometry:
  col_seps: [10, 50]
  row_y0: 40
  row_pitch: 10
  max_rows: 5
  columns:
    price: {zone: [1, 1, 20, 8], grammar: price}
"""

# --- Позитивный сценарий: минимальная модель проходит ---


def test_minimal_model_passes(tmp_path):
    model, assets = _make_model(tmp_path, MINIMAL_MODEL)
    assert validate_model(model, assets) == []


# --- Правило state_signatures: минимум два признака на состояние ---


def test_state_with_single_signature_rejected(tmp_path):
    model, assets = _make_model(
        tmp_path,
        MINIMAL_MODEL.replace(
            "      - {kind: pixel_probe, points: [{at: [5, 5], color: [255, 220, 120]}]}\n",
            "",
        ),
    )
    violations = validate_model(model, assets)
    hits = [v for v in violations if v.rule == "state_signatures"]
    assert len(hits) == 1
    assert hits[0].subject == "list_home"
    assert "list_home" in format_report(violations)


def test_state_with_two_signatures_passes(tmp_path):
    model, assets = _make_model(tmp_path, MINIMAL_MODEL)
    assert not [v for v in validate_model(model, assets) if v.rule == "state_signatures"]


# --- Правило state_recover_path: путь к recover-состоянию по графу ---


def test_state_without_path_to_recover_rejected(tmp_path):
    """Переход ведёт не в recover-состояние — оба состояния без пути."""
    broken = MINIMAL_MODEL.replace("expected: recover_closed", "expected: list_home")
    model, assets = _make_model(tmp_path, broken)
    violations = validate_model(model, assets)
    subjects = {v.subject for v in violations if v.rule == "state_recover_path"}
    # recover_closed — само recover-состояние (проверке не подлежит);
    # list_home зациклён на себе и recover не достигает.
    assert subjects == {"list_home"}


def test_state_with_path_to_recover_passes(tmp_path):
    model, assets = _make_model(tmp_path, MINIMAL_MODEL)
    assert not [v for v in validate_model(model, assets) if v.rule == "state_recover_path"]


def test_empty_transitions_do_not_trigger_recover_rule(tmp_path):
    """R3: пустая секция transitions — не ошибка; правило не применяется."""
    model, assets = _make_model(
        tmp_path,
        MINIMAL_MODEL.replace(
            "transitions:\n  - from: list_home\n    action: click\n    element: tab_home\n    expected: recover_closed\n    verify:\n      - {kind: template, element: tab_home, variant: normal}\n    on_fail: [press_esc]\n",
            "transitions: []\n",
        ),
    )
    assert not [v for v in validate_model(model, assets) if v.rule == "state_recover_path"]


# --- Правило check_refs: ссылки проверок резолвятся ---


def test_signature_with_unknown_element_rejected(tmp_path):
    broken = MINIMAL_MODEL.replace(
        "- {kind: template, element: tab_home, variant: normal}\n"
        "  recover_closed:",
        "- {kind: template, element: tab_nowhere, variant: normal}\n"
        "  recover_closed:",
    )
    model, assets = _make_model(tmp_path, broken)
    hits = [v for v in validate_model(model, assets) if v.rule == "state_signature_refs"]
    assert len(hits) == 1
    assert hits[0].subject == "list_home"
    assert "tab_nowhere" in hits[0].detail


def test_signature_with_unknown_variant_rejected(tmp_path):
    broken = MINIMAL_MODEL.replace(
        "- {kind: template, element: tab_home, variant: normal}\n"
        "  recover_closed:",
        "- {kind: template, element: tab_home, variant: active}\n"
        "  recover_closed:",
    )
    model, assets = _make_model(tmp_path, broken)
    hits = [v for v in validate_model(model, assets) if v.rule == "state_signature_refs"]
    assert len(hits) == 1
    assert "active" in hits[0].detail


def test_verify_with_unknown_region_rejected(tmp_path):
    broken = MINIMAL_MODEL.replace(
        "verify:\n      - {kind: template, element: tab_home, variant: normal}",
        "verify:\n      - {kind: ocr, region: nowhere, expect: текст}",
    )
    model, assets = _make_model(tmp_path, broken)
    hits = [v for v in validate_model(model, assets) if v.rule == "transition_verify_refs"]
    assert len(hits) == 1
    assert "nowhere" in hits[0].detail


# --- Правило transition_states: from/expected — известные состояния ---


def test_transition_from_unknown_state_rejected(tmp_path):
    broken = MINIMAL_MODEL.replace("from: list_home", "from: list_nowhere")
    model, assets = _make_model(tmp_path, broken)
    hits = [v for v in validate_model(model, assets) if v.rule == "transition_states"]
    assert len(hits) == 1
    assert "list_nowhere" in hits[0].subject or "list_nowhere" in hits[0].detail


# --- Правило rect_in_window: все rect'ы вписываются в окно ---


def test_element_rect_outside_window_rejected(tmp_path):
    """Абсолютные экранные координаты не вписываются в окно (спека ui-model)."""
    broken = MINIMAL_MODEL.replace("rect: [10, 10, 20, 20]", "rect: [900, 10, 20, 20]")
    model, assets = _make_model(tmp_path, broken)
    hits = [v for v in validate_model(model, assets) if v.rule == "rect_in_window"]
    assert len(hits) == 1
    assert hits[0].subject == "tab_home"
    assert "tab_home" in format_report(hits)


def test_ocr_region_outside_window_rejected(tmp_path):
    broken = MINIMAL_MODEL.replace(
        "window_title: {rect: [0, 0, 60, 12], grammar: text}",
        "window_title: {rect: [0, 0, 60, 120], grammar: text}",
    )
    model, assets = _make_model(tmp_path, broken)
    hits = [v for v in validate_model(model, assets) if v.rule == "rect_in_window"]
    assert len(hits) == 1
    assert hits[0].subject == "window_title"


def test_probe_point_outside_window_rejected(tmp_path):
    broken = MINIMAL_MODEL.replace("at: [5, 5]", "at: [500, 5]")
    model, assets = _make_model(tmp_path, broken)
    hits = [v for v in validate_model(model, assets) if v.rule == "rect_in_window"]
    assert len(hits) == 1
    assert hits[0].subject == "list_home"


def test_rects_inside_window_pass(tmp_path):
    model, assets = _make_model(tmp_path, MINIMAL_MODEL)
    assert not [v for v in validate_model(model, assets) if v.rule == "rect_in_window"]


# --- Правило grid_in_window: сетка слотов лавки в окне (6.2) ---


def test_grid_geometry_outside_window_rejected(tmp_path):
    """Ячейки сетки, выходящие за окно, — нарушение (координаты от якоря)."""
    broken = MINIMAL_MODEL + (
        "grid_geometry:\n"
        "  origin: [90, 90]\n"
        "  pitch: [10, 10]\n"
        "  cell: [20, 20]\n"
        "  cols: 2\n"
        "  rows: 1\n"
    )
    model, assets = _make_model(tmp_path, broken)
    hits = [v for v in validate_model(model, assets) if v.rule == "grid_in_window"]
    assert len(hits) == 1
    assert hits[0].subject == "grid_geometry"


def test_grid_geometry_inside_window_pass(tmp_path):
    broken = MINIMAL_MODEL + (
        "grid_geometry:\n"
        "  origin: [10, 10]\n"
        "  pitch: [20, 20]\n"
        "  cell: [10, 10]\n"
        "  cols: 2\n"
        "  rows: 2\n"
    )
    model, assets = _make_model(tmp_path, broken)
    assert not [v for v in validate_model(model, assets) if v.rule == "grid_in_window"]


# --- Правило template_missing: шаблоны есть в assets/ ---


def test_missing_template_rejected(tmp_path):
    model, assets = _make_model(tmp_path, MINIMAL_MODEL, templates=[])
    hits = [v for v in validate_model(model, assets) if v.rule == "template_missing"]
    assert len(hits) == 1
    assert hits[0].subject == "market/tab_home_normal.png"


def test_existing_template_passes(tmp_path):
    model, assets = _make_model(tmp_path, MINIMAL_MODEL)
    assert not [v for v in validate_model(model, assets) if v.rule == "template_missing"]


# --- Правило table_geometry_required: окна со списком обязаны иметь геометрию ---


@pytest.mark.parametrize("window", list(WINDOWS_WITH_LISTS))
def test_window_with_list_requires_table_geometry(tmp_path, window):
    text = MINIMAL_MODEL.replace("id: market", f"id: {window}")
    text = text.replace(
        "table_geometry:\n  col_seps: [10, 50]\n  row_y0: 40\n  row_pitch: 10\n  max_rows: 5\n  columns:\n    price: {zone: [1, 1, 20, 8], grammar: price}\n",
        "",
    )
    model, assets = _make_model(tmp_path, text)
    hits = [v for v in validate_model(model, assets) if v.rule == "table_geometry_required"]
    assert len(hits) == 1


def test_table_geometry_present_passes(tmp_path):
    model, assets = _make_model(tmp_path, MINIMAL_MODEL)
    assert not [
        v for v in validate_model(model, assets) if v.rule == "table_geometry_required"
    ]


# --- Структура нарушений и отчёт ---


def test_violation_is_structured(tmp_path):
    broken = MINIMAL_MODEL.replace("rect: [10, 10, 20, 20]", "rect: [900, 10, 20, 20]")
    model, assets = _make_model(tmp_path, broken)
    (hit,) = validate_model(model, assets)
    assert hit.rule == "rect_in_window"
    assert hit.file == str(model.path)
    assert hit.subject == "tab_home"
    assert hit.detail  # человекочитаемая деталь на русском


def test_format_report_lists_every_violation(tmp_path):
    broken = MINIMAL_MODEL.replace(
        "      - {kind: pixel_probe, points: [{at: [5, 5], color: [255, 220, 120]}]}\n", ""
    )
    model, assets = _make_model(tmp_path, broken)
    report = format_report(validate_model(model, assets))
    assert "state_signatures" in report
    assert "list_home" in report
    assert "market.yaml" in report


# --- CLI ---


def test_cli_exit_zero_on_valid_model(tmp_path, capsys):
    model, assets = _make_model(tmp_path, MINIMAL_MODEL, name="market.yaml")
    code = main([model.path, "--assets", str(assets)])
    assert code == 0
    assert "валидна" in capsys.readouterr().out


def test_cli_exit_one_on_violations(tmp_path, capsys):
    broken = MINIMAL_MODEL.replace("rect: [10, 10, 20, 20]", "rect: [900, 10, 20, 20]")
    model, assets = _make_model(tmp_path, broken, name="market.yaml")
    code = main([model.path, "--assets", str(assets)])
    assert code == 1
    out = capsys.readouterr().out
    assert "rect_in_window" in out
    assert "tab_home" in out


def test_cli_exit_two_on_broken_file(tmp_path, capsys):
    path = tmp_path / "market.yaml"
    path.write_text("window: [разорвано\n", encoding="utf-8")
    assert main([str(path), "--assets", str(_assets_dir(tmp_path))]) == 2


# --- Allowlist пробелов отгруженных моделей (ruling R10) ---


def test_shipped_models_match_known_gaps():
    """Отгруженные модели: каждое нарушение записано в KNOWN_GAPS.yaml,
    каждая запись покрывает нарушение (лишние записи падают)."""
    gaps = load_gaps(KNOWN_GAPS)
    violations = []
    for window in ("market", "shop_buy", "shop_sell"):
        model = load_model(UI_MODEL / f"{window}.yaml")
        violations.extend(validate_model(model, ASSETS))

    result = check_against_gaps(violations, gaps)
    assert not result.unmatched  # нарушение вне allowlist
    assert not result.stale  # запись без нарушения


def test_check_against_gaps_flags_violation_outside_allowlist(tmp_path):
    violations = [_violation(rule="state_signatures", subject="s1")]
    result = check_against_gaps(violations, [])
    assert list(result.unmatched) == violations


def test_check_against_gaps_flags_stale_gap(tmp_path):
    gaps = load_gaps(_write_gaps(tmp_path, "market.yaml", "state_signatures", "s1"))
    result = check_against_gaps([], gaps)
    assert len(result.stale) == 1


def test_gap_with_wildcard_subject_covers_any_subject(tmp_path):
    gaps = load_gaps(_write_gaps(tmp_path, "market.yaml", "state_signatures", "*"))
    violations = [_violation(rule="state_signatures", subject="anything")]
    result = check_against_gaps(violations, gaps)
    assert not result.unmatched
    assert not result.stale


def test_gap_matches_only_same_file_and_rule(tmp_path):
    gaps = load_gaps(_write_gaps(tmp_path, "market.yaml", "state_signatures", "s1"))
    other_file = [_violation(rule="state_signatures", subject="s1", file="shop_buy.yaml")]
    assert list(check_against_gaps(other_file, gaps).unmatched) == other_file
    other_rule = [_violation(rule="rect_in_window", subject="s1", file="market.yaml")]
    assert list(check_against_gaps(other_rule, gaps).unmatched) == other_rule


# --- Вспомогательные функции ---


def _assets_dir(tmp_path):
    assets = tmp_path / "assets"
    (assets / "market").mkdir(parents=True, exist_ok=True)
    (assets / "market" / "tab_home_normal.png").write_bytes(b"png")
    return assets


def _make_model(tmp_path, text, *, name="market.yaml", templates=None):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    if templates is None:
        assets = _assets_dir(tmp_path)
    else:
        assets = tmp_path / "assets"
        assets.mkdir(exist_ok=True)
    return load_model(path), assets


def _violation(rule, subject, file="market.yaml"):
    from l2core.model.validator import Violation

    return Violation(rule=rule, file=file, subject=subject, detail="тест")


def _write_gaps(directory, file, rule, subject):
    path = directory / "gaps.yaml"
    path.write_text(
        yaml.safe_dump(
            {"gaps": [{"file": file, "rule": rule, "subject": subject}]},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path
