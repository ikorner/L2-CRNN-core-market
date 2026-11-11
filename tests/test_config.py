"""Тесты загрузчика конфигурации l2core.config.

Покрывают: дефолты без файла, полный YAML, частичный YAML поверх дефолтов,
валидацию диапазонов (тайминги, доли, пороги) и обязательные пути.
"""

import pytest

from l2core.config import ConfigError, load_config

# Полный конфиг со всеми секциями (значения отличны от дефолтов, чтобы
# отличать «прочитано из файла» от «взято из дефолта»).
FULL_YAML = """
timing:
  wait_default_timeout: 20.0
  verify_interval: 0.5
  hover_pause: 0.8
  click_retry_count: 4

ocr:
  confidence_threshold: 0.7
  reread_count: 5

behavior:
  numpad_fraction: 0.3
  enter_fraction: 0.6
  click_jitter_px: 5
  move_jitter_px: 12
  press_delay_min_ms: 30
  press_delay_max_ms: 90
  key_delay_min_ms: 50
  key_delay_max_ms: 150
  move_step_delay_min_ms: 6
  move_step_delay_max_ms: 18
  drag_midpoints_min: 0
  drag_midpoints_max: 2

paths:
  model: /opt/models/crnn_best.pt
  arduino_port: COM7

kill_switch:
  hotkey: ctrl+alt+p
  inactivity_timeout: 600.0

resolution:
  width: 1600
  height: 900

dry_run: true
"""


def test_defaults_without_file():
    config = load_config(None)
    assert config.ocr.confidence_threshold == 0.85
    assert config.ocr.reread_count == 2
    assert config.resolution.width == 1920
    assert config.resolution.height == 1080
    assert config.paths.model == "retrain/crnn_retrained.pt"
    assert config.paths.arduino_port
    assert config.kill_switch.hotkey
    assert config.timing.wait_default_timeout > 0
    assert config.timing.verify_interval > 0
    assert config.timing.hover_pause > 0
    assert config.timing.click_retry_count >= 0
    assert config.timing.poll_interval > 0
    assert 0.0 <= config.behavior.numpad_fraction <= 1.0
    assert 0.0 <= config.behavior.enter_fraction <= 1.0
    assert config.behavior.press_delay_min_ms <= config.behavior.press_delay_max_ms
    assert config.behavior.key_delay_min_ms <= config.behavior.key_delay_max_ms
    assert config.behavior.move_step_delay_min_ms <= config.behavior.move_step_delay_max_ms
    assert config.behavior.drag_midpoints_min <= config.behavior.drag_midpoints_max
    assert config.kill_switch.inactivity_timeout > 0
    assert config.dry_run is False


def test_load_full_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(FULL_YAML, encoding="utf-8")
    config = load_config(cfg_path)

    assert config.timing.wait_default_timeout == 20.0
    assert config.timing.verify_interval == 0.5
    assert config.timing.hover_pause == 0.8
    assert config.timing.click_retry_count == 4
    assert config.ocr.confidence_threshold == 0.7
    assert config.ocr.reread_count == 5
    assert config.behavior.numpad_fraction == 0.3
    assert config.behavior.enter_fraction == 0.6
    assert config.behavior.click_jitter_px == 5
    assert config.behavior.move_jitter_px == 12
    assert config.behavior.press_delay_min_ms == 30
    assert config.behavior.press_delay_max_ms == 90
    assert config.behavior.key_delay_min_ms == 50
    assert config.behavior.key_delay_max_ms == 150
    assert config.behavior.move_step_delay_min_ms == 6
    assert config.behavior.move_step_delay_max_ms == 18
    assert config.behavior.drag_midpoints_min == 0
    assert config.behavior.drag_midpoints_max == 2
    assert config.paths.model == "/opt/models/crnn_best.pt"
    assert config.paths.arduino_port == "COM7"
    assert config.kill_switch.hotkey == "ctrl+alt+p"
    assert config.kill_switch.inactivity_timeout == 600.0
    assert config.resolution.width == 1600
    assert config.resolution.height == 900
    assert config.dry_run is True


def test_partial_yaml_overrides_defaults(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
ocr:
  confidence_threshold: 0.9
timing:
  wait_default_timeout: 3.0
""",
        encoding="utf-8",
    )
    config = load_config(cfg_path)

    # Переопределённые значения
    assert config.ocr.confidence_threshold == 0.9
    assert config.timing.wait_default_timeout == 3.0
    # Неупомянутые — из дефолтов
    assert config.ocr.reread_count == 2
    assert config.timing.verify_interval == 0.2
    assert config.resolution.width == 1920
    assert config.paths.model == "retrain/crnn_retrained.pt"


def test_empty_file_gives_defaults(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("", encoding="utf-8")
    config = load_config(cfg_path)
    assert config.ocr.confidence_threshold == 0.85
    assert config.resolution.width == 1920


@pytest.mark.parametrize(
    "yaml_text,path_to_none",
    [
        # Отрицательный таймаут wait_until
        ("timing:\n  wait_default_timeout: -1.0\n", "timing.wait_default_timeout"),
        ("timing:\n  verify_interval: -0.1\n", "timing.verify_interval"),
        ("timing:\n  hover_pause: -0.5\n", "timing.hover_pause"),
        ("timing:\n  poll_interval: 0.0\n", "timing.poll_interval"),
        # Доля вне [0, 1]
        ("behavior:\n  numpad_fraction: 1.5\n", "behavior.numpad_fraction"),
        ("behavior:\n  numpad_fraction: -0.1\n", "behavior.numpad_fraction"),
        ("behavior:\n  enter_fraction: 2.0\n", "behavior.enter_fraction"),
        # Порог confidence вне [0, 1]
        ("ocr:\n  confidence_threshold: 1.2\n", "ocr.confidence_threshold"),
        ("ocr:\n  confidence_threshold: -0.01\n", "ocr.confidence_threshold"),
        # Нарушен порядок min/max человеческих таймингов
        (
            "behavior:\n  press_delay_min_ms: 200\n  press_delay_max_ms: 100\n",
            "behavior.press_delay_",
        ),
        (
            "behavior:\n  key_delay_min_ms: 300\n  key_delay_max_ms: 50\n",
            "behavior.key_delay_",
        ),
        # Нарушен порядок min/max траектории drag
        (
            "behavior:\n  move_step_delay_min_ms: 40\n  move_step_delay_max_ms: 10\n",
            "behavior.move_step_delay_",
        ),
        (
            "behavior:\n  drag_midpoints_min: 5\n  drag_midpoints_max: 2\n",
            "behavior.drag_midpoints_",
        ),
        # Не-числовое значение там, где ждём число
        ("timing:\n  wait_default_timeout: abc\n", "timing.wait_default_timeout"),
        ("ocr:\n  reread_count: много\n", "ocr.reread_count"),
        # Отсутствует обязательный путь модели
        ("paths:\n  arduino_port: COM4\n", "paths.model"),
        ("paths:\n  model: ''\n", "paths.model"),
        ("paths:\n  model: null\n", "paths.model"),
        # Отсутствует порт arduino
        ("paths:\n  model: crnn_best.pt\n", "paths.arduino_port"),
        # Некорректное разрешение
        ("resolution:\n  width: 0\n", "resolution.width"),
        ("resolution:\n  height: -1080\n", "resolution.height"),
        # Пустой хоткей kill switch
        ("kill_switch:\n  hotkey: ''\n", "kill_switch.hotkey"),
        ("kill_switch:\n  inactivity_timeout: 0.0\n", "kill_switch.inactivity_timeout"),
        # Неизвестные ключи — опечатка молча игнорироваться не должна
        ("ocr_typo:\n  confidence_threshold: 0.9\n", None),
        ("ocr:\n  conf_threshold: 0.9\n", None),
    ],
)
def test_validation_errors(tmp_path, yaml_text, path_to_none):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(cfg_path)
    # Сообщение об ошибке называет проблемный ключ (если он одиночный)
    if path_to_none is not None:
        key = path_to_none.split(".")[-1] if "." in path_to_none else path_to_none
        assert key in str(excinfo.value) or path_to_none in str(excinfo.value)


def test_top_level_not_mapping(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_shipped_config_yaml_valid_and_matches_defaults():
    """Поставляемый в корне config.yaml валиден и не расходится с дефолтами."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    assert load_config(repo_root / "config.yaml") == load_config(None)
