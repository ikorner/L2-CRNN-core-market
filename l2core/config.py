"""Схема конфигурации ядра и типизированный загрузчик.

Единственная точка, через которое ядро получает тайминги, пороги OCR,
поведенческий профиль, пути и настройки kill switch.

Структура зеркалит config.yaml (см. design.md, решение 20):
- timing — тайминги/таймауты для wait_until и повторов действий;
- ocr — порог confidence и число перечитываний;
- behavior — поведенческий профиль (доли способов ввода, джиттеры,
  человеческие тайминги HID);
- paths — пути (чекпоинт CRNN, порт arduino);
- kill_switch — хоткей и глобальный таймаут неактивности;
- resolution — референсное разрешение клиентской области;
- window — критерии поиска окна игры (заголовок/класс, задача 3.2);
- dry_run — режим без HID (действия только логируются).

Частичный YAML накладывается поверх дефолтов (dataclass-поля);
неизвестные ключи и значения вне диапазонов отклоняются с ConfigError.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

import yaml


class ConfigError(Exception):
    """Некорректный config.yaml: валидация или структура."""


# --- Секции конфигурации (значения по умолчанию — зеркало config.yaml) ---


@dataclass(frozen=True)
class TimingConfig:
    """Тайминги/таймауты (секунды, если не сказано иное)."""

    wait_default_timeout: float = 10.0  # таймаут wait_until по умолчанию
    verify_interval: float = 0.2  # интервал опроса predicate в wait_until
    hover_pause: float = 0.4  # пауза после hover до verify (тултип/подсветка)
    click_retry_count: int = 2  # число повторов действия при неподтверждении
    poll_interval: float = 1.0  # пауза между циклами state observer (7.2)


@dataclass(frozen=True)
class OcrConfig:
    confidence_threshold: float = 0.85  # порог confidence OCR, [0, 1]
    reread_count: int = 2  # число перечитываний при низком conf/несовпадении


@dataclass(frozen=True)
class BehaviorConfig:
    """Поведенческий профиль (антидетект): распределения и человеческие тайминги."""

    numpad_fraction: float = 0.5  # доля ввода цены экранным нумпадом, [0, 1]
    enter_fraction: float = 0.5  # доля Enter против кнопки «Поиск», [0, 1]
    click_jitter_px: int = 3  # джиттер точки клика, пиксели
    move_jitter_px: int = 8  # джиттер траектории движения курсора, пиксели
    press_delay_min_ms: int = 40  # задержка нажатия кнопки мыши, мин
    press_delay_max_ms: int = 120  # ... макс
    key_delay_min_ms: int = 60  # пауза между клавишами, мин
    key_delay_max_ms: int = 180  # ... макс
    move_step_delay_min_ms: int = 8  # пауза между шагами траектории drag, мин
    move_step_delay_max_ms: int = 24  # ... макс
    drag_midpoints_min: int = 1  # промежуточные точки траектории drag, мин
    drag_midpoints_max: int = 3  # ... макс


@dataclass(frozen=True)
class PathsConfig:
    model: str = "crnn_best.pt"  # чекпоинт CRNN (файл вне git)
    arduino_port: str = "COM3"  # порт HID-адаптера arduino


@dataclass(frozen=True)
class KillSwitchConfig:
    hotkey: str = "ctrl+shift+k"  # локальный хоткей мгновенной остановки
    inactivity_timeout: float = 300.0  # глобальный таймаут неактивности, сек


@dataclass(frozen=True)
class ResolutionConfig:
    width: int = 1920  # референсное разрешение клиентской области
    height: int = 1080


@dataclass(frozen=True)
class WindowConfig:
    """Критерии поиска окна игры (задача 3.2)."""

    title: str = "Lineage II"  # точный заголовок окна игры (уточняется у владельца)
    class_name: str | None = None  # опциональный класс окна для фильтрации


@dataclass(frozen=True)
class CoreConfig:
    """Корневой конфиг ядра."""

    timing: TimingConfig = field(default_factory=TimingConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    dry_run: bool = False  # действия только логируются, без HID


# --- Вспомогательные проверки значений ---


def _fail(key: str, value: Any, why: str) -> ConfigError:
    return ConfigError(f"{key}: {why} (получено {value!r})")


def _as_number(key: str, value: Any) -> float:
    """Число из YAML; bool и не-числа отклоняются."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(key, value, "ожидается число")
    return float(value)


def _positive_float(key: str, value: Any) -> float:
    number = _as_number(key, value)
    if number <= 0:
        raise _fail(key, value, "ожидается положительное число")
    return number


def _nonnegative_int(key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _fail(key, value, "ожидается целое число >= 0")
    return value


def _positive_int(key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail(key, value, "ожидается целое число > 0")
    return value


def _fraction(key: str, value: Any) -> float:
    number = _as_number(key, value)
    if not 0.0 <= number <= 1.0:
        raise _fail(key, value, "ожидается число в [0, 1]")
    return number


def _nonempty_str(key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(key, value, "ожидается непустая строка")
    return value


# --- Загрузка и валидация ---

Validator = Callable[[str, Any], Any]

_SECTION_FIELDS = {
    name: frozenset(cls.__dataclass_fields__)
    for name, cls in (
        ("timing", TimingConfig),
        ("ocr", OcrConfig),
        ("behavior", BehaviorConfig),
        ("paths", PathsConfig),
        ("kill_switch", KillSwitchConfig),
        ("resolution", ResolutionConfig),
        ("window", WindowConfig),
    )
}


def _section(raw: Any, name: str) -> dict[str, Any]:
    """Секция как dict; неизвестные ключи внутри секции отклоняются."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _fail(name, raw, "ожидается mapping ключ: значение")
    unknown = set(raw) - _SECTION_FIELDS[name]
    if unknown:
        raise ConfigError(f"{name}: неизвестные ключи {sorted(unknown)}")
    return dict(raw)


def _build_section(
    name: str, cls: type, raw: dict[str, Any], validators: dict[str, Validator]
) -> Any:
    """Собрать секцию: дефолты из полей dataclass, заданные в YAML — валидируются."""
    defaults = cls()
    kwargs = {}
    for f in fields(cls):
        if f.name in raw:
            kwargs[f.name] = validators[f.name](f"{name}.{f.name}", raw[f.name])
        else:
            kwargs[f.name] = getattr(defaults, f.name)
    return cls(**kwargs)


def _check_min_max(cfg: BehaviorConfig) -> None:
    """Пары min/max человеческих таймингов и траектории: min не больше max."""
    for lo, hi in (
        ("press_delay", "нажатия"),
        ("key_delay", "между клавишами"),
        ("move_step_delay", "между шагами траектории"),
    ):
        low = getattr(cfg, f"{lo}_min_ms")
        high = getattr(cfg, f"{lo}_max_ms")
        if low > high:
            raise _fail(
                f"behavior.{lo}_max_ms",
                high,
                f"не меньше behavior.{lo}_min_ms ({low}, задержка {hi})",
            )
    if cfg.drag_midpoints_min > cfg.drag_midpoints_max:
        raise _fail(
            "behavior.drag_midpoints_max",
            cfg.drag_midpoints_max,
            "не меньше behavior.drag_midpoints_min "
            f"({cfg.drag_midpoints_min}, число промежуточных точек)",
        )


def _build_paths(raw: dict[str, Any]) -> PathsConfig:
    # Секция не задана — дефолты. Задана — перекрывается целиком: пропуск
    # пути в ней ошибка, а не тихий дефолт; заданные значения валидируются
    # раньше проверки обязательности (пустой model важнее незаданного порта).
    if not raw:
        return PathsConfig()
    values = {
        key: _nonempty_str(f"paths.{key}", value) for key, value in raw.items()
    }
    for key in ("model", "arduino_port"):
        if key not in values:
            raise ConfigError(f"paths.{key}: обязательный параметр не задан")
    return PathsConfig(**values)


def load_config(path: str | Path | None = None) -> CoreConfig:
    """Загрузить конфиг.

    path=None — дефолты без файла. Иначе читается YAML; неизвестные ключи,
    неверные типы и значения вне диапазонов → ConfigError. Отсутствующий
    файл → FileNotFoundError.
    """
    if path is None:
        return CoreConfig()

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return CoreConfig()
    if not isinstance(data, dict):
        raise _fail("config.yaml", data, "ожидается mapping секций")

    unknown_top = set(data) - (set(_SECTION_FIELDS) | {"dry_run"})
    if unknown_top:
        raise ConfigError(f"config.yaml: неизвестные секции {sorted(unknown_top)}")

    timing = _build_section(
        "timing",
        TimingConfig,
        _section(data.get("timing"), "timing"),
        {
            "wait_default_timeout": _positive_float,
            "verify_interval": _positive_float,
            "hover_pause": _positive_float,
            "poll_interval": _positive_float,
            "click_retry_count": _nonnegative_int,
        },
    )
    ocr = _build_section(
        "ocr",
        OcrConfig,
        _section(data.get("ocr"), "ocr"),
        {"confidence_threshold": _fraction, "reread_count": _nonnegative_int},
    )
    behavior = _build_section(
        "behavior",
        BehaviorConfig,
        _section(data.get("behavior"), "behavior"),
        {
            "numpad_fraction": _fraction,
            "enter_fraction": _fraction,
            "click_jitter_px": _nonnegative_int,
            "move_jitter_px": _nonnegative_int,
            "press_delay_min_ms": _nonnegative_int,
            "press_delay_max_ms": _nonnegative_int,
            "key_delay_min_ms": _nonnegative_int,
            "key_delay_max_ms": _nonnegative_int,
            "move_step_delay_min_ms": _nonnegative_int,
            "move_step_delay_max_ms": _nonnegative_int,
            "drag_midpoints_min": _nonnegative_int,
            "drag_midpoints_max": _nonnegative_int,
        },
    )
    _check_min_max(behavior)
    kill_switch = _build_section(
        "kill_switch",
        KillSwitchConfig,
        _section(data.get("kill_switch"), "kill_switch"),
        {"hotkey": _nonempty_str, "inactivity_timeout": _positive_float},
    )
    resolution = _build_section(
        "resolution",
        ResolutionConfig,
        _section(data.get("resolution"), "resolution"),
        {"width": _positive_int, "height": _positive_int},
    )
    window = _build_section(
        "window",
        WindowConfig,
        _section(data.get("window"), "window"),
        {
            "title": _nonempty_str,
            "class_name": lambda key, value: None
            if value is None
            else _nonempty_str(key, value),
        },
    )
    dry_run = data.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise _fail("dry_run", dry_run, "ожидается true/false")

    return CoreConfig(
        timing=timing,
        ocr=ocr,
        behavior=behavior,
        paths=_build_paths(_section(data.get("paths"), "paths")),
        kill_switch=kill_switch,
        resolution=resolution,
        window=window,
        dry_run=dry_run,
    )
