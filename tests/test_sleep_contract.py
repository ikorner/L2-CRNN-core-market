"""Guard-тест запрета sleep в коде ядра (задача 5.1, инвариант 1 AGENTS.md:
«Никаких sleep — только wait_until(predicate, timeout)»).

Механика: AST-проход по исходникам l2core/ ищет вызовы блокирующих ожиданий:
- attr-вызовы `.sleep(...)` (time.sleep, asyncio.sleep и любые другие);
- имя-вызовы `sleep(...)` (форма «from time import sleep»);
- `.wait(...)` на threading-примитивах (threading.Event()/Condition(),
  Event()/Condition()) — блокирующее ожидание вместо poll-предиката.

Assert: срабатываний нет вне allowlist. Allowlist (явный, с обоснованием):
- l2core/adapters/hid.py — сон у железа (перезагрузка Arduino при открытии
  порта), править вендоренный/железный слой нельзя;
- l2core/adapters/ocr_vendor/, l2core/adapters/table_segmentation/ —
  байт-идентичный вендор (задачи 4.1/4.3, правки запрещены);
- l2core/primitives/wait.py — единственная sanctioned-точка ожидания:
  сон там — вызов инъецированного `sleep` (дефолт time.sleep), механизм
  детерминизма тестов, а не обход инварианта.

AST, а не regex: вызовы распознаются синтаксисом — строки/комментарии с
текстом «time.sleep» ложных срабатываний не дают. Ссылка `time.sleep` как
значение (без вызова) — не нарушение: это механизм инъекции wait_until
(l2core/primitives/wait.py, дефолтный sleeper); guard ловит именно вызовы.

Сообщение об ошибке — файл:строка. Сканируется только l2core/: в tests/
sleep разрешён (инвариант 1 касается кода ядра).
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "l2core"

# Разрешённые пути (относительно scan root, т.е. без префикса «l2core/») —
# с обоснованием:
# - adapters/hid.py — сон у железа (перезагрузка Arduino при открытии порта),
#   править железный слой нельзя;
# - adapters/ocr_vendor/, adapters/table_segmentation/ — байт-идентичный
#   вендор (задачи 4.1/4.3, правки запрещены);
# - primitives/wait.py — единственная sanctioned-точка ожидания: вызовы
#   идут через инъецированный `sleep` (дефолт time.sleep), это механизм
#   детерминизма тестов, а не обход инварианта.
ALLOWED_FILES = frozenset({
    ("adapters", "hid.py"),
    ("primitives", "wait.py"),
})
ALLOWED_PREFIXES = (
    ("adapters", "ocr_vendor"),
    ("adapters", "table_segmentation"),
)


def _is_allowed(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    parts = tuple(rel.parts)
    return parts in ALLOWED_FILES or any(
        parts[: len(prefix)] == prefix for prefix in ALLOWED_PREFIXES
    )


def _call_segment(node: ast.Call, source: str) -> str:
    line = source.splitlines()[node.lineno - 1].strip()
    return line


def find_sleep_calls(roots) -> list[tuple[Path, int, str]]:
    """Все (файл, строка, текст вызова) блокирующих ожиданий вне allowlist."""
    hits = []
    for root in roots:
        root = Path(root)
        for path in sorted(p for p in root.rglob("*.py") if p.is_file()):
            if _is_allowed(path, root):
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                banned = False
                if isinstance(func, ast.Attribute) and func.attr == "sleep":
                    banned = True
                elif isinstance(func, ast.Name) and func.id == "sleep":
                    banned = True
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr == "wait"
                    and isinstance(func.value, ast.Call)
                    and (
                        (
                            isinstance(func.value.func, ast.Attribute)
                            and func.value.func.attr in ("Event", "Condition")
                        )
                        or (
                            isinstance(func.value.func, ast.Name)
                            and func.value.func.id in ("Event", "Condition")
                        )
                    )
                ):
                    banned = True
                if banned:
                    hits.append((path, node.lineno, _call_segment(node, source)))
    return sorted(hits)


def test_no_sleep_in_core():
    hits = find_sleep_calls([SCAN_ROOT])
    details = "\n".join(
        f"{path.relative_to(REPO_ROOT)}:{lineno}: {segment}"
        for path, lineno, segment in hits
    )
    assert not hits, (
        "блокирующие sleep/wait в коде ядра запрещены (инвариант 1: только "
        f"wait_until с poll-предикатом):\n{details}"
    )


def test_detector_catches_all_forms(tmp_path):
    # Сам детектор: фикстура-нарушитель — все формы вызова.
    pkg = tmp_path / "l2core" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "violator.py").write_text(
        "import time\n"
        "import asyncio\n"
        "import threading\n"
        "from time import sleep\n"
        "\n"
        "def f():\n"
        "    time.sleep(1.0)\n"
        "    asyncio.sleep(1.0)\n"
        "    sleep(1.0)\n"
        "    threading.Event().wait(1.0)\n"
        "    threading.Condition().wait(1.0)\n",
        encoding="utf-8",
    )
    hits = find_sleep_calls([tmp_path])
    assert [(h[0].name, h[1]) for h in hits] == [
        ("violator.py", 7),
        ("violator.py", 8),
        ("violator.py", 9),
        ("violator.py", 10),
        ("violator.py", 11),
    ]


def test_detector_respects_allowlist(tmp_path):
    # Файл из allowlist (та же относительная структура) — не срабатывает.
    hid = tmp_path / "l2core" / "adapters"
    hid.mkdir(parents=True)
    (hid / "hid.py").write_text(
        "import time\n\ndef f():\n    time.sleep(2.0)\n",
        encoding="utf-8",
    )
    vendor = tmp_path / "l2core" / "adapters" / "ocr_vendor"
    vendor.mkdir(parents=True)
    (vendor / "v.py").write_text(
        "import time\n\ndef f():\n    time.sleep(0.001)\n",
        encoding="utf-8",
    )
    wait_mod = tmp_path / "l2core" / "primitives"
    wait_mod.mkdir(parents=True)
    (wait_mod / "wait.py").write_text(
        "def wait_until(sleep):\n    sleep(0.2)\n",
        encoding="utf-8",
    )
    assert find_sleep_calls([tmp_path / "l2core"]) == []


def test_detector_ignores_mentions_and_references(tmp_path):
    # Строки/комментарии и ссылка без вызова (инъекция sleeper'а в wait.py)
    # — ложных срабатываний быть не должно (обоснование выбора AST).
    pkg = tmp_path / "l2core" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "ok.py").write_text(
        'HINT = "вместо time.sleep(1.0) используйте wait_until"\n'
        "import time\n"
        "DEFAULT_SLEEP = time.sleep  # ссылка, не вызов\n"
        "\n"
        "def g():\n"
        "    return time.monotonic()\n",
        encoding="utf-8",
    )
    assert find_sleep_calls([tmp_path]) == []
