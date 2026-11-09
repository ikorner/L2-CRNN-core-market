"""Guard-тест запрета прямых импортов ocr_vendor (задача 4.2b, ruling R13,
spec ocr-reading, требование 2: «OCR-вызовы вне контракта OcrClient
запрещены»).

Механика: AST-проход по исходникам l2core/ (кроме самого ocr_vendor/),
tests/ и tools/ собирает все импорты, чьим таргетом является пакет
l2core.adapters.ocr_vendor — абсолютные («import l2core.adapters.ocr_vendor»,
«from l2core.adapters.ocr_vendor import …», «from l2core.adapters import
ocr_vendor») и относительные (с разрешением уровня по пакету файла).
Assert: каждый такой файл лежит внутри l2core/adapters/.

AST, а не regex: импорты распознаются синтаксисом, а не подстрокой —
строки/комментарии с текстом «ocr_vendor» ложных срабатываний не дают,
а формы «from l2core.adapters import ocr_vendor» и относительные импорты
видны только через разбор дерева.

Единственное исключение — tests/test_ocr_vendor.py: это guard-тест самого
вендоренного кода (задача 4.1), он по определению обязан импортировать
ocr_vendor, чтобы сравнивать его с эталонами L2-CRNN-MODEL-TRAIN.
Исключение зафиксировано явно (frozenset ALLOWLIST); новые потребители
вне l2core/adapters/ запрещены — guard упадёт с именем файла и строки.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_PACKAGE = "l2core.adapters.ocr_vendor"
ADAPTERS_DIR = REPO_ROOT / "l2core" / "adapters"
SCAN_ROOTS = (REPO_ROOT / "l2core", REPO_ROOT / "tests", REPO_ROOT / "tools")
ALLOWLIST = frozenset({(REPO_ROOT / "tests" / "test_ocr_vendor.py").resolve()})


def _iter_py_files(root: Path):
    yield from sorted(p for p in root.rglob("*.py") if p.is_file())


def _file_module(path: Path, base: Path) -> tuple[str, bool]:
    """Dotted-модуль файла и флаг «файл — __init__.py пакета».

    base — путь-родитель scan root (для l2core/tests/tools это REPO_ROOT,
    для фикстур tmp_path — родитель tmp_path): относительные импорты
    разрешаются лексически от пакета файла внутри сканируемого дерева.
    """
    try:
        rel = path.resolve().relative_to(base.resolve())
    except ValueError:
        return "", False
    parts = list(rel.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return ".".join(parts), is_package


def _resolve_from_target(node: ast.ImportFrom, module: str, is_package: bool) -> list[str]:
    """Кандидатные dotted-таргеты from-импорта (base и base + каждый alias).

    Относительный импорт разрешается от пакета файла: уровень level=1 —
    пакет, содержащий файл; каждый следующий уровень поднимается на пакет
    вверх. Для __init__.py пакет — сам его модуль.
    """
    if node.level == 0:
        base_parts = node.module.split(".") if node.module else []
    else:
        if not module:
            return []  # пакет файла неизвестен — относительный не разрешаем
        package = module if is_package else module.rsplit(".", 1)[0]
        package_parts = package.split(".") if package else []
        keep = len(package_parts) - (node.level - 1)
        if keep < 0:
            return []  # уровень выше корня — не наш пакет
        base_parts = package_parts[:keep]
        if node.module:
            base_parts += node.module.split(".")
    base = ".".join(base_parts)
    targets = [base] if base else []
    targets += [f"{base}.{alias.name}" for alias in node.names] if base else []
    return targets


def _matches_vendor(target: str, relative: bool) -> bool:
    """Таргет — импорт вендор-пакета.

    Абсолютный: точное совпадение или подмодель (l2core.adapters.ocr_vendor[.x]).
    Относительный: разрешённый путь заканчивается на пакет вендора (суффикс) —
    префикс у него пакетная иерархия от scan root.
    """
    if not relative:
        return target == VENDOR_PACKAGE or target.startswith(VENDOR_PACKAGE + ".")
    parts = target.split(".")
    vendor_parts = VENDOR_PACKAGE.split(".")
    return len(parts) >= len(vendor_parts) and parts[-len(vendor_parts):] == vendor_parts


def find_vendor_importers(roots) -> list[tuple[Path, int, str]]:
    """Все (файл, строка, таргет) с импортом l2core.adapters.ocr_vendor.

    По одному срабатыванию на узел импорта (первый совпавший таргет).
    """
    hits = []
    for root in roots:
        root = Path(root)
        base = root.resolve().parent  # путь-родитель scan root
        for path in _iter_py_files(root):
            if ADAPTERS_DIR in path.resolve().parents:
                continue  # внутри l2core/adapters — свободно (ocr.py, ocr_vendor)
            if path.resolve() in ALLOWLIST:
                continue  # guard-тест самого вендора (задача 4.1)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module, is_package = _file_module(path, base)
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Import):
                    targets = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    targets = _resolve_from_target(node, module, is_package)
                relative = isinstance(node, ast.ImportFrom) and node.level > 0
                for target in targets:
                    if _matches_vendor(target, relative):
                        hits.append((path, node.lineno, target))
                        break
    return sorted(hits)


def test_no_ocr_vendor_imports_outside_adapters():
    hits = find_vendor_importers(SCAN_ROOTS)
    details = "\n".join(
        f"{path.relative_to(REPO_ROOT)}:{lineno}: {target}"
        for path, lineno, target in hits
    )
    assert not hits, (
        "импорты l2core.adapters.ocr_vendor вне l2core/adapters/ запрещены "
        "(spec ocr-reading требование 2, ruling R13) — потребители OCR работают "
        f"через контракт OcrClient:\n{details}"
    )


def test_detector_catches_absolute_and_from_and_relative(tmp_path):
    # Сам детектор: фикстура-нарушитель во временной директории — AST-проход
    # видит все формы импорта (absolute, from-import submodule, relative).
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "violator.py").write_text(
        "import l2core.adapters.ocr_vendor\n"
        "from l2core.adapters import ocr_vendor\n"
        "from l2core.adapters.ocr_vendor import load_model\n",
        encoding="utf-8",
    )
    (pkg / "rel_violator.py").write_text(
        "from ..l2core.adapters.ocr_vendor import preprocess\n",
        encoding="utf-8",
    )
    hits = find_vendor_importers([tmp_path])
    by_file: dict[str, list[tuple[int, str]]] = {}
    for path, lineno, target in hits:
        by_file.setdefault(path.name, []).append((lineno, target))
    assert by_file["violator.py"] == [
        (1, "l2core.adapters.ocr_vendor"),
        (2, "l2core.adapters.ocr_vendor"),
        (3, "l2core.adapters.ocr_vendor"),
    ]
    rel_hits = by_file["rel_violator.py"]
    assert len(rel_hits) == 1
    rel_lineno, rel_target = rel_hits[0]
    assert rel_lineno == 1
    assert rel_target.endswith(".l2core.adapters.ocr_vendor")  # разрешённый путь
    assert len(hits) == 4  # 3 формы в violator + 1 относительный


def test_detector_ignores_non_import_mentions(tmp_path):
    # Строки/комментарии с текстом «ocr_vendor» — не импорты, ложных
    # срабатываний быть не должно (обоснование выбора AST вместо regex).
    f = tmp_path / "mentions.py"
    f.write_text(
        'VENDOR = "l2core.adapters.ocr_vendor"\n'
        "X = 'from l2core.adapters.ocr_vendor import load_model'\n",
        encoding="utf-8",
    )
    assert find_vendor_importers([tmp_path]) == []
