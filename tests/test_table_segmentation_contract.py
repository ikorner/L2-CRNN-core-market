"""Guard запрета прямых импортов вендоренной сегментации таблицы
(задача 4.3, по аналогии с test_ocr_contract.py, ruling R13 для OCR).

Зачем: сырой вендор (build_dataset внутри пакета) несёт ДЕФОЛТНЫЕ константы
калибровки датасета 1136x833 — устаревшие для рантайма (ruling R5). Рантайм
обязан нарезать таблицу константами table_geometry модели — только через
wrapper (TableSegmentation из l2core.adapters.table_segmentation). Прямой
импорт вендора вне l2core/adapters/ — падение с именем файла и строки.

Механика — та же, что в test_ocr_contract: AST-проход, относительные
импорты разрешаются от пакета файла. Исключения зафиксированы явно
(ALLOWLIST): tests/test_table_segmentation_vendor.py — guard самого вендора;
tests/test_table_segmentation.py — тест wrapper'а, проверяющий подмену
module-level констант вендора (снапшот/возврат дефолтов) — без прямого
доступа к модулю не пишется.
"""
from pathlib import Path

from tests.test_ocr_contract import (
    ADAPTERS_DIR,
    REPO_ROOT,
    SCAN_ROOTS,
    _file_module,
    _iter_py_files,
    _resolve_from_target,
)
import ast

VENDOR_MODULE = "l2core.adapters.table_segmentation.build_dataset"
ALLOWLIST = frozenset(
    {
        (REPO_ROOT / "tests" / "test_table_segmentation_vendor.py").resolve(),
        (REPO_ROOT / "tests" / "test_table_segmentation.py").resolve(),
    }
)


def find_vendor_importers(roots) -> list[tuple[Path, int, str]]:
    """Все (файл, строка, таргет) с импортом модуля-вендора сегментации."""
    hits = []
    for root in roots:
        root = Path(root)
        base = root.resolve().parent
        for path in _iter_py_files(root):
            if ADAPTERS_DIR in path.resolve().parents:
                continue  # внутри l2core/adapters — свободно (wrapper)
            if path.resolve() in ALLOWLIST:
                continue  # guard-тест самого вендора (задача 4.3)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module, is_package = _file_module(path, base)
            for node in ast.walk(tree):
                targets = []
                relative = isinstance(node, ast.ImportFrom) and node.level > 0
                if isinstance(node, ast.Import):
                    targets = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    targets = _resolve_from_target(node, module, is_package)
                for target in targets:
                    matched = (
                        target == VENDOR_MODULE or target.startswith(VENDOR_MODULE + ".")
                        if not relative
                        else target.split(".")[-3:] == VENDOR_MODULE.split(".")[-3:]
                    )
                    if matched:
                        hits.append((path, node.lineno, target))
                        break
    return sorted(hits)


def test_no_table_segmentation_vendor_imports_outside_adapters():
    hits = find_vendor_importers(SCAN_ROOTS)
    details = "\n".join(
        f"{path.relative_to(REPO_ROOT)}:{lineno}: {target}"
        for path, lineno, target in hits
    )
    assert not hits, (
        "импорты вендоренного build_dataset (сегментация) вне "
        "l2core/adapters/ запрещены: рантайм нарезает таблицу только "
        f"константами table_geometry модели — через TableSegmentation:\n{details}"
    )


def test_detector_catches_import_forms(tmp_path):
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "violator.py").write_text(
        "import l2core.adapters.table_segmentation.build_dataset\n"
        "from l2core.adapters.table_segmentation import build_dataset\n"
        "from l2core.adapters.table_segmentation.build_dataset import extract_cells\n",
        encoding="utf-8",
    )
    (pkg / "rel_violator.py").write_text(
        "from ..l2core.adapters.table_segmentation.build_dataset import trim_to_text\n",
        encoding="utf-8",
    )
    hits = find_vendor_importers([tmp_path])
    by_file: dict[str, list[int]] = {}
    for path, lineno, _target in hits:
        by_file.setdefault(path.name, []).append(lineno)
    assert by_file["violator.py"] == [1, 2, 3]
    assert [lineno for lineno in by_file["rel_violator.py"]] == [1]
    assert len(hits) == 4
