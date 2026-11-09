"""Тесты конвертера meta.json → UI-модель (l2core.model.converter).

Покрывают: импорт кропов в assets/ (включая идемпотентность повторного
прогона), сбор карточек элементов и состояний, diff-отчёт и политику
слияния «ручное побеждает, машинное дополняет», пропуск ролей-сырья meta
(decor/ocr_ground_truth) и кропов с неизвестным вариантом, ветки
MetaFormatError сырья, замечания (issues) о неоднозначностях импорта и
круглый путь: сгенерированная модель распарсивается существующим
load_model (задача 9.1: покрытие доведено до всех веток).
"""

import json
from pathlib import Path

import pytest
import yaml

from l2core.model import load_common, load_model
from l2core.model.converter import MetaFormatError, convert

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_MODEL = REPO_ROOT / "ui_model"

# Минимальное сырьё, имитирующее base_screenshot/scr/: два скриншота окна
# market (одно состояние market.list + один скриншот без состояния с
# HUD-иконкой), кропы-файлы в crops/obzor_lavok/.
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-crop-bytes"


def _crop(crop_id, file, screenshot, element, rect, roles, variant="normal"):
    return {
        "file": file,
        "screenshot": screenshot,
        "element": element,
        "window": "market",
        "slot": None,
        "variant": variant,
        "roles": roles,
        "rect": rect,
        "description": "",
        "click_point": None,
        "text_gt": None,
        "links": {"part_of": None, "click_leads_to": None},
        "notes": "",
    }


def make_scr(tmp_path: Path) -> Path:
    """Собрать каталог сырья meta.json + crops/ во временной директории."""
    scr = tmp_path / "scr"
    (scr / "crops" / "obzor_lavok").mkdir(parents=True)
    for name, payload in {
        "s1_win.png": PNG_BYTES,
        "s1_tab.png": PNG_BYTES + b"-tab",
        "s2_pic.png": PNG_BYTES + b"-pic",
    }.items():
        (scr / "crops" / "obzor_lavok" / name).write_bytes(payload)

    meta = {
        "version": 1,
        "dicts": {"windows": {}, "states": {}, "elements": {}},
        "screenshots": {
            "s1": {
                "file": "s1.png",
                "window": "market",
                "state": "market.list",
                "state_params": {"tab": "home"},
                "resolution": [1920, 1080],
                "description": "",
                "crops": ["s1_win", "s1_tab"],
            },
            "s2": {
                "file": "s2.png",
                "window": "market",
                "state": None,
                "state_params": {},
                "resolution": [1920, 1080],
                "description": "",
                "crops": ["s2_pic"],
            },
        },
        "crops": {
            # Кроп-пример всего окна: самый большой -> он же якорь окна.
            "s1_win": _crop(
                "s1_win",
                "crops/obzor_lavok/s1_win.png",
                "s1",
                "obzor_lavok.win",
                [100, 100, 400, 300],
                ["icon_sample"],
            ),
            "s1_tab": _crop(
                "s1_tab",
                "crops/obzor_lavok/s1_tab.png",
                "s1",
                "obzor_lavok.tab",
                [110, 150, 50, 20],
                ["click_target"],
                variant="active",
            ),
            # Не вписывается в прямоугольник окна -> HUD-элемент.
            "s2_pic": _crop(
                "s2_pic",
                "crops/obzor_lavok/s2_pic.png",
                "s2",
                "obzor_lavok.pic",
                [900, 900, 30, 30],
                ["icon_sample"],
            ),
        },
        "conventions": "",
    }
    (scr / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scr


@pytest.fixture()
def workspace(tmp_path):
    scr = make_scr(tmp_path)
    model_dir = tmp_path / "ui_model"
    assets = tmp_path / "assets"
    return scr, model_dir, assets


def test_import_copies_crops_to_assets(workspace):
    scr, model_dir, assets = workspace
    report = convert(scr / "meta.json", scr, model_dir, assets)

    assert (assets / "market" / "s1_win.png").read_bytes() == PNG_BYTES
    assert (assets / "market" / "s1_tab.png").read_bytes() == PNG_BYTES + b"-tab"
    assert (assets / "market" / "s2_pic.png").read_bytes() == PNG_BYTES + b"-pic"
    assert sorted(report.assets_added) == [
        "market/s1_tab.png",
        "market/s1_win.png",
        "market/s2_pic.png",
    ]


def test_generated_model_parses_and_is_anchored(workspace):
    scr, model_dir, assets = workspace
    convert(scr / "meta.json", scr, model_dir, assets)

    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(model_dir / "market.yaml", common)

    # window: размер взят от кропа-примера всего окна.
    assert model.window.id == "market"
    assert model.window.size == (400, 300)

    # Обычный элемент: rect от якоря (левый верх окна = 100, 100).
    tab = model.elements["tab"]
    assert tab.rect == (10, 50, 50, 20)
    assert tab.role == "click_target"
    assert tab.variants["active"].template == "market/s1_tab.png"

    # Кроп, не вписавшийся в окно, — HUD-элемент с search_area.
    pic = model.elements["pic"]
    assert pic.hud
    assert pic.search_area == (900, 900, 30, 30)
    assert pic.rect is None

    # Состояние собрано из meta: id, params и template-сигнатуры по факту
    # наблюдения элементов на скриншоте состояния.
    state = model.states["list__tab_home"]
    assert state.params == {"tab": "home"}
    kinds = {(c.kind, c.element, c.variant) for c in state.signatures}
    assert ("template", "tab", "active") in kinds


def test_second_run_is_idempotent(workspace):
    scr, model_dir, assets = workspace
    convert(scr / "meta.json", scr, model_dir, assets)
    first_model = (model_dir / "market.yaml").read_bytes()
    first_report_render = convert(
        scr / "meta.json", scr, model_dir, assets
    ).render()

    # Повторный прогон не добавляет файлы и не переписывает модель.
    assert (model_dir / "market.yaml").read_bytes() == first_model
    assert "добавлено 0," in first_report_render


def test_manual_edits_win_over_machine(workspace):
    scr, model_dir, assets = workspace
    model_dir.mkdir()
    manual = {
        "window": {"id": "market", "title": "Обзор лавок", "size": [400, 300]},
        "elements": {
            # Ручная правка того же элемента, что привезёт машина: rect
            # отличается от машинного (10, 50, 50, 20).
            "tab": {
                "rect": [77, 77, 50, 20],
                "role": "click_target",
                "variants": {
                    "active": {"template": "market/hand_tab.png"},
                },
            },
            # Элемент, которого у машины нет, — сохраняется.
            "hand_only": {
                "rect": [1, 2, 3, 4],
                "role": "icon_sample",
            },
        },
        "states": {
            "list_home": {
                "params": {"tab": "home"},
                "signatures": [
                    {"kind": "ocr", "region": "title", "expect": "Обзор лавок"}
                ],
            }
        },
        "transitions": [],
        "ocr_regions": {"title": {"rect": [0, 0, 100, 20], "grammar": "text"}},
    }
    (model_dir / "market.yaml").write_text(
        yaml.safe_dump(manual, allow_unicode=True), encoding="utf-8"
    )

    report = convert(scr / "meta.json", scr, model_dir, assets)

    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(model_dir / "market.yaml", common)

    # Ручной элемент не перезаписан машиной.
    assert model.elements["tab"].rect == (77, 77, 50, 20)
    assert model.elements["tab"].variants["active"].template == "market/hand_tab.png"
    assert "hand_only" in model.elements
    # Ручное состояние с тем же смыслом, но другим id сохранено;
    # машинное добавлено рядом.
    assert "list_home" in model.states
    assert "list__tab_home" in model.states
    # О конфликте «tab» сообщено в отчёте.
    assert "tab" in report.elements_conflicts["market"]


def test_asset_conflict_is_not_overwritten(workspace):
    scr, model_dir, assets = workspace
    convert(scr / "meta.json", scr, model_dir, assets)

    # Портим целевой файл: повторный прогон не должен его перезаписать.
    victim = assets / "market" / "s1_win.png"
    victim.write_bytes(b"tampered")
    report = convert(scr / "meta.json", scr, model_dir, assets)

    assert victim.read_bytes() == b"tampered"
    assert "market/s1_win.png" in report.assets_conflicts


def test_raw_meta_roles_are_not_imported(workspace):
    scr, model_dir, assets = workspace
    meta_path = scr / "meta.json"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    meta["screenshots"]["s1"]["crops"].append("s1_decor")
    meta["crops"]["s1_decor"] = _crop(
        "s1_decor",
        "crops/obzor_lavok/s1_decor.png",
        "s1",
        "obzor_lavok.decor_line",
        [120, 200, 10, 17],
        ["decor"],
    )
    (scr / "crops" / "obzor_lavok" / "s1_decor.png").write_bytes(PNG_BYTES)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = convert(meta_path, scr, model_dir, assets)

    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(model_dir / "market.yaml", common)
    # Роль-сырьё meta в модель ядра не переносится: элемента нет,
    # кроп не скопирован, пропуск зафиксирован в отчёте.
    assert "decor_line" not in model.elements
    assert not (assets / "market" / "s1_decor.png").exists()
    assert any("decor_line" in issue for issue in report.issues)


def test_hud_element_keeps_own_role():
    """HUD-классификация не должна затирать роль элемента ролью последнего
    по сортировке кропа окна (латентный дефект: `role` протухал между
    циклами)."""
    import json
    import tempfile

    # Собираем свой workspace: кроп HUD с icon_sample идёт в сортировке
    # раньше кропа click_target внутри окна — последний обработанный
    # кроп окна имеет click_target.
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        scr = make_scr(root)
        meta_path = scr / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["screenshots"]["s3"] = {
            "file": "s3.png",
            "window": "market",
            "state": None,
            "state_params": {},
            "resolution": [1920, 1080],
            "description": "",
            "crops": ["s9_last"],
        }
        meta["crops"]["s9_last"] = _crop(
            "s9_last",
            "crops/obzor_lavok/s9_last.png",
            "s3",
            "obzor_lavok.last_tab",
            [120, 200, 30, 30],  # внутри окна [100, 100, 400, 300]
            ["click_target"],
        )
        (scr / "crops" / "obzor_lavok" / "s9_last.png").write_bytes(PNG_BYTES)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        model_dir = root / "ui_model"
        assets = root / "assets"
        convert(meta_path, scr, model_dir, assets)

        common = load_common(UI_MODEL / "common.yaml")
        model = load_model(model_dir / "market.yaml", common)
        # s2_pic (icon_sample) — HUD и сохраняет СВОЮ роль, а не роль
        # последнего по сортировке кропа окна (click_target).
        assert model.elements["pic"].hud
        assert model.elements["pic"].role == "icon_sample"


def test_manual_wins_reports_state_conflict(workspace):
    """Состояние с тем же id, но расходящимся содержимым: ручное побеждает
    и конфликт фиксируется в diff-отчёте (по аналогии с элементами)."""
    scr, model_dir, assets = workspace
    model_dir.mkdir()
    manual = {
        "window": {"id": "market", "title": "Обзор лавок", "size": [1008, 674]},
        "elements": {
            "tab": {"rect": [77, 77, 50, 20], "role": "click_target"},
        },
        "states": {
            # То же id, что у машинного list__tab_home, но другое содержимое.
            "list__tab_home": {
                "params": {"tab": "home"},
                "signatures": [
                    {"kind": "ocr", "region": "title", "expect": "Обзор лавок"}
                ],
            }
        },
        "transitions": [],
        "ocr_regions": {},
    }
    # Мета даёт окну market: заменяем rect якоря на большой, чтобы
    # совпала window-секция и не мешал issue о расхождении window.
    import json

    meta_path = scr / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["crops"]["s1_win"]["rect"] = [0, 0, 1008, 674]
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (model_dir / "market.yaml").write_text(
        yaml.safe_dump(manual, allow_unicode=True), encoding="utf-8"
    )

    report = convert(meta_path, scr, model_dir, assets)

    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(model_dir / "market.yaml", common)

    # Ручное содержимое состояния на месте.
    assert [s.kind for s in model.states["list__tab_home"].signatures] == ["ocr"]
    # Конфликт зафиксирован — неотличимым от полного совпадения не остался.
    assert "list__tab_home" in report.states_conflicts["market"]
    assert "list__tab_home" not in report.states_kept.get("market", [])


def test_shipped_models_generated_by_converter_parse():
    """Поставляемые модели (результат импорта настоящей меты) парсятся."""
    common = load_common(UI_MODEL / "common.yaml")
    for window in ("market", "shop_buy", "shop_sell"):
        model = load_model(UI_MODEL / f"{window}.yaml", common)
        assert model.window.id == window


# --- Задача 9.1: аудит покрытия — ветки MetaFormatError и diff-отчёта ---


def _rewrite_meta(scr, mutate):
    """Прочитать meta.json в workspace, применить mutate(meta) и записать."""
    meta_path = scr / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mutate(meta)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta_path


def test_broken_meta_json_raises(workspace):
    scr, model_dir, assets = workspace
    (scr / "meta.json").write_text("{ разорвано", encoding="utf-8")

    with pytest.raises(MetaFormatError, match="не читается"):
        convert(scr / "meta.json", scr, model_dir, assets)


@pytest.mark.parametrize("section", ["screenshots", "crops"])
def test_meta_missing_sections_raise(workspace, section):
    scr, model_dir, assets = workspace
    meta_path = _rewrite_meta(scr, lambda meta: meta.pop(section))

    with pytest.raises(MetaFormatError, match="отсутствует секция"):
        convert(meta_path, scr, model_dir, assets)


def test_element_without_window_prefix_raises(workspace):
    scr, model_dir, assets = workspace
    meta_path = _rewrite_meta(
        scr, lambda meta: meta["crops"]["s1_tab"].__setitem__("element", "lonely")
    )

    with pytest.raises(MetaFormatError, match="префикса окна"):
        convert(meta_path, scr, model_dir, assets)


def test_crop_with_broken_rect_raises(workspace):
    scr, model_dir, assets = workspace
    meta_path = _rewrite_meta(
        scr, lambda meta: meta["crops"]["s1_tab"].__setitem__("rect", [1, 2, 3])
    )

    with pytest.raises(MetaFormatError, match="битый rect"):
        convert(meta_path, scr, model_dir, assets)


def test_element_id_sanitized_when_not_alpha(workspace):
    """id элемента, не начинающийся с буквы, санитизируется префиксом el_."""
    scr, model_dir, assets = workspace
    meta_path = _rewrite_meta(
        scr, lambda meta: meta["crops"]["s1_tab"].__setitem__("element", "obzor_lavok.9tab")
    )
    convert(meta_path, scr, model_dir, assets)

    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(model_dir / "market.yaml", common)
    assert "el_9tab" in model.elements


def test_unknown_variant_crop_skipped_and_reported(workspace):
    """Кроп с вариантом вне словаря модели не импортируется; пропуск — в
    issues отчёта (честная деградация, не исключение)."""
    scr, model_dir, assets = workspace
    meta_path = _rewrite_meta(
        scr, lambda meta: meta["crops"]["s1_tab"].__setitem__("variant", "weird")
    )
    report = convert(meta_path, scr, model_dir, assets)

    assert not (assets / "market" / "s1_tab.png").exists()
    assert any("weird" in issue for issue in report.issues)
    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(model_dir / "market.yaml", common)
    assert "tab" not in model.elements


def test_second_observation_rect_discrepancy_issue(workspace):
    """Второй кроп того же элемента с другим rect: в отчёте — замечание,
    в модели — rect от первого наблюдения (детерминизм импорта)."""
    scr, model_dir, assets = workspace

    def mutate(meta):
        meta["screenshots"]["s1"]["crops"].append("s1_tab2")
        meta["crops"]["s1_tab2"] = _crop(
            "s1_tab2",
            "crops/obzor_lavok/s1_tab2.png",
            "s1",
            "obzor_lavok.tab",
            [120, 160, 50, 20],  # первое наблюдение: [110, 150, 50, 20]
            ["click_target"],
            variant="hover",
        )

    meta_path = _rewrite_meta(scr, mutate)
    (scr / "crops" / "obzor_lavok" / "s1_tab2.png").write_bytes(PNG_BYTES)
    report = convert(meta_path, scr, model_dir, assets)

    assert any("отличается от первого наблюдения" in i for i in report.issues)
    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(model_dir / "market.yaml", common)
    assert model.elements["tab"].rect == (10, 50, 50, 20)


def test_fewer_than_two_signatures_issue(workspace):
    """Мета даёт меньше двух template-сигнатур состояния (активный вариант
    один) — досигнатура вручную фиксируется в issues (правило 2.3)."""
    scr, model_dir, assets = workspace
    report = convert(scr / "meta.json", scr, model_dir, assets)

    assert any("меньше двух template-сигнатур" in i for i in report.issues)


def test_window_section_mismatch_reported(workspace):
    """Ручная window-секция, расходящаяся с машинной (size/title), не
    перезаписывается; расхождение — в issues отчёта."""
    scr, model_dir, assets = workspace
    model_dir.mkdir()
    manual = {
        "window": {"id": "market", "title": "Обзор лавок", "size": [500, 300]},
        "elements": {},
        "states": {},
        "transitions": [],
        "ocr_regions": {},
    }
    (model_dir / "market.yaml").write_text(
        yaml.safe_dump(manual, allow_unicode=True), encoding="utf-8"
    )

    report = convert(scr / "meta.json", scr, model_dir, assets)

    assert any("секция window" in i for i in report.issues)
    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(model_dir / "market.yaml", common)
    assert model.window.size == (500, 300)  # ручное побеждает


def test_existing_broken_model_yaml_is_replaced(workspace):
    """Существующий файл модели не mapping (битый/не YAML-словарь) —
    слиянию нечего беречь, модель перезаписывается машинной."""
    scr, model_dir, assets = workspace
    model_dir.mkdir()
    (model_dir / "market.yaml").write_text("- просто\n- список\n", encoding="utf-8")

    report = convert(scr / "meta.json", scr, model_dir, assets)

    assert str(model_dir / "market.yaml") in report.written_models
    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(model_dir / "market.yaml", common)
    assert model.window.size == (400, 300)


def test_unchanged_second_run_report_fields(workspace):
    """Повторный прогон без изменений: файлы модели не тронуты (поле
    written_models пусто), кропы отчитываются как совпадающие, замечание
    «модель без изменений» — в issues (детали diff-отчёта поверх
    байтовой идемпотентности test_second_run_is_idempotent)."""
    scr, model_dir, assets = workspace
    convert(scr / "meta.json", scr, model_dir, assets)
    second = convert(scr / "meta.json", scr, model_dir, assets)

    assert second.written_models == []
    assert second.assets_added == []
    assert second.assets_conflicts == []
    assert second.assets_identical == [
        "market/s1_tab.png",
        "market/s1_win.png",
        "market/s2_pic.png",
    ]
    assert any("без изменений" in i for i in second.issues)
