"""Тесты конвертера meta.json → UI-модель (l2core.model.converter).

Покрывают: импорт кропов в assets/ (включая идемпотентность повторного
прогона), сбор карточек элементов и состояний, diff-отчёт и политику
слияния «ручное побеждает, машинное дополняет», пропуск ролей-сырья meta
(decor/ocr_ground_truth) и круглый путь: сгенерированная модель
распарсивается существующим load_model.
"""

import json
from pathlib import Path

import pytest
import yaml

from l2core.model import load_common, load_model
from l2core.model.converter import convert

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


def test_shipped_models_generated_by_converter_parse():
    """Поставляемые модели (результат импорта настоящей меты) парсятся."""
    common = load_common(UI_MODEL / "common.yaml")
    for window in ("market", "shop_buy", "shop_sell"):
        model = load_model(UI_MODEL / f"{window}.yaml", common)
        assert model.window.id == window
