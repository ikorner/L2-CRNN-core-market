"""Тесты парсера UI-модели (l2core.model).

Покрывают: круглый путь load -> объектная модель -> доступ к элементам,
состояниям, переходам, OCR-регионам и геометрии таблицы; общий словарь;
диагностику ошибок формата с именем файла и секции (все ветки
ModelFormatError, задача 9.1). Семантические инварианты (число сигнатур,
вписывание rect'ов и т.п.) — зона валидатора (задача 2.3), здесь не
проверяются.
"""

from pathlib import Path

import pytest

from l2core.model import (
    ModelFormatError,
    load_common,
    load_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_MODEL = REPO_ROOT / "ui_model"
# Формат-образец из задачи 2.1 (вымышленная разметка, все конструкции
# формата). Мутационные тесты диагностики опираются на него, чтобы не
# зависеть от эволюции реальных моделей (результата импорта 2.2).
SAMPLE_MODEL = Path(__file__).resolve().parent / "fixtures" / "market_sample.yaml"


def test_load_common_dictionary():
    common = load_common(UI_MODEL / "common.yaml")

    assert "normal" in common.variants
    assert "active" in common.variants
    assert "click_target" in common.roles
    assert set(common.grammars) >= {"digits", "price", "text"}

    price = common.grammars["price"]
    assert price.kind == "price"
    assert "," in price.separators and "." in price.separators
    assert dict(price.suffixes) == {
        "к": 1000,
        "кк": 1000000,
        "тыс": 1000,
        "млн": 1000000,
        "млрд": 1000000000,
    }
    assert price.units == ("аден",)

    assert common.grammars["digits"].kind == "digits"
    assert common.grammars["text"].kind == "text"


def test_load_shipped_market_model_round_trip():
    """Поставляемая модель (реальный импорт конвертера 2.2) загружается."""
    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(UI_MODEL / "market.yaml", common)

    # window: размер от кропа-примера открытого окна.
    assert model.window.id == "market"
    assert model.window.title == "Обзор лавок"
    assert model.window.size == (1008, 674)

    # элемент из меты: rect от якоря окна, кроп-шаблон из assets/.
    button = model.elements["market_main_button_active"]
    assert button.role == "click_target"
    assert button.rect == (12, 155, 137, 42)
    assert button.variants["active"].template == (
        "market/scr_153551_market_main_button_active.png"
    )

    # HUD-элемент: не вписался в окно геометрически (окно на скриншоте
    # закрыто) -> search_area. Каноническое описание навигации живёт в
    # navigation.yaml; здесь — дубликаты для сигнатур recover_closed
    # (menu-quickslot-navigation).
    opener = model.elements["hamburger_button"]
    assert opener.hud
    assert opener.search_area == (1000, 740, 136, 100)
    assert opener.rect is None

    # quickslot_sell_icon: HUD-элемент панели быстрого доступа (слот F2).
    qs = model.elements["quickslot_sell_icon"]
    assert qs.hud
    assert qs.search_area == (430, 700, 500, 100)

    # состояние из меты: params + template-сигнатуры фактов наблюдания.
    # Вторая сигнатура добавлена в 2.5 (пробел KNOWN_GAPS закрыт правкой
    # модели): открытое окно целиком, scr_153551.
    home = model.states["list__tab_home"]
    assert home.params == {"tab": "home"}
    assert [(s.kind, s.element) for s in home.signatures] == [
        ("template", "market_main_button_active"),
        ("template", "market_main_menu"),
    ]

    # семантика подтверждённых переходов размечена в 2.5 (данные —
    # тесты/test_transitions.py); неподтверждённая — KNOWN_GAPS/OWNER_QUESTIONS.
    assert model.transitions != ()
    assert "recover_closed" in model.states
    assert model.ocr_regions == {}
    assert model.table_geometry is not None
    assert model.table_geometry.columns["price"].grammar.kind == "price"


def test_shipped_market_close_from_every_list_state():
    """Q10 (владелец, приёмка): крестик закрытия = ESC — закрывает окно из
    ЛЮБОГО list-состояния (проверено на «Снаряжение»+«Продажа» и
    «Коллекции»). Страж: у каждого list__*-состояния размечен переход
    закрытия в recover_closed — новые состояния без него ломают тест."""
    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(UI_MODEL / "market.yaml", common)
    list_states = {sid for sid in model.states if sid.startswith("list__")}
    assert list_states, "list-состояний нет — тест ничего не проверяет"
    close_sources = {
        t.source
        for t in model.transitions
        if getattr(t.action, "element", None) == "main_market_close_button"
        and t.expected == "recover_closed"
    }
    missing = sorted(list_states - close_sources)
    assert not missing, f"из состояний нет перехода закрытия: {missing}"


def test_empty_states_and_ocr_regions_are_valid(tmp_path):
    """Начальный импорт (2.2) до ручной разметки: пустые секции валидны."""
    path = tmp_path / "market.yaml"
    path.write_text(
        "window: {id: market, title: Обзор лавок, size: [100, 100]}\n"
        "elements:\n"
        "  e: {rect: [0, 0, 1, 1], role: icon_sample}\n"
        "states: {}\n"
        "transitions: []\n"
        "ocr_regions: {}\n",
        encoding="utf-8",
    )
    model = load_model(path)
    assert model.states == {}
    assert model.ocr_regions == {}


def test_load_sample_fixture_round_trip():
    """Формат-образец 2.1 загружается; объектная модель отражает YAML."""
    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(SAMPLE_MODEL, common)

    # window
    assert model.window.id == "market"
    assert model.window.title == "Обзор лавок"
    assert model.window.size == (804, 598)

    # elements: обычный элемент с вариантами
    tab_home = model.elements["tab_home"]
    assert tab_home.role == "click_target"
    assert tab_home.rect == (10, 40, 142, 42)
    assert not tab_home.hud
    assert set(tab_home.variants) == {"normal", "active"}
    assert tab_home.variants["active"].template == "market/tab_home_active.png"
    assert tab_home.variants["active"].click_point == (81, 61)
    assert tab_home.variants["normal"].click_point is None

    # elements: HUD-элемент задан областью поиска, а не rect
    market_pic = model.elements["market_pic"]
    assert market_pic.hud
    assert market_pic.search_area == (1000, 420, 200, 160)
    assert market_pic.rect is None
    assert market_pic.variants["normal"].template == "hud/market_pic.png"

    # states: параметры и иерархия сигнатур
    home = model.states["list_home"]
    assert home.params == {"tab": "home"}
    assert [s.kind for s in home.signatures] == [
        "pixel_probe",
        "template",
        "ocr",
    ]
    probe = home.signatures[0]
    assert probe.points[0].at == (20, 20)
    assert probe.points[0].color == (255, 220, 120)
    tpl = home.signatures[1]
    assert tpl.element == "tab_home" and tpl.variant == "active"
    ocr_sig = home.signatures[2]
    assert ocr_sig.region == "window_title"
    assert ocr_sig.expect == "Обзор лавок"

    equipment = model.states["list_equipment"]
    assert equipment.params["tab"] == "equipment"
    assert equipment.params["global_filter"] == "all"

    # transitions: (state, action) -> (expected, verify, timeout, on_fail)
    click_tr, key_tr = model.transitions
    assert click_tr.source == "list_home"
    assert click_tr.action.type == "click"
    assert click_tr.action.element == "tab_equipment"
    assert click_tr.expected == "list_equipment"
    assert click_tr.timeout == 5.0
    assert click_tr.verify[0].kind == "template"
    assert click_tr.on_fail == ("press_esc", "reopen_market")

    assert key_tr.action.type == "key"
    assert key_tr.action.key == "escape"
    assert key_tr.timeout is None  # не задан — из конфига при исполнении

    # ocr_regions: ссылки на грамматики разрешены
    assert model.ocr_regions["price_cell"].grammar.kind == "price"
    assert model.ocr_regions["window_title"].grammar.kind == "text"
    assert model.ocr_regions["price_cell"].rect == (300, 100, 120, 20)

    # table_geometry
    tg = model.table_geometry
    assert tg is not None
    assert tg.col_seps == (12, 240, 372, 500, 640, 780)
    assert tg.row_y0 == 96
    assert tg.row_pitch == 24
    assert tg.max_rows == 18
    assert tg.columns["price"].grammar.kind == "price"
    assert tg.columns["item"].zone == (4, 2, 220, 20)


def test_model_without_table_geometry_loads(tmp_path):
    """Окон без списка table_geometry нет; парсер такую модель принимает."""
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        drop_section="table_geometry",
    )
    common = load_common(UI_MODEL / "common.yaml")
    model = load_model(path, common)
    assert model.table_geometry is None


def test_grid_geometry_parsed(tmp_path):
    """Опциональная секция grid_geometry (сетка слотов лавки, 6.2)."""
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        extra=(
            "grid_geometry:\n"
            "  origin: [10, 20]\n"
            "  pitch: [30, 10]\n"
            "  cell: [28, 8]\n"
            "  cols: 3\n"
            "  rows: 2\n"
        ),
    )
    model = load_model(path)
    grid = model.grid_geometry
    assert grid is not None
    assert grid.origin == (10, 20)
    assert grid.pitch == (30, 10)
    assert grid.cell == (28, 8)
    assert grid.cols == 3
    assert grid.rows == 2


def test_model_without_grid_geometry_loads(tmp_path):
    """Секция опциональна: не размечена — модель валидна (grid=None)."""
    model = load_model(SAMPLE_MODEL)
    assert model.grid_geometry is None


@pytest.mark.parametrize("bad_section", [
    "grid_geometry:\n  origin: [10, 20]\n  pitch: [0, 10]\n"
    "  cell: [28, 8]\n  cols: 3\n  rows: 2\n",
    "grid_geometry:\n  origin: [10, 20]\n  pitch: [30, 10]\n"
    "  cell: [28, 8]\n  cols: 0\n  rows: 2\n",
    "grid_geometry:\n  origin: [10, 20]\n  pitch: [30, 10]\n"
    "  cell: [28, 8]\n  cols: 3\n",
])
def test_grid_geometry_bad_values_diagnosed(tmp_path, bad_section):
    """Некорректная геометрия сетки — ошибка формата с секцией."""
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        extra=bad_section,
    )
    with pytest.raises(ModelFormatError, match="grid_geometry"):
        load_model(path)


def test_shipped_models_table_geometry_presence():
    """2.4: геометрия размечена у market; у shop_* текстовой таблицы нет."""
    common = load_common(UI_MODEL / "common.yaml")
    market = load_model(UI_MODEL / "market.yaml", common)
    assert market.table_geometry is not None
    for window in ("shop_buy", "shop_sell"):
        model = load_model(UI_MODEL / f"{window}.yaml", common)
        assert model.table_geometry is None


def test_load_model_with_default_common():
    """common не передан — встроенный словарь (совпадает с поставляемым)."""
    default_model = load_model(SAMPLE_MODEL)
    common = load_common(UI_MODEL / "common.yaml")
    explicit_model = load_model(SAMPLE_MODEL, common)

    assert default_model.ocr_regions["price_cell"].grammar == (
        explicit_model.ocr_regions["price_cell"].grammar
    )


def test_unknown_top_level_section_rejected(tmp_path):
    path = _write_model(tmp_path, base=UI_MODEL / "market.yaml", extra="wat: 1\n")
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "wat" in str(excinfo.value)
    assert "market" in str(excinfo.value)  # имя файла в диагностике


@pytest.mark.parametrize("section", ["window", "elements", "states", "transitions", "ocr_regions"])
def test_missing_required_section_diagnosed(tmp_path, section):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        drop_section=section,
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    msg = str(excinfo.value)
    assert section in msg
    assert "market" in msg


@pytest.mark.parametrize(
    "rect_yaml",
    [
        "[10, 40]",  # меньше 4 чисел
        "[10, 40, 142]",  # меньше 4 чисел
        "[10, 40, 142, 42, 0]",  # больше 4 чисел
        "[10, 40, -1, 42]",  # отрицательная ширина
        "[10, 40, 142, 0]",  # нулевая высота
        "[10.5, 40, 142, 42]",  # не целые
        "[10, 40, широко, 42]",  # не числа
        "просто строка",
    ],
)
def test_invalid_rect_diagnosed(tmp_path, rect_yaml):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="rect: [10, 40, 142, 42]",
        with_text=f"rect: {rect_yaml}",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    msg = str(excinfo.value)
    assert "elements" in msg and "tab_home" in msg


def test_unknown_role_diagnosed(tmp_path):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="role: click_target",
        with_text="role: blinking_thing",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "blinking_thing" in str(excinfo.value)


def test_unknown_variant_diagnosed(tmp_path):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="normal: {template: market/tab_home_normal.png}",
        with_text="nomral: {template: market/tab_home_normal.png}",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "nomral" in str(excinfo.value)


def test_unknown_grammar_reference_diagnosed(tmp_path):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="grammar: price",
        with_text="grammar: hex_number",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "hex_number" in str(excinfo.value)


def test_unknown_check_kind_diagnosed(tmp_path):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="{kind: template, element: tab_home, variant: active}",
        with_text="{kind: magic, element: tab_home}",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "magic" in str(excinfo.value)


def test_check_missing_required_field_diagnosed(tmp_path):
    """template-проверка без variant — поле обязательно для вида."""
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="{kind: template, element: tab_home, variant: active}",
        with_text="{kind: template, element: tab_home}",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "variant" in str(excinfo.value)


def test_hud_element_requires_search_area(tmp_path):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="search_area: [1000, 420, 200, 160]",
        with_text="rect: [1000, 420, 200, 160]",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "market_pic" in str(excinfo.value)
    assert "search_area" in str(excinfo.value)


def test_plain_element_rejects_search_area(tmp_path):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="rect: [10, 40, 142, 42]",
        with_text="search_area: [10, 40, 142, 42]",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "tab_home" in str(excinfo.value)


def test_broken_yaml_diagnosed(tmp_path):
    path = tmp_path / "market.yaml"
    path.write_text("window:\n  id: [разорвано\n", encoding="utf-8")
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "market.yaml" in str(excinfo.value)


def test_top_level_not_mapping(tmp_path):
    path = tmp_path / "market.yaml"
    path.write_text("- просто\n- список\n", encoding="utf-8")
    with pytest.raises(ModelFormatError):
        load_model(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.yaml")


def test_unknown_key_inside_element_diagnosed(tmp_path):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="role: click_target",
        with_text="role: click_target\n    color: red",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert "color" in str(excinfo.value)


def test_bad_price_suffix_diagnosed(tmp_path):
    """Суффикс грамматики price должен быть строка -> положительное число."""
    path = tmp_path / "common.yaml"
    path.write_text(
        'variants: [normal]\n'
        'roles: [click_target]\n'
        'grammars:\n'
        '  price: {kind: price, separators: [","], suffixes: {к: -1000}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_common(path)
    assert "suffixes" in str(excinfo.value) or "к" in str(excinfo.value)


def test_error_message_contains_file_and_section(tmp_path):
    path = _write_model(
        tmp_path,
        base=SAMPLE_MODEL,
        replace="role: click_target",
        with_text="role: blinking_thing",
    )
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    msg = str(excinfo.value)
    assert "market.yaml" in msg  # файл
    assert "elements" in msg  # секция
    assert "tab_home" in msg  # элемент


# --- Задача 9.1: доведение покрытия диагностики формата до всех веток ---

# Мутации формат-образца: (id, kwargs для _write_model, фрагмент диагностики).
# Каждая параправка должна приводить к ModelFormatError своей ветки парсера
# (loader.py): проверяем и саму ошибку, и наличие в сообщении якоря ветки.
_FORMAT_CASES = [
    # --- window ---
    pytest.param(
        {"replace": (
            'window:\n  id: market\n  title: "Обзор лавок"\n  size: [804, 598]'),
         "with_text": "window: просто строка"},
        "window",
        id="window-не-mapping",
    ),
    pytest.param(
        {"replace": "  id: market", "with_text": "  id: bazaar"},
        "неизвестное окно",
        id="window-неизвестный-id",
    ),
    pytest.param(
        {"replace": '  title: "Обзор лавок"', "with_text": '  title: ""'},
        "title",
        id="window-пустой-title",
    ),
    pytest.param(
        {"replace": "  size: [804, 598]", "with_text": "  size: [804]"},
        "size",
        id="window-size-не-пара",
    ),
    pytest.param(
        {"replace": "  size: [804, 598]", "with_text": "  size: [804, -598]"},
        "size",
        id="window-size-не-положительные",
    ),
    pytest.param(
        {"replace": "  size: [804, 598]",
         "with_text": "  size: [804, 598]\n  zorder: 3"},
        "неизвестные ключи",
        id="window-неизвестный-ключ",
    ),
    # --- elements ---
    pytest.param(
        {"drop_section": "elements", "extra": "elements: [tab_home]\n"},
        "elements",
        id="elements-не-mapping",
    ),
    pytest.param(
        {"drop_section": "elements", "extra": "elements: {}\n"},
        "непустой mapping",
        id="elements-пусто",
    ),
    pytest.param(
        {"drop_section": "elements", "extra": "elements:\n  tab_home: 42\n"},
        "tab_home",
        id="elements-спека-не-mapping",
    ),
    pytest.param(
        {"replace": "    role: click_target", "with_text": ""},
        "неизвестная роль",
        id="elements-роль-отсутствует",
    ),
    pytest.param(
        {"replace": "    hud: true", "with_text": "    hud: 1"},
        "hud",
        id="elements-hud-не-bool",
    ),
    pytest.param(
        {"replace": "    search_area: [1000, 420, 200, 160]",
         "with_text": "    rect: [1, 2, 3, 4]\n    search_area: [1000, 420, 200, 160]"},
        "не задаёт rect",
        id="elements-hud-с-rect",
    ),
    # --- variants ---
    pytest.param(
        {"replace": (
            "    variants:\n"
            "      normal: {template: market/tab_home_normal.png}\n"
            "      active: {template: market/tab_home_active.png, click_point: [81, 61]}"),
         "with_text": "    variants: [normal, active]"},
        "variants",
        id="variants-не-mapping",
    ),
    pytest.param(
        {"replace": "      normal: {template: market/tab_home_normal.png}",
         "with_text": "      normal: 5"},
        "normal",
        id="variants-спека-не-mapping",
    ),
    pytest.param(
        {"replace": "      normal: {template: market/tab_home_normal.png}",
         "with_text": "      normal: {template: market/tab_home_normal.png, blur: 3}"},
        "неизвестные ключи",
        id="variants-неизвестный-ключ",
    ),
    pytest.param(
        {"replace": "      normal: {template: market/tab_home_normal.png}",
         "with_text": "      normal: {click_point: [1, 2]}"},
        "template",
        id="variants-template-отсутствует",
    ),
    pytest.param(
        {"replace": "      normal: {template: market/tab_home_normal.png}",
         "with_text": "      normal: {template: ''}"},
        "template",
        id="variants-template-пустой",
    ),
    pytest.param(
        {"replace": (
            "      active: {template: market/tab_home_active.png, click_point: [81, 61]}"),
         "with_text": (
            "      active: {template: market/tab_home_active.png, click_point: [81]}")},
        "click_point",
        id="variants-click_point-не-пара",
    ),
    # --- states ---
    pytest.param(
        {"drop_section": "states", "extra": "states: [list_home]\n"},
        "states",
        id="states-не-mapping",
    ),
    pytest.param(
        {"drop_section": "states", "extra": "states:\n  list_home: 7\n"},
        "list_home",
        id="states-спека-не-mapping",
    ),
    pytest.param(
        {"replace": "    params:\n      tab: home",
         "with_text": "    params:\n      tab: home\n    depth: 3"},
        "неизвестные ключи",
        id="states-неизвестный-ключ",
    ),
    pytest.param(
        {"replace": "      tab: home", "with_text": "      tab: [home]"},
        "params",
        id="states-params-не-строки",
    ),
    pytest.param(
        {"replace": (
            "    signatures:\n"
            "      - {kind: pixel_probe, points: [{at: [20, 20], color: [255, 220, 120]}]}\n"
            "      - {kind: template, element: tab_home, variant: active}\n"
            '      - {kind: ocr, region: window_title, expect: "Обзор лавок"}'),
         "with_text": "    signatures: []"},
        "signatures",
        id="states-signatures-пусто",
    ),
    pytest.param(
        {"replace": (
            "    signatures:\n"
            "      - {kind: pixel_probe, points: [{at: [20, 20], color: [255, 220, 120]}]}\n"
            "      - {kind: template, element: tab_home, variant: active}\n"
            '      - {kind: ocr, region: window_title, expect: "Обзор лавок"}'),
         "with_text": "    signatures: {a: b}"},
        "signatures",
        id="states-signatures-не-список",
    ),
    # --- проверки (сигнатуры/verify) ---
    pytest.param(
        {"replace": "{kind: pixel_probe, points: [{at: [20, 20], color: [255, 220, 120]}]}",
         "with_text": "строка"},
        "ожидается mapping",
        id="check-не-mapping",
    ),
    pytest.param(
        {"replace": "points: [{at: [20, 20], color: [255, 220, 120]}]",
         "with_text": "points: []"},
        "points",
        id="pixel_probe-points-пусто",
    ),
    pytest.param(
        {"replace": "points: [{at: [20, 20], color: [255, 220, 120]}]",
         "with_text": "points: [[20, 20]]"},
        "at/color",
        id="pixel_probe-точка-не-mapping",
    ),
    pytest.param(
        {"replace": "points: [{at: [20, 20], color: [255, 220, 120]}]",
         "with_text": "points: [{at: [20, 20], color: [255, 220, 120], radius: 3}]"},
        "radius",
        id="pixel_probe-неизвестный-ключ-точки",
    ),
    pytest.param(
        {"replace": "points: [{at: [20, 20], color: [255, 220, 120]}]",
         "with_text": "points: [{at: [20, 20], color: [255, 220]}]"},
        "color",
        id="pixel_probe-color-не-triplet",
    ),
    pytest.param(
        {"replace": "- {kind: template, element: tab_home, variant: active}",
         "with_text": "- {kind: template, variant: active}"},
        "element",
        id="template-element-отсутствует",
    ),
    pytest.param(
        {"replace": '- {kind: ocr, region: window_title, expect: "Обзор лавок"}',
         "with_text": '- {kind: ocr, expect: "Обзор лавок"}'},
        "region",
        id="ocr-region-отсутствует",
    ),
    pytest.param(
        {"replace": '- {kind: ocr, region: window_title, expect: "Обзор лавок"}',
         "with_text": "- {kind: ocr, region: window_title}"},
        "expect",
        id="ocr-expect-отсутствует",
    ),
    # --- transitions ---
    pytest.param(
        {"drop_section": "transitions", "extra": "transitions: просто текст\n"},
        "список переходов",
        id="transitions-не-список",
    ),
    pytest.param(
        {"drop_section": "transitions", "extra": "transitions:\n  - просто строка\n"},
        "ожидается mapping",
        id="transition-спека-не-mapping",
    ),
    pytest.param(
        {"replace": "    element: tab_equipment\n    expected: list_equipment",
         "with_text": "    element: tab_equipment\n    wobble: 3\n    expected: list_equipment"},
        "неизвестные ключи",
        id="transition-неизвестный-ключ",
    ),
    pytest.param(
        {"replace": "  - from: list_home\n    action: click",
         "with_text": "  - action: click"},
        "from",
        id="transition-from-отсутствует",
    ),
    pytest.param(
        {"replace": "    action: click", "with_text": "    action: wave"},
        "action",
        id="transition-action-неизвестен",
    ),
    pytest.param(
        {"replace": "    element: tab_equipment\n    expected: list_equipment",
         "with_text": "    expected: list_equipment"},
        "click требует",
        id="transition-click-без-element",
    ),
    pytest.param(
        {"replace": "    element: tab_equipment\n    expected: list_equipment",
         "with_text": "    element: tab_equipment\n    key: escape\n    expected: list_equipment"},
        "key недопустим",
        id="transition-key-при-click",
    ),
    pytest.param(
        {"replace": "    action: key\n    key: escape", "with_text": "    action: key"},
        "key-действие требует",
        id="transition-key-без-key",
    ),
    pytest.param(
        {"replace": "    key: escape\n    expected: closed",
         "with_text": "    key: escape\n    element: tab_equipment\n    expected: closed"},
        "element недопустим",
        id="transition-element-при-key",
    ),
    pytest.param(
        {"replace": "    expected: list_equipment\n    timeout: 5.0",
         "with_text": "    timeout: 5.0"},
        "expected",
        id="transition-expected-отсутствует",
    ),
    pytest.param(
        {"replace": "    timeout: 5.0", "with_text": "    timeout: -1"},
        "timeout",
        id="transition-timeout-отрицательный",
    ),
    pytest.param(
        {"replace": "    timeout: 5.0", "with_text": "    timeout: true"},
        "timeout",
        id="transition-timeout-bool",
    ),
    pytest.param(
        {"replace": "    on_fail: [press_esc, reopen_market]", "with_text": ""},
        "on_fail",
        id="transition-on_fail-отсутствует",
    ),
    pytest.param(
        {"replace": "    on_fail: [press_esc, reopen_market]",
         "with_text": "    on_fail: []"},
        "on_fail",
        id="transition-on_fail-пусто",
    ),
    pytest.param(
        {"replace": "    on_fail: [press_esc, reopen_market]",
         "with_text": "    on_fail: [press_esc, 42]"},
        "on_fail",
        id="transition-on_fail-не-строки",
    ),
    # --- ocr_regions ---
    pytest.param(
        {"drop_section": "ocr_regions", "extra": "ocr_regions: [window_title]\n"},
        "ожидается mapping id: регион",
        id="ocr_regions-не-mapping",
    ),
    pytest.param(
        {"drop_section": "ocr_regions", "extra": "ocr_regions:\n  window_title: 9\n"},
        "window_title",
        id="ocr_region-спека-не-mapping",
    ),
    pytest.param(
        {"replace": "    grammar: text",
         "with_text": "    grammar: text\n    lang: ru"},
        "неизвестные ключи",
        id="ocr_region-неизвестный-ключ",
    ),
    pytest.param(
        {"replace": "    grammar: text", "with_text": ""},
        "имя грамматики",
        id="ocr_region-grammar-отсутствует",
    ),
    # --- table_geometry ---
    pytest.param(
        {"replace": "  col_seps: [12, 240, 372, 500, 640, 780]",
         "with_text": "  col_seps: [12]"},
        "col_seps",
        id="table_geometry-col_seps-коротко",
    ),
    pytest.param(
        {"replace": "  col_seps: [12, 240, 372, 500, 640, 780]",
         "with_text": "  col_seps: [12.5, 240]"},
        "col_seps",
        id="table_geometry-col_seps-не-целые",
    ),
    pytest.param(
        {"replace": "  row_y0: 96", "with_text": "  row_y0: высоко"},
        "row_y0",
        id="table_geometry-row_y0-не-целое",
    ),
    pytest.param(
        {"replace": "  row_pitch: 24", "with_text": "  row_pitch: 0"},
        "row_pitch",
        id="table_geometry-row_pitch-ноль",
    ),
    pytest.param(
        {"replace": "  max_rows: 18", "with_text": "  max_rows: -1"},
        "max_rows",
        id="table_geometry-max_rows-отрицательное",
    ),
    pytest.param(
        {"replace": (
            "  columns:\n"
            "    item:\n"
            "      zone: [4, 2, 220, 20]\n"
            "      grammar: text\n"
            "    price:\n"
            "      zone: [4, 2, 120, 20]\n"
            "      grammar: price"),
         "with_text": "  columns: [item, price]"},
        "columns",
        id="table_geometry-columns-не-mapping",
    ),
    pytest.param(
        {"replace": (
            "    item:\n"
            "      zone: [4, 2, 220, 20]\n"
            "      grammar: text"),
         "with_text": "    item: 4"},
        "columns.item",
        id="table_geometry-column-спека-не-mapping",
    ),
    pytest.param(
        {"replace": "      zone: [4, 2, 220, 20]\n      grammar: text",
         "with_text": "      zone: [4, 2, 220, 20]\n      grammar: text\n      weight: 1"},
        "weight",
        id="table_geometry-column-неизвестный-ключ",
    ),
    # --- прочие базовые разборы ---
    pytest.param(
        {"replace": "    rect: [10, 40, 142, 42]",
         "with_text": "    rect: [-1, 40, 142, 42]"},
        "отрицательными",
        id="rect-отрицательные-координаты",
    ),
    pytest.param(
        {"replace": "click_point: [81, 61]", "with_text": "click_point: [-1, 61]"},
        "отрицательными",
        id="point-отрицательные-координаты",
    ),
    # --- grid_geometry ---
    pytest.param(
        {"extra": (
            "grid_geometry:\n"
            "  origin: [10, 20]\n"
            "  pitch: [30, 10]\n"
            "  cell: [28, 8]\n"
            "  cols: 3\n"
            "  rows: 2\n"
            "  angle: 45\n")},
        "неизвестные ключи",
        id="grid_geometry-неизвестный-ключ",
    ),
]


@pytest.mark.parametrize("kwargs, fragment", _FORMAT_CASES)
def test_format_violations_all_branches_diagnosed(tmp_path, kwargs, fragment):
    """Мутация формата -> ModelFormatError своей ветки парсера (9.1: аудит
    покрытия — каждая проверка формата из loader.py имеет тест)."""
    path = _write_model(tmp_path, **kwargs)
    with pytest.raises(ModelFormatError) as excinfo:
        load_model(path)
    assert fragment in str(excinfo.value)


# (yaml common, фрагмент диагностики): ветки _parse_grammar/_string_list/
# _reject_unknown общего словаря (load_common).
_COMMON_CASES = [
    pytest.param(
        'variants: [normal]\nroles: [click_target]\ngrammars: [price]\n',
        "grammars",
        id="common-grammars-не-mapping",
    ),
    pytest.param(
        'variants: [normal]\nroles: [click_target]\n'
        "grammars:\n  price: [kind, price]\n",
        "ожидается mapping",
        id="common-grammar-спека-не-mapping",
    ),
    pytest.param(
        'variants: [normal]\nroles: [click_target]\n'
        "grammars:\n  hex: {kind: hex}\n",
        "неизвестный вид грамматики",
        id="common-grammar-вид-неизвестен",
    ),
    pytest.param(
        'variants: [normal]\nroles: [click_target]\n'
        "grammars:\n  price: {kind: price, suffixes: [к]}\n",
        "суффикс: множитель",
        id="common-suffixes-не-mapping",
    ),
    pytest.param(
        'variants: [normal]\nroles: [click_target]\n'
        "grammars:\n  digits: {kind: digits, min: true}\n",
        "min",
        id="common-min-не-целое",
    ),
    pytest.param(
        'variants: [normal]\nroles: [click_target]\n'
        'grammars:\n  price: {kind: price, separators: ","}\n',
        "separators",
        id="common-separators-не-список",
    ),
    pytest.param(
        'variants: normal\nroles: [click_target]\ngrammars: {}\n',
        "непустой список строк",
        id="common-variants-не-список",
    ),
    pytest.param(
        'variants: [normal, normal]\nroles: [click_target]\ngrammars: {}\n',
        "повторы",
        id="common-variants-дубли",
    ),
    pytest.param(
        'variants: [normal]\nroles: [click_target, 5]\ngrammars: {}\n',
        "непустых строк",
        id="common-roles-не-строки",
    ),
    pytest.param(
        'variants: [normal]\nroles: [click_target]\ngrammars: {}\nextra: 1\n',
        "неизвестные ключи",
        id="common-неизвестный-ключ",
    ),
]


@pytest.mark.parametrize("yaml_text, fragment", _COMMON_CASES)
def test_common_format_violations_all_branches_diagnosed(tmp_path, yaml_text, fragment):
    """Ветки диагностики общего словаря (load_common): каждая проверка
    формата имеет тест (9.1: аудит покрытия)."""
    path = tmp_path / "common.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ModelFormatError) as excinfo:
        load_common(path)
    assert fragment in str(excinfo.value)


# --- Вспомогательные функции ---


def _write_model(tmp_path, *, base=None, replace=None, with_text=None,
                 extra=None, drop_section=None):
    """Собрать копию модели с правкой и записать в tmp_path/market.yaml.

    Правки линейные (первое вхождение), этого достаточно для тестов
    диагностики. base по умолчанию — поставляемый пример.
    """
    if base is None:
        base = SAMPLE_MODEL
    text = base.read_text(encoding="utf-8")

    if drop_section is not None:
        lines = text.splitlines(keepends=True)
        out, skip_indent = [], None
        for line in lines:
            if skip_indent is None:
                if line.rstrip("\n") == f"{drop_section}:":
                    skip_indent = len(line) - len(line.lstrip())
                    continue  # заголовок секции тоже убираем
                out.append(line)
            else:
                indent = len(line) - len(line.lstrip())
                if line.strip() and indent <= skip_indent:
                    skip_indent = None
                    out.append(line)
        text = "".join(out)
    if replace is not None:
        assert with_text is not None
        assert replace in text, f"тест написан неверно: {replace!r} нет в модели"
        text = text.replace(replace, with_text, 1)
    if extra is not None:
        text += extra

    target_dir = tmp_path if tmp_path is not None else Path(".")
    path = target_dir / "market.yaml"
    path.write_text(text, encoding="utf-8")
    return path
