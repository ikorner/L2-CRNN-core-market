"""Тесты парсера UI-модели (l2core.model).

Покрывают: круглый путь load -> объектная модель -> доступ к элементам,
состояниям, переходам, OCR-регионам и геометрии таблицы; общий словарь;
диагностику ошибок формата с именем файла и секции. Семантические
инварианты (число сигнатур, вписывание rect'ов и т.п.) — зона валидатора
(задача 2.3), здесь не проверяются.
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
    assert dict(price.suffixes) == {"к": 1000, "кк": 1000000}

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
    # закрыто) -> search_area.
    opener = model.elements["open_buy_trade"]
    assert opener.hud
    assert opener.search_area == (751, 920, 34, 24)
    assert opener.rect is None

    # market_pic: ручная правка поверх импорта (HUD по спеке ui-model),
    # конвертер её не перезаписывает.
    pic = model.elements["market_pic"]
    assert pic.hud
    assert pic.search_area == (1000, 420, 200, 160)
    assert pic.variants["normal"].click_point == (86, 64)

    # состояние из меты: params + template-сигнатуры фактов наблюдения.
    home = model.states["list__tab_home"]
    assert home.params == {"tab": "home"}
    assert [(s.kind, s.element) for s in home.signatures] == [
        ("template", "market_main_button_active")
    ]

    # мета transitions/ocr_regions/table_geometry не знает — пусто.
    assert model.transitions == ()
    assert model.ocr_regions == {}
    assert model.table_geometry is None


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


def test_shipped_models_have_no_table_geometry_yet():
    """Реальный импорт (2.2) table_geometry не знает — секции нет."""
    common = load_common(UI_MODEL / "common.yaml")
    for window in ("market", "shop_buy", "shop_sell"):
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
