"""Грамматики OCR-регионов: разбор значений, валидация по диапазону,
механизм перечитывания при несовпадении (задача 4.4, spec ocr-reading).

OcrClient подменяется фейком (duck-typing, без наследования) — детерминизм:
ни тест, ни код под тестом не зависят от времени и внешних сервисов
(инвариант 7 AGENTS.md).
"""
import numpy as np
import pytest

from l2core.model import GrammarSpec, ModelFormatError, load_common
from l2core.model.grammars import (
    DEFAULT_DIGITS_MAX,
    DEFAULT_DIGITS_MIN,
    DEFAULT_PRICE_MAX,
    DEFAULT_PRICE_MIN,
    GrammarError,
    GrammarMismatch,
    ValueOutOfRange,
    expand_rect,
    parse_value,
    read_region,
)


def _price(**kw):
    base = dict(
        name="price",
        kind="price",
        separators=frozenset({",", "."}),
        suffixes=(("к", 1000), ("кк", 1000000)),
    )
    base.update(kw)
    return GrammarSpec(**base)


def _digits(**kw):
    return GrammarSpec(name="digits", kind="digits", **kw)


def _text():
    return GrammarSpec(name="text", kind="text")


class TestPrice:
    def test_fraction_with_kk_suffix(self):
        assert parse_value(_price(), "1.5кк") == 1500000

    def test_plain_integer(self):
        assert parse_value(_price(), "240") == 240

    def test_comma_grouping(self):
        assert parse_value(_price(), "1,500") == 1500
        assert parse_value(_price(), "1.500") == 1500

    def test_multi_group(self):
        assert parse_value(_price(), "1,500,000") == 1500000

    def test_grouping_with_suffix(self):
        assert parse_value(_price(), "2,500к") == 2500000

    def test_fraction_with_k_suffix(self):
        assert parse_value(_price(), "1,5к") == 1500
        assert parse_value(_price(), "10к") == 10000
        assert parse_value(_price(), "10кк") == 10000000

    def test_longest_suffix_wins(self):
        # «к» — префикс «кк»: сначала пробуем самый длинный суффикс
        assert parse_value(_price(), "3кк") == 3000000

    def test_fraction_without_suffix_is_mismatch(self):
        # игровая цена целая: дробь допустима только с суффиксом
        with pytest.raises(GrammarMismatch):
            parse_value(_price(), "1.5")

    def test_letters_are_mismatch(self):
        with pytest.raises(GrammarMismatch):
            parse_value(_price(), "абв")

    def test_empty_is_mismatch(self):
        with pytest.raises(GrammarMismatch):
            parse_value(_price(), "")

    def test_bad_group_length_is_mismatch(self):
        # ни разрядная группа (≠3 цифр), ни корректная дробь
        with pytest.raises(GrammarMismatch):
            parse_value(_price(), "1,2345")
        with pytest.raises(GrammarMismatch):
            parse_value(_price(), "1,50,000")

    def test_nonintegral_fraction_is_mismatch(self):
        # 1.2345 * 1000 = 1234.5 — нецелая адена
        with pytest.raises(GrammarMismatch):
            parse_value(_price(), "1.2345к")

    def test_zero_below_default_min(self):
        with pytest.raises(ValueOutOfRange):
            parse_value(_price(), "0")

    def test_above_default_max(self):
        with pytest.raises(ValueOutOfRange):
            parse_value(_price(), str(2**63))

    def test_custom_range(self):
        g = _price(min_value=10, max_value=99)
        assert parse_value(g, "50") == 50
        with pytest.raises(ValueOutOfRange):
            parse_value(g, "5")
        with pytest.raises(ValueOutOfRange):
            parse_value(g, "100")

    def test_default_range_constants(self):
        assert (DEFAULT_PRICE_MIN, DEFAULT_PRICE_MAX) == (1, 2**63 - 1)
        assert (DEFAULT_DIGITS_MIN, DEFAULT_DIGITS_MAX) == (0, 2**63 - 1)

    def test_whitespace_inside_number_removed(self):
        assert parse_value(_price(), "2 400") == 2400

    def test_suffix_case_exact(self):
        # суффиксы матчатся точно, регистр не нормализуется
        with pytest.raises(GrammarMismatch):
            parse_value(_price(), "1.5КК")


class TestPriceUnits:
    def _grammar(self):
        return _price(units=("аден",))

    def test_unit_stripped(self):
        assert parse_value(self._grammar(), "240 аден") == 240

    def test_unit_case_insensitive(self):
        assert parse_value(self._grammar(), "240 АДЕН") == 240

    def test_suffix_with_unit(self):
        assert parse_value(self._grammar(), "1.5кк аден") == 1500000

    def test_no_space_before_unit_is_mismatch(self):
        with pytest.raises(GrammarMismatch):
            parse_value(self._grammar(), "240аден")

    def test_unknown_unit_word_is_mismatch(self):
        # «млрд» в спеку не входит — не разбирается (вопрос владельцу Q22)
        with pytest.raises(GrammarMismatch):
            parse_value(self._grammar(), "240 млрд аден")


class TestDigits:
    def test_plain(self):
        assert parse_value(_digits(), "42") == 42

    def test_zero_allowed(self):
        assert parse_value(_digits(), "0") == 0
        assert parse_value(_digits(), "007") == 7

    def test_letters_are_mismatch(self):
        with pytest.raises(GrammarMismatch):
            parse_value(_digits(), "12а")
        with pytest.raises(GrammarMismatch):
            parse_value(_digits(), "1.5")

    def test_custom_range(self):
        g = _digits(min_value=1)
        with pytest.raises(ValueOutOfRange):
            parse_value(g, "0")
        assert parse_value(g, "1") == 1


class TestText:
    def test_free_text(self):
        assert parse_value(_text(), "Обзор лавок") == "Обзор лавок"

    def test_empty_is_mismatch(self):
        with pytest.raises(GrammarMismatch):
            parse_value(_text(), "")


class TestExpandRect:
    def test_expand_and_clamp(self):
        from l2core.model import Rect

        frame = np.zeros((100, 200), dtype=np.uint8)
        assert expand_rect(Rect(10, 10, 20, 20), frame.shape, 5) == Rect(
            5, 5, 30, 30
        )
        # у краёв кадра — прижимается, не вылезает
        assert expand_rect(Rect(0, 0, 20, 20), frame.shape, 5) == Rect(
            0, 0, 25, 25
        )


class FakeOcr:
    """Фейк OcrClient: очередь ответов (text, confidence), записанные кропы."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.crops = []

    def read(self, region_img):
        self.crops.append(np.asarray(region_img).copy())
        text, conf = self._answers.pop(0)
        return text, conf


def _frame(h=100, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestReadRegion:
    def test_ok_first_attempt(self):
        from l2core.model import Rect

        ocr = FakeOcr([("1.5кк", 0.97)])
        result = read_region(ocr, _frame(), Rect(10, 10, 30, 10), _price())
        assert result.status == "ok"
        assert result.value == 1500000
        assert result.attempts == 1
        assert result.confidence == 0.97
        assert len(ocr.crops) == 1

    def test_mismatch_triggers_single_reread_with_expansion(self):
        from l2core.model import Rect

        ocr = FakeOcr([("мусор", 0.9), ("1.5кк", 0.91)])
        rect = Rect(50, 40, 30, 10)
        result = read_region(ocr, _frame(), rect, _price(), pad=5)
        assert result.status == "ok"
        assert result.value == 1500000
        assert result.attempts == 2
        assert result.confidence == 0.91
        # второе чтение — с увеличенным регионом
        first, second = ocr.crops
        assert second.shape[0] > first.shape[0]
        assert second.shape[1] > first.shape[1]

    def test_repeated_mismatch_is_invalid_with_low_confidence(self):
        from l2core.model import Rect

        ocr = FakeOcr([("мусор", 0.9), ("мусор", 0.9)])
        result = read_region(ocr, _frame(), Rect(10, 10, 30, 10), _price())
        assert result.status == "invalid"
        assert result.value is None
        assert result.attempts == 2
        assert result.confidence == 0.0  # сигнал «низкий confidence» для 4.5

    def test_out_of_range_also_reread_then_invalid(self):
        from l2core.model import Rect

        ocr = FakeOcr([("999", 0.9), ("999", 0.9)])
        g = _price(max_value=100)
        result = read_region(ocr, _frame(), Rect(10, 10, 30, 10), g)
        assert result.status == "invalid"
        assert result.attempts == 2

    def test_text_grammar_reread(self):
        from l2core.model import Rect

        ocr = FakeOcr([("", 0.9), ("Меч", 0.88)])
        result = read_region(ocr, _frame(), Rect(10, 10, 30, 10), _text())
        assert result.status == "ok"
        assert result.value == "Меч"

    def test_reread_rect_clamped_to_frame(self):
        from l2core.model import Rect

        ocr = FakeOcr([("мусор", 0.9), ("42", 0.9)])
        frame = _frame(h=40, w=60)
        result = read_region(ocr, frame, Rect(0, 0, 10, 10), _digits(), pad=50)
        assert result.status == "ok"
        assert ocr.crops[-1].shape[:2] == (40, 60)


class TestLoaderGrammarRange:
    def test_min_max_units_parsed(self, tmp_path):
        path = tmp_path / "common.yaml"
        path.write_text(
            "variants: [normal]\n"
            "roles: [click_target]\n"
            'grammars:\n'
            '  price: {kind: price, separators: [","], suffixes: {к: 1000}, '
            'min: 10, max: 999, units: ["аден"]}\n',
            encoding="utf-8",
        )
        common = load_common(path)
        g = common.grammars["price"]
        assert g.min_value == 10
        assert g.max_value == 999
        assert g.units == ("аден",)

    def test_min_greater_max_rejected(self, tmp_path):
        path = tmp_path / "common.yaml"
        path.write_text(
            "variants: [normal]\n"
            "roles: [click_target]\n"
            "grammars:\n"
            "  price: {kind: price, min: 100, max: 10}\n",
            encoding="utf-8",
        )
        with pytest.raises(ModelFormatError) as excinfo:
            load_common(path)
        assert "min" in str(excinfo.value)

    def test_range_fields_rejected_for_text(self, tmp_path):
        path = tmp_path / "common.yaml"
        path.write_text(
            "variants: [normal]\n"
            "roles: [click_target]\n"
            "grammars:\n"
            "  text: {kind: text, min: 1}\n",
            encoding="utf-8",
        )
        with pytest.raises(ModelFormatError):
            load_common(path)

    def test_units_rejected_for_digits(self, tmp_path):
        path = tmp_path / "common.yaml"
        path.write_text(
            "variants: [normal]\n"
            "roles: [click_target]\n"
            'grammars:\n'
            '  digits: {kind: digits, units: ["аден"]}\n',
            encoding="utf-8",
        )
        with pytest.raises(ModelFormatError):
            load_common(path)

    def test_builtin_common_has_no_explicit_range(self):
        # дефолтный диапазон применяется на разборе, модель не меняется
        import l2core.model.loader as loader

        common = loader._builtin_common()
        assert common.grammars["price"].min_value is None
        assert common.grammars["price"].units == ()
