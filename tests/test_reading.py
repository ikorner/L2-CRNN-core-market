"""Обработка низкого confidence и стабилизация кадра для мигающей каретки
(задача 4.5, spec ocr-reading).

OcrClient подменяется фейком (duck-typing), источник кадров стабилизатора —
скриптованной последовательностью кадров: ни тест, ни код под тестом не
зависят от времени и внешних сервисов (инвариант 7 AGENTS.md).
"""
import numpy as np
import pytest

from l2core.model import (
    GrammarSpec,
    ModelFormatError,
    OcrRegionSpec,
    Rect,
    load_model,
)
from l2core.model.grammars import STATUS_INVALID, STATUS_OK
from l2core.model.reading import (
    OcrReadError,
    ReadAttempt,
    StableReadResult,
    read_confident,
    read_stable,
)

# --- Фейки ---


class FakeOcr:
    """Фейк OcrClient: очередь ответов (text, confidence), записанные кропы."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.crops = []

    def read(self, region_img):
        self.crops.append(np.asarray(region_img).copy())
        text, conf = self._answers.pop(0)
        return text, conf


class ScriptedFrames:
    """Источник кадров: callable, отдающий следующий кадр скрипта."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self._frames.pop(0)


def _frame(h=100, w=200, seed=0):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = seed  # кадры различимы по содержимому
    return frame


def _price_region(rect=None, **kw):
    return OcrRegionSpec(
        id="price_field",
        rect=rect or Rect(50, 40, 30, 10),
        grammar=GrammarSpec(
            name="price",
            kind="price",
            separators=frozenset({",", "."}),
            suffixes=(("к", 1000), ("кк", 1000000)),
        ),
        **kw,
    )


def _text_region(**kw):
    return OcrRegionSpec(
        id="search_field",
        rect=Rect(10, 10, 60, 12),
        grammar=GrammarSpec(name="text", kind="text"),
        **kw,
    )


# --- Цикл перечитываний при низком confidence ---


class TestReadConfident:
    def test_ok_first_attempt_no_rereads(self):
        ocr = FakeOcr([("240", 0.9)])
        result = read_confident(
            ocr, _frame(), _price_region(), threshold=0.85, reread_count=2
        )
        assert result.status == STATUS_OK
        assert result.value == 240
        assert len(ocr.crops) == 1

    def test_low_confidence_rereads_until_above_threshold(self):
        ocr = FakeOcr([("240", 0.5), ("240", 0.6), ("240", 0.9)])
        result = read_confident(
            ocr, _frame(), _price_region(), threshold=0.85, reread_count=2
        )
        assert result.status == STATUS_OK
        assert result.value == 240
        # 1 + reread_count внешних попыток, каждая — с расширением региона
        assert len(ocr.crops) == 3

    def test_each_outer_attempt_expands_region(self):
        ocr = FakeOcr([("240", 0.5), ("240", 0.6), ("240", 0.9)])
        read_confident(
            ocr, _frame(), _price_region(), threshold=0.85, reread_count=2, pad=5
        )
        heights = [crop.shape[0] for crop in ocr.crops]
        widths = [crop.shape[1] for crop in ocr.crops]
        # попытка k стартует с региона, увеличенного на k*pad от исходного
        assert heights == [10, 20, 30]
        assert widths == [30, 40, 50]

    def test_no_double_expansion_within_attempt(self):
        # несовпадение грамматики внутри попытки даёт одну внутреннюю
        # перечитку с расширением на pad; внешняя попытка k стартует с k*pad
        # от исходного региона — расширение не дублируется
        ocr = FakeOcr([("мусор", 0.9), ("мусор", 0.9), ("240", 0.5), ("240", 0.99)])
        result = read_confident(
            ocr, _frame(), _price_region(), threshold=0.85, reread_count=2, pad=5
        )
        assert result.status == STATUS_OK
        shapes = [crop.shape[:2] for crop in ocr.crops]
        assert shapes == [(10, 30), (20, 40), (20, 40), (30, 50)]

    def test_invalid_inside_attempt_continues_cycle(self):
        # invalid (грамматика не сошлась дважды) — confidence 0.0 < порога:
        # цикл переходит к следующей внешней попытке
        ocr = FakeOcr([("мусор", 0.9), ("мусор", 0.9), ("240", 0.9)])
        result = read_confident(
            ocr, _frame(), _price_region(), threshold=0.85, reread_count=1, pad=5
        )
        assert result.status == STATUS_OK
        assert len(ocr.crops) == 3  # 2 чтения попытки 0 + 1 чтение попытки 1

    def test_exhaustion_raises_with_diagnostics(self):
        ocr = FakeOcr([("240", 0.5), ("240", 0.6), ("240", 0.7)])
        with pytest.raises(OcrReadError) as excinfo:
            read_confident(
                ocr, _frame(), _price_region(), threshold=0.85, reread_count=2
            )
        err = excinfo.value
        assert err.region_id == "price_field"
        assert err.threshold == 0.85
        assert len(err.attempts) == 3
        assert all(isinstance(a, ReadAttempt) for a in err.attempts)
        assert err.attempts[-1].text == "240"
        assert err.attempts[-1].confidence == 0.7
        assert err.attempts[-1].status == STATUS_OK  # грамматика сошлась, conf низкий
        assert err.last_text == "240"
        assert err.last_confidence == 0.7
        # диагностика региона и значениями — в сообщении
        message = str(err)
        assert "price_field" in message
        assert "240" in message
        assert "0.85" in message

    def test_exhaustion_after_invalid(self):
        ocr = FakeOcr([("мусор", 0.9)] * 6)
        with pytest.raises(OcrReadError) as excinfo:
            read_confident(
                ocr, _frame(), _price_region(), threshold=0.85, reread_count=2
            )
        err = excinfo.value
        # все попытки invalid: последнее сырое значение в диагностике
        assert all(a.status == STATUS_INVALID for a in err.attempts)
        assert err.attempts[-1].text == "мусор"
        assert err.attempts[-1].confidence == 0.0

    def test_threshold_is_inclusive(self):
        ocr = FakeOcr([("240", 0.85)])
        result = read_confident(
            ocr, _frame(), _price_region(), threshold=0.85, reread_count=2
        )
        assert result.status == STATUS_OK
        assert len(ocr.crops) == 1

    def test_reread_count_zero_single_attempt(self):
        ocr = FakeOcr([("240", 0.5), ("240", 0.9)])
        with pytest.raises(OcrReadError) as excinfo:
            read_confident(
                ocr, _frame(), _price_region(), threshold=0.85, reread_count=0
            )
        assert len(excinfo.value.attempts) == 1
        assert len(ocr.crops) == 1  # второй ответ не израсходован

    def test_arguments_validated(self):
        ocr = FakeOcr([])
        with pytest.raises(ValueError):
            read_confident(
                ocr, _frame(), _price_region(), threshold=1.5, reread_count=1
            )
        with pytest.raises(ValueError):
            read_confident(
                ocr, _frame(), _price_region(), threshold=0.85, reread_count=-1
            )
        with pytest.raises(ValueError):
            read_confident(
                ocr, _frame(), _price_region(), threshold="0.9", reread_count=1
            )


# --- Стабилизация кадра для мигающей каретки ---


class TestReadStable:
    def test_caret_blink_stabilizes_on_two_equal_reads(self):
        # OCR чередует «ab|»/«ab» из-за каретки: стабильно «ab» со 2-го кадра
        ocr = FakeOcr([("ab|", 0.9), ("ab", 0.92), ("ab", 0.91)])
        frames = ScriptedFrames([_frame(seed=1), _frame(seed=2), _frame(seed=3)])
        result = read_stable(ocr, frames, _text_region())
        assert isinstance(result, StableReadResult)
        assert result.status == STATUS_OK
        assert result.value == "ab"
        assert result.text == "ab"
        assert result.frames == 3
        assert frames.calls == 3

    def test_first_two_frames_equal_stabilizes_immediately(self):
        ocr = FakeOcr([("Меч", 0.9), ("Меч", 0.9)])
        frames = ScriptedFrames([_frame(seed=1), _frame(seed=2)])
        result = read_stable(ocr, frames, _text_region())
        assert result.status == STATUS_OK
        assert result.value == "Меч"
        assert result.frames == 2

    def test_never_stabilized_raises_with_history(self):
        ocr = FakeOcr([("ab|", 0.9), ("ab", 0.9), ("ab|", 0.9), ("ab", 0.9)])
        frames = ScriptedFrames([_frame(seed=i) for i in range(4)])
        with pytest.raises(OcrReadError) as excinfo:
            read_stable(ocr, frames, _text_region(), max_frames=4)
        err = excinfo.value
        assert err.region_id == "search_field"
        assert [a.text for a in err.attempts] == ["ab|", "ab", "ab|", "ab"]
        assert "search_field" in str(err)

    def test_max_frames_limits_reads(self):
        ocr = FakeOcr([("x", 0.9)] * 5)
        frames = ScriptedFrames([_frame(seed=i) for i in range(5)])
        # стабилизировалось бы на 2-м кадре, но лимит — 1 кадр
        with pytest.raises(OcrReadError):
            read_stable(ocr, frames, _text_region(), max_frames=1)
        assert frames.calls == 1

    def test_placeholder_is_not_entered_text(self):
        region = _text_region(placeholder=r"Введите\s+название")
        ocr = FakeOcr([("Введите название", 0.9), ("Введите название", 0.9)])
        frames = ScriptedFrames([_frame(seed=1), _frame(seed=2)])
        result = read_stable(ocr, frames, region)
        assert result.status == "placeholder"
        assert result.value is None  # плейсхолдер ≠ введённый текст
        assert result.text == "Введите название"

    def test_entered_text_differs_from_placeholder(self):
        region = _text_region(placeholder=r"Введите\s+название")
        ocr = FakeOcr([("Меч", 0.9), ("Меч", 0.9)])
        frames = ScriptedFrames([_frame(seed=1), _frame(seed=2)])
        result = read_stable(ocr, frames, region)
        assert result.status == STATUS_OK
        assert result.value == "Меч"

    def test_empty_field_is_placeholder(self):
        # пустое поле (пустой OCR-текст) — не введённый текст; на каждый
        # кадр — до двух чтений (внутренняя перечитка при несовпадении)
        ocr = FakeOcr([("", 0.9)] * 4)
        frames = ScriptedFrames([_frame(seed=1), _frame(seed=2)])
        result = read_stable(ocr, frames, _text_region())
        assert result.status == "placeholder"
        assert result.value is None

    def test_stable_text_must_match_grammar(self):
        # стабильный мусор — не стабильный результат: диагностика стратегии
        ocr = FakeOcr([("abc|", 0.9)] * 4)
        frames = ScriptedFrames([_frame(seed=1), _frame(seed=2)])
        region = OcrRegionSpec(
            id="qty", rect=Rect(10, 10, 30, 10),
            grammar=GrammarSpec(name="digits", kind="digits"),
        )
        with pytest.raises(OcrReadError) as excinfo:
            read_stable(ocr, frames, region)
        assert "abc|" in str(excinfo.value)

    def test_max_frames_validated(self):
        with pytest.raises(ValueError):
            read_stable(FakeOcr([]), ScriptedFrames([]), _text_region(), max_frames=0)


# --- Поле placeholder в модели (loader) ---


_MINIMAL_MODEL = """\
window: {id: market, title: T, size: [100, 100]}
elements:
  btn: {rect: [1, 1, 8, 8], role: click_target}
states: {}
transitions: []
ocr_regions:
  search_field:
    rect: [1, 2, 10, 5]
    grammar: text
    placeholder: 'Введите\\s+название'
"""


class TestPlaceholderField:
    def test_placeholder_parsed(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text(_MINIMAL_MODEL, encoding="utf-8")
        model = load_model(path)
        region = model.ocr_regions["search_field"]
        assert region.placeholder == r"Введите\s+название"

    def test_placeholder_optional(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text(_MINIMAL_MODEL.replace(
            "    placeholder: 'Введите\\s+название'\n", ""
        ), encoding="utf-8")
        model = load_model(path)
        assert model.ocr_regions["search_field"].placeholder is None

    def test_invalid_regex_rejected(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text(_MINIMAL_MODEL.replace(
            r"'Введите\s+название'", "'['"
        ), encoding="utf-8")
        with pytest.raises(ModelFormatError) as excinfo:
            load_model(path)
        assert "placeholder" in str(excinfo.value)

    def test_placeholder_must_be_nonempty_string(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text(_MINIMAL_MODEL.replace(
            r"'Введите\s+название'", "''"
        ), encoding="utf-8")
        with pytest.raises(ModelFormatError):
            load_model(path)
