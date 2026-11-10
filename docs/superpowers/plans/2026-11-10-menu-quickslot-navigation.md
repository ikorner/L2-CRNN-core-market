# План реализации: menu-quickslot-navigation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** заменить сломанный HUD-путь открытия окон торговли навигацией «Меню/quick-slot»: навигационная модель `ui_model/navigation.yaml` с декларативными рецептами, примитив `press_key` (X), исполнитель рецептов, API на рецептах.

**Architecture:** всё знание об UI навигации — в YAML (рецепты: шаги key/click с verify/on_fail/when/skip_if/alternate); код — только исполнитель поверх существующих примитивов и инъекций (тот же паттерн, что `click_element` 5.3 и `_open_via_hud`). Кейкод X — в отдельном модуле (`adapters/hid.py` вендорен, правки запрещены).

**Tech Stack:** Python 3.10+, pytest, numpy, OpenCV (template matching), PyYAML.

**Spec:** `openspec/changes/menu-quickslot-navigation/` (proposal.md, design.md, specs/, tasks.md). Дизайн: `docs/superpowers/specs/2026-11-10-menu-quickslot-navigation-design.md`.

## Global Constraints

- Код и комментарии — на русском (инвариант 8).
- Никаких `sleep` — только `wait_until(predicate, timeout)` с таймингами из `CoreConfig` (инвариант 1).
- Verify на каждое действие; трёхзначный исход OK/FAILED/UNKNOWN (`unconfirmed` при неподтверждённом HID-эффекте) (инварианты 2, 4).
- Никаких абсолютных координат вне слоя якоря/search_area; слепых кликов нет (инварианты 3, 9).
- TDD: сначала красный тест, потом код (инвариант 6). Детерминизм: rng только с фиксированным seed, replay-входы, без живого экрана (инвариант 7).
- Вендоренный код не правится: `l2core/adapters/hid.py`, `l2core/adapters/ocr_vendor/` — кейкоды в новом модуле.
- Тесты: `python -m pytest` — единственная проверяемая команда.
- Обновление `AGENTS.md` и затронутых `docs/agents/` — часть DoD ломающих задач.

## Review Focus

1. **X — toggle**: повторное нажатие при открытом меню закроет его → шаг `open_market` обязан иметь `skip_if: [menu_open]`, шаг лавок — `when: [menu_open]`; тесты исполнителя на оба случая (Task 8).
2. **Клик ушёл, окно не появилось**: результат UNKNOWN (`unconfirmed`), не FAILED — повтор без анализа запрещён; тест через HID, бросающий `HidError` (Task 6, Task 8).
3. **Иконка не локализована** (HUD не найден в search_area / якорь меню не найден): клик не выполняется, FAILED; тест assert'ит, что `hid.click` не вызывался (Task 8).
4. **Многоопорный якорь**: первая опора не находится, вторая находится → якорь по второй; поведение market не меняется (guard-тест на эталонных кадрах) (Task 1).
5. **`expect: absent`**: «меню закрыто» = якорь меню НЕ найден; ложное срабатывание при найденном якоре обязано провалить проверку (Task 3, Task 7).
6. **Существующие вызывающие `open_element="market_pic"`** (test_api.py, test_scan_market.py, tools/) — миграция полная: ни одного потребителя старого пути (Task 9, Task 10).

---

### Task 1: Многоопорный якорь окон

**Files:**
- Modify: `l2core/detector/anchor.py:42-46` (`DEFAULT_ANCHOR_TEMPLATES`), `:159-232` (`AnchorDetector`)
- Test: `tests/test_anchor.py` (modify), `tests/test_detector_coverage.py` (modify `ANCHOR_ABSENT_OPEN`)

**Interfaces:**
- Consumes: `TemplateMatcher.match(gray, template) -> Match(x, y, score)` (`l2core/adapters/template.py`).
- Produces: `DEFAULT_ANCHOR_TEMPLATES: Mapping[str, Tuple[Tuple[str, str], ...]]` — окно -> кортеж опор `(element_id, variant)`; `AnchorDetector(..., anchor_templates=...)` принимает старый формат `(element, variant)` (нормализуется в 1-опорный кортеж) и новый кортеж опор; `find_anchor(frame, window_id) -> Optional[Anchor]` перебирает опоры, первая выше порога окна даёт Anchor (`method="template"`); поведение `frame_stage` для market не меняется.

- [ ] **Step 1: Написать failing-тесты**

В `tests/test_anchor.py`:
```python
def test_multi_support_second_support_matches():
    # Опора 1 не находится (matcher возвращает score 0), опора 2 — score 1.0
    # -> Anchor по опоре 2, x/y с учётом rect офсета элемента опоры 2.

def test_multi_support_none_matches_returns_none():
    # Обе опоры ниже порога -> None.

def test_default_anchor_templates_is_tuple_of_tuples():
    # DEFAULT_ANCHOR_TEMPLATES["market"] == (("market_main_menu", "normal"),)
```
В `tests/test_detector_coverage.py`: сценарий `ANCHOR_ABSENT_OPEN` — на досъёмке open-окна лавки (новый скрин, см. Task 11/досъёмку) якорь находится по новой опоре-заголовку; категория удаляется либо переводится на отсутствие оконного якоря вообще (по факту сырья — честно, без натяжек).

- [ ] **Step 2: Запустить — убедиться, что падают**

Run: `python -m pytest tests/test_anchor.py -v` → FAIL (кортеж опор не поддержан).

- [ ] **Step 3: Реализовать в `l2core/detector/anchor.py`**

`DEFAULT_ANCHOR_TEMPLATES` — новый формат (кортеж опор на окно; market и лавки пока по одной опоре — заголовки open-окон добавит Task 11). `_load_templates` грузит список шаблонов на окно. `find_anchor`: для `frame_stage`-окна первый этап без изменений (подтверждение первой опорой); полный поиск — перебор опор по порогу окна, первая сошедшаяся даёт Anchor с офсетом `rect` её элемента.

- [ ] **Step 4: Запустить — зелёные**

Run: `python -m pytest tests/test_anchor.py tests/test_detector_coverage.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add l2core/detector/anchor.py tests/test_anchor.py tests/test_detector_coverage.py
git commit -m "feat(anchor): многоопорный якорь окна (кортеж опор, перебор по порогу)"
```

### Task 2: Кейкод KEY_X (вне вендоренного hid.py)

**Files:**
- Create: `l2core/adapters/keycodes.py`
- Test: `tests/test_keycodes.py`

**Interfaces:**
- Produces: `KEY_X: int = 0x1B` (HID usage id клавиши X, как `KEY_ESC = 0xB1` в Keyboard.h); `KEYCODES: Mapping[str, int]` — имя клавиши из модели -> код (`{"X": KEY_X}`).

- [ ] **Step 1: Failing-тест**
```python
from l2core.adapters.keycodes import KEY_X, KEYCODES

def test_key_x_hid_usage_id():
    assert KEY_X == 0x1B
    assert KEYCODES["X"] == KEY_X
```
- [ ] **Step 2: Run → FAIL** (модуль отсутствует)
- [ ] **Step 3: Создать `l2core/adapters/keycodes.py`** — константы + docstring: модуль создан, т.к. `adapters/hid.py` вендорен и заморожен.
- [ ] **Step 4: Run → PASS**: `python -m pytest tests/test_keycodes.py -v`
- [ ] **Step 5: Commit** `feat(keycodes): KEY_X и словарь имён клавиш вне вендоренного hid.py`

### Task 3: Парсер навигационной модели

**Files:**
- Create: `l2core/model/navigation.py`
- Modify: `l2core/model/loader.py` (разбор проверок: `expect: absent` для template), `l2core/model/__init__.py` (экспорты)
- Test: `tests/test_navigation.py` (новый)

**Interfaces:**
- Consumes: `Check`, `ElementSpec`, `Rect` из `l2core.model.types`; `ModelFormatError`, `_load_yaml` из `l2core.model.loader`.
- Produces:
  - `EXPECT_ABSENT = "absent"` — значение `expect` для отрицания template-проверки.
  - `@dataclass(frozen=True) RecipeStep`: поля `action: str` (`"key"|"click"`), `element: Optional[str]`, `key: Optional[str]`, `verify: Tuple[Check, ...]`, `on_fail: Tuple[str, ...]`, `when: Tuple[Check, ...] = ()`, `skip_if: Tuple[Check, ...] = ()`.
  - `@dataclass(frozen=True) Recipe`: `id`, `opens: str`, `steps: Tuple[RecipeStep, ...]`, `on_fail: Tuple[str, ...]`, `alternate: Optional[str] = None`.
  - `@dataclass(frozen=True) NavigationModel`: `path: str`, `elements: dict` (id -> ElementSpec), `signatures: Dict[str, Tuple[Check, ...]]`, `recipes: Dict[str, Recipe]`.
  - `load_navigation(path, *, common: Optional[CommonDicts] = None) -> NavigationModel`.
- Формат YAML (закрепить в тестах и позже в FORMAT.md, Task 12):
  - элементы — как в моделях окон (`role`, `hud`+`search_area` или `rect`, `variants` с `template`/`click_point`); `rect` без якоря меню допустим только если в модели есть элемент `menu_anchor`.
  - `signatures: {menu_open: [{kind: template, element: menu_anchor, variant: normal}, ...]}` — список проверок; минимум 2 проверяет валидатор (Task 4).
  - элемент списка `verify`/`when`/`skip_if` — mapping проверки **или строка**-ссылка на сигнатуру (разворачивается в её проверки при парсинге).
  - `recipes: {open_market: {opens: market, steps: [...], on_fail: [...], alternate: ...}}`; шаг: `{action: key, key: X, verify: [...], on_fail: [...], when|skip_if: [...]}` или `{action: click, element: ..., ...}`.
  - `action: key` требует `key` и запрещает `element`; `action: click` — наоборот. Неизвестные ключи → `ModelFormatError`.
  - Проверки в navigation: `kind: template` (с опциональным `expect: absent`), `kind: pixel_probe`; `kind: ocr` → `ModelFormatError` (в навигации не поддержан).

- [ ] **Step 1: Failing-тесты** (`tests/test_navigation.py`), синтетическая модель во временном каталоге:
```python
NAV_YAML = """\
elements:
  menu_anchor:
    role: icon_sample
    hud: true
    search_area: [700, 300, 520, 400]
    variants:
      normal: {template: navigation/menu_anchor.png}
  menu_market_icon:
    role: click_target
    rect: [40, 60, 48, 48]
    variants:
      normal: {template: navigation/menu_market_icon.png, click_point: [24, 24]}
  hamburger_button:
    role: icon_sample
    hud: true
    search_area: [1800, 980, 100, 80]
    variants:
      normal: {template: navigation/hamburger.png, click_point: [50, 40]}
  quickslot_buy_icon:
    role: icon_sample
    hud: true
    search_area: [100, 900, 1700, 120]
    variants:
      normal: {template: navigation/quickslot_buy.png, click_point: [20, 20]}
signatures:
  menu_open:
    - {kind: template, element: menu_anchor, variant: normal}
    - {kind: template, element: menu_market_icon, variant: normal}
recipes:
  open_market:
    opens: market
    steps:
      - action: key
        key: X
        skip_if: [menu_open]
        verify: [menu_open]
        on_fail: [press_esc]
      - action: click
        element: menu_market_icon
        verify:
          - {kind: template, element: menu_anchor, expect: absent}
        on_fail: [press_esc, click_free_area]
    on_fail: [press_esc, click_free_area]
    alternate: open_market_via_hamburger
  open_shop_buy:
    opens: shop_buy
    steps:
      - action: key
        key: X
        when: [menu_open]
        verify:
          - {kind: template, element: menu_anchor, expect: absent}
      - action: click
        element: quickslot_buy_icon
        verify:
          - {kind: template, element: quickslot_buy_icon, variant: normal}
        on_fail: [press_esc, click_free_area]
    on_fail: [press_esc, click_free_area]
"""
```
Тесты:
```python
def test_load_navigation_recipes_and_signatures(tmp_path):
    nav = load_navigation(_write(tmp_path))   # элементы, signatures (2 проверки), оба рецепта
    assert nav.recipes["open_market"].alternate == "open_market_via_hamburger"
    assert nav.recipes["open_shop_buy"].steps[0].when == nav.signatures["menu_open"]
    assert nav.recipes["open_shop_buy"].steps[1].verify[0].expect == "absent"

def test_load_navigation_unknown_recipe_key_fails(tmp_path):
    # добавить ключ typo: 1 в рецепт -> ModelFormatError с секцией recipes

def test_load_navigation_key_step_with_element_fails(tmp_path):
    # шаг action: key с element -> ModelFormatError

def test_load_navigation_ocr_check_fails(tmp_path):
    # проверка kind: ocr в verify шага -> ModelFormatError

def test_load_navigation_signature_ref_unknown_fails(tmp_path):
    # verify: [no_such_signature] -> ModelFormatError
```
- [ ] **Step 2: Run → FAIL** (модуль отсутствует)
- [ ] **Step 3: Реализовать `l2core/model/navigation.py`** — типы + `load_navigation()`: YAML → dict (переиспользовать `ElementSpec`), разбор проверок с допуском `expect: absent` только для `kind: template` (расширить `_parse_check` в loader.py — там сейчас `expect` только для ocr), развёртывание строк-ссылок сигнатур, полные проверки формата с `ModelFormatError`.
- [ ] **Step 4: Run → PASS**: `python -m pytest tests/test_navigation.py -v`
- [ ] **Step 5: Commit** `feat(model): парсер навигационной модели navigation.yaml`

### Task 4: Валидатор навигационной модели

**Files:**
- Modify: `l2core/model/validator.py` (новая функция), `l2core/model/trace.py` (`KNOWN_ON_FAIL_STEPS` — источник словаря шагов)
- Test: `tests/test_navigation.py` (дополнить)

**Interfaces:**
- Consumes: `NavigationModel` (Task 3), `KNOWN_ON_FAIL_STEPS` из `l2core.model.trace`, `KEYCODES` из `l2core.adapters.keycodes`.
- Produces: `validate_navigation(nav: NavigationModel, assets_dir, *, known_keys: Optional[Collection[str]] = None) -> List[Violation]`; правила: `nav_signatures` (≥2 проверки на сигнатуру), `nav_step_verify` (непустой verify/on_fail шага), `nav_recipe` (непустой steps/on_fail, существование alternate, opens непуст), `nav_refs` (элементы/сигнатуры шагов существуют), `nav_on_fail_steps` (только словарные), `nav_key_known` (key из known_keys, default `KEYCODES`), `nav_templates` (шаблоны в assets), `nav_menu_anchor` (у не-hud элемента при hud-элементе меню в модели есть `menu_anchor`).

- [ ] **Step 1: Failing-тесты** — на синтетических моделях: сигнатура из 1 проверки → `nav_signatures`; шаг без verify → `nav_step_verify`; alternate на несуществующий рецепт → `nav_recipe`; шаг со ссылкой на несуществующий элемент → `nav_refs`; on_fail `["press_esc", "click_at_0_0"]` → `nav_on_fail_steps`; `key: Q` → `nav_key_known`; отсутствующий файл шаблона (assets_dir=tmp) → `nav_templates`; элемент меню без `menu_anchor` в модели → `nav_menu_anchor`. И позитив: валидная модель из Task 3 (с файлами-заглушками шаблонов) → пустой список.
- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Реализовать `validate_navigation`** рядом с `validate_model`; реюзать `Violation`.
- [ ] **Step 4: Run → PASS**: `python -m pytest tests/test_navigation.py -v`
- [ ] **Step 5: Commit** `feat(validator): инварианты навигационной модели`

### Task 5: Трассировка рецептов

**Files:**
- Modify: `l2core/model/trace.py`
- Test: `tests/test_navigation.py` (дополнить)

**Interfaces:**
- Produces: `trace_navigation(models: Mapping[str, UiModel], nav: NavigationModel) -> List[Violation]`:
  - на каждое окно из `models` обязан существовать рецепт `open_<window_id>` с `opens == <window_id>` (иначе violation `nav_window_unreachable`);
  - в модели окна обязано быть состояние `recover_closed` (violation `nav_recover_state_missing`);
  - у рецепта непустой `on_fail` или задан `alternate` — выход неуспеха в recover (violation `nav_recipe_no_exit`);
  - шаги on_fail рецептов — только `KNOWN_ON_FAIL_STEPS` (violation `nav_on_fail_steps`, паритет с трассировкой переходов).

- [ ] **Step 1: Failing-тесты**: синтетические UiModel (минимум: `states={"recover_closed": ...}`) + nav из Task 3: полный набор → пусто; удалить `open_shop_buy` → `nav_window_unreachable`; рецепт с `on_fail: []` и без alternate → `nav_recipe_no_exit`.
- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Реализовать `trace_navigation`** в trace.py.
- [ ] **Step 4: Run → PASS**
- [ ] **Step 5: Commit** `feat(trace): трассировка навигационных рецептов`

### Task 6: Примитив press_key

**Files:**
- Create: `l2core/primitives/key.py`
- Test: `tests/test_press_key.py`

**Interfaces:**
- Consumes: `VerifySpec`, `CheckReport` из `l2core.primitives.click`; `wait_until` из `l2core.primitives.wait`; `KEYCODES` из `l2core.adapters.keycodes`; `_run_on_fail` из `l2core.primitives.click` (приватный, но общий для примитивов — импорт на уровне модуля ок).
- Produces:
  - `EvaluateChecks = Callable[[Tuple[Check, ...], np.ndarray], CheckReport]` — оценка проверок по кадру **без якоря окна** (меню — HUD-объект; якоря окна торговли при закрытом окне нет).
  - `@dataclass(frozen=True) KeyResult`: `key_id, success, attempts, cancelled=False, unconfirmed=False, on_fail_executed=False, failures=()` + свойство `needs_recover` (как `ClickResult`).
  - `class PressKeyError(Exception)`.
  - `press_key(key_id: str, *, verify: VerifySpec, evaluate: EvaluateChecks, screen, hid, activate, config, on_fail: Tuple[str, ...] = (), cancelled=None, rng=None, clock=time.monotonic, sleep=time.sleep) -> KeyResult`.
- Семантика: `activate()` (отмена → KeyResult cancelled) → цикл `1 + config.timing.click_retry_count` попыток: `hid.key_press(KEYCODE)` (неизвестный key_id → PressKeyError; `HidError` → KeyResult UNKNOWN `unconfirmed=True`, прерыв) → `wait_until(evaluate(verify.checks, fresh_frame), timeout=config.timing.wait_default_timeout, interval=config.timing.verify_interval)` → успех → KeyResult OK. Исчерпание → повторный `activate()` → `_run_on_fail(on_fail, ...)` → KeyResult FAILED, `on_fail_executed`. Отмена внутри ожидания → cancelled.

- [ ] **Step 1: Failing-тесты** (фейки по образцу tests/test_click.py): OK-путь (evaluate возвращает passed после первого же кадра; assert `key_press` вызван с `0x1B`); verify-таймаут (evaluate всегда failed) → FAILED, `on_fail_executed=True`, ESC ушёл в HID; `HidError` → UNKNOWN, `unconfirmed=True`, on_fail не выполнялся; отмена до нажатия → cancelled, `key_press` не вызывался; неизвестный key_id → PressKeyError. Часы/сон — фейки из паттерна test_click.py (детерминизм).
- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Реализовать `l2core/primitives/key.py`** по контракту докстринга click.py (тот же стиль, hover-логики нет).
- [ ] **Step 4: Run → PASS**: `python -m pytest tests/test_press_key.py -v`
- [ ] **Step 5: Commit** `feat(primitives): press_key 5.9 — нажатие клавиши с verify и трёхзначным исходом`

### Task 7: Навигационный оценщик проверок

**Files:**
- Create: `l2core/detector/navchecks.py`
- Test: `tests/test_navchecks.py`

**Interfaces:**
- Consumes: `NavigationModel`, `Check` (Task 3); `TemplateMatcher` (`l2core/adapters/template.py`); `CheckReport` из `l2core.primitives.click`.
- Produces: `MENU_ANCHOR_ID = "menu_anchor"` (соглашение id якоря меню); `build_nav_evaluator(nav: NavigationModel, matcher: TemplateMatcher, assets_dir, *, nav_threshold: float = 0.9, probe_tolerance: int = 16) -> EvaluateChecks`.
- Семантика оценки набора проверок на кадре (passed = сошлись ВСЕ): для каждой проверки находит элемент по `check.element`; hud-элемент — template matching шаблона `check.variant` (первый вариант, если не задан) по `search_area` на сером кадре, порог `nav_threshold`; не-hud элемент — сначала локация `menu_anchor` (тот же matching по его `search_area`), без якоря проверка провалена; затем matching шаблона элемента в `anchor + rect` (точечная область). `expect == "absent"` инвертирует итог проверки. `pixel_probe` — только для не-hud элементов: точка `anchor + at`, допуск цвета `probe_tolerance` (евклидово расстояние RGB). Провал элемента добавляет причину в `CheckReport.failures`.

- [ ] **Step 1: Failing-тесты**: синтетический кадр (numpy-массив) + фейковый matcher (контролируемый score по шаблону): hud-проверка найдена → passed; `expect: absent` при найденном → failed, при не найденном → passed; якорь меню не найден → проверка иконки failed (причина про якорь); pixel_probe с цветом в допуске/вне → passed/failed.
- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Реализовать `l2core/detector/navchecks.py`** (шаблоны грузить cv2.imread grayscale, как `AnchorDetector._load_templates`).
- [ ] **Step 4: Run → PASS**: `python -m pytest tests/test_navchecks.py -v`
- [ ] **Step 5: Commit** `feat(detector): оценщик проверок навигационной модели (HUD + якорь меню, expect absent)`

### Task 8: Исполнитель рецептов

**Files:**
- Create: `l2core/primitives/navigation.py`
- Test: `tests/test_recipes.py`

**Interfaces:**
- Consumes: `NavigationModel`, `Recipe`, `RecipeStep` (Task 3); `press_key`, `KeyResult`, `EvaluateChecks`, `PressKeyError` (Task 6); `_run_on_fail` (click.py); `wait_until` (wait.py); `KEYCODES` (keycodes.py).
- Produces:
  - `@dataclass(frozen=True) RecipeResult`: `recipe_id, success, attempts, cancelled=False, unconfirmed=False, on_fail_executed=False, failures=()` + `needs_recover`.
  - `class RecipeError(Exception)`.
  - `run_recipe(recipe_id: str, *, nav_model: NavigationModel, evaluate: EvaluateChecks, hud_locator, screen, hid, cursor_pos, activate, config, free_area_point=None, cancelled=None, rng=None, clock=time.monotonic, sleep=time.sleep) -> RecipeResult`. `hud_locator` — тот же прототип, что в click.py: `(ElementSpec, Optional[str], np.ndarray) -> Optional[Tuple[int, int]]`.
- Семантика: рецепт есть в модели (нет → RecipeError). Для каждого шага на свежем кадре (FrameTransaction): `skip_if` сошлись → шаг пропущен (успех); `when` задан и не сошлось → шаг пропущен (успех).
  - `key`-шаг → `press_key(step.key, verify=VerifySpec(step.verify), evaluate=evaluate, ..., on_fail=step.on_fail)`; неуспех → зафиксировать, идём дальше; UNKNOWN (`unconfirmed`) → прерыв весь рецепт с UNKNOWN.
  - `click`-шаг по HUD-элементу: цикл `1 + config.timing.click_retry_count`: точка = `hud_locator(element, None, frame)`; `None` → попытка потрачена (failures «не локализован в search_area», клика нет); иначе `_move_to` + `hid.click(BTN_LEFT, 1, hold_ms)` → `wait_until(evaluate(step.verify, fresh))` → успех → следующий шаг; `HidError` → прерыв с UNKNOWN.
  - `click`-шаг по элементу меню (не-hud): точка = `hud_locator(nav_model.elements["menu_anchor"], "normal", frame)`; якорь не найден → попытка потрачена (меню не открыто); точка иконки = якорь + `rect` + `click_point` (первый заданный, иначе центр); далее как HUD-шаг.
  - Шаг не подтверждён после всех попыток → его on_fail уже выполнил press_key (key-шаг) или выполняется `_run_on_fail(step.on_fail)` здесь (click-шаг, после повторного activate + проверки статуса — как в click.py), зафиксировать, идём дальше.
  - Все шаги успешны → RecipeResult OK. Есть провалы → `on_fail` рецепта (`_run_on_fail`), затем `alternate`: `run_recipe(alternate, ...)` рекурсивно, итог alternate с префиксом failures основного. Нет alternate → RecipeResult FAILED, `needs_recover`.

- [ ] **Step 1: Failing-тесты** (фейки как в test_click.py + nav-модель из NAV_YAML Task 3, matcher-фейки через evaluate):
  1. `open_market` happy path: key X → evaluate OK; click menu_market_icon (hud_locator якоря вернул точку) → verify OK → RecipeResult success, `hid.key_press` с `0x1B`, один клик.
  2. `skip_if`: evaluate сошёлся уже на входе (меню открыто) → `key_press` НЕ вызывался, шаг клика выполнился.
  3. `when` у лавок: evaluate не сошёлся → key-шаг пропущен (key_press не вызывался), клик по quickslot выполнен.
  4. HUD-иконка не локализована (hud_locator → None) → RecipeResult FAILED, **assert `hid.click` не вызывался**, failures содержит «не локализован».
  5. Клик ушёл, verify не сошёлся, окно ни разу не подтверждено → после retry/alternate попыток RecipeResult FAILED с `on_fail_executed=True`; `HidError` на клике → UNKNOWN, `unconfirmed=True`, продолжения рецепта нет.
  6. Неудача основного → alternate `open_market_via_hamburger` исполнился (клик по hamburger_button), итог success alternate.
  7. Отмена до шага → cancelled, HID чист.
- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Реализовать `l2core/primitives/navigation.py`** по контракту выше (единый стиль с click.py: docstring-контракт, инъекции, FrameTransaction).
- [ ] **Step 4: Run → PASS**: `python -m pytest tests/test_recipes.py -v`
- [ ] **Step 5: Commit** `feat(primitives): исполнитель рецептов навигации (key/click шаги, when/skip_if, alternate)`

### Task 9: Wiring и API на рецептах

**Files:**
- Modify: `l2core/api/wiring.py` (`ApiWiring`, `build_wiring`), `l2core/api/core.py` (`open_market`, `open_shop_buy`, `open_shop_sell`; удалить `_open_via_hud`), `l2core/engine` и `tools/live_dry_run.py` (точки сборки wiring — по факту grep `build_wiring`)
- Test: `tests/test_api.py` (modify), `tests/test_wiring`-тесты на загрузку navigation (где сейчас гоняется build_wiring — по факту)

**Interfaces:**
- Consumes: `load_navigation` (Task 3), `build_nav_evaluator` (Task 7), `run_recipe`, `RecipeResult` (Task 8), `to_outcome` (outcome.py).
- Produces: в `ApiWiring` новые поля `navigation: Optional[NavigationModel] = None` и `nav_evaluate: Optional[EvaluateChecks] = None`; `build_wiring(..., navigation_path: Optional[Path] = None)` — грузит `ui_model/navigation.yaml` (если есть), строит оценщик, включает navigation-модель в загрузку HUD-шаблонов (`_load_hud_templates`); `L2CoreAPI.open_market(cancelled=None) -> NavigationResult` — рецепт `open_market`; `open_shop_buy(verify=None, cancelled=None)` / `open_shop_sell(...)` — рецепты `open_shop_buy`/`open_shop_sell`, `verify` — override: заменяет verify последнего шага рецепта (`dataclasses.replace` на копии рецепта). Navigation без wiring → ApiError (fail fast). `_open_via_hud` удалён.
- Адаптер: `_navigation_from_recipe(r: RecipeResult) -> NavigationResult` (поля 1:1, `outcome=to_outcome(r)` — duck-typing по success/cancelled/unconfirmed).

- [ ] **Step 1: Failing-тесты** (tests/test_api.py, паттерн существующих навигационных тестов): open_market через фейковый wiring с navigation (рецепт исполняется фейковым evaluate/hud_locator) → OK; FAILED-исход рецепта → NavigationResult FAILED с failures; UNKNOWN → `unconfirmed`; `verify`-override меняет проверки последнего шага; wiring без navigation → ApiError.
- [ ] **Step 2: Run → FAIL** (в т.ч. существующие тесты open_market/open_shop_* на `_open_via_hud` — красные до миграции)
- [ ] **Step 3: Реализовать** изменения wiring/core; удалить `_open_via_hud`; мигрировать всех вызывающих (grep `open_element=`, `market_pic`, `open_buy_trade`, `open_sale_trade`, `_open_via_hud` по `l2core/`, `tests/`, `tools/`).
- [ ] **Step 4: Run → PASS**: `python -m pytest tests/test_api.py -v`, затем полный `python -m pytest`.
- [ ] **Step 5: Commit** `feat(api): open_* на рецептах навигации, _open_via_hud удалён`

### Task 10: Сценарий scan_market через рецепт

**Files:**
- Modify: `l2core/scenarios/scan.py` (`scan_market`, `_resolve_open_verify`, `_click` — открытие), `tests/test_scan_market.py`
- Test: `tests/test_scan_market.py` (modify)

**Interfaces:**
- Produces: `scan_market(..., open_recipe: str = "open_market", recipe_runner: Optional[Callable[[str], RecipeResultProtocol]] = None, ...)` — параметры `open_element`/`open_from` **удалены**; открытие: `recipe_runner(open_recipe)`; `None` → ScanError (fail fast); результат с `needs_recover`/`cancelled` → прежняя семантика ScanResult. `RecipeResultProtocol` — структурный (success/cancelled/failures/needs_recover), импортировать `RecipeResult` допустимо (сценарии зависят от примитивов вниз). API `scan_market`-обёртка (если есть в core.py) передаёт `lambda: run_recipe(...)` wiring'а.

- [ ] **Step 1: Failing-тест**: открытие идёт через `recipe_runner` (фейк записывает имя рецепта, возвращает success) → обход продолжается; `recipe_runner` вернул needs_recover → ScanResult с failures, кликов open-element не было; `recipe_runner=None` → ScanError.
- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Реализовать** в scan.py; удалить `_resolve_open_verify` и open-часть `_click`.
- [ ] **Step 4: Run → PASS**: `python -m pytest tests/test_scan_market.py -v`, полный `python -m pytest`.
- [ ] **Step 5: Commit** `feat(scenarios): scan_market открывает обзор через рецепт навигации`

### Task 11: Отгруженная модель: navigation.yaml, чистка HUD, recover_closed, якоря open-окон

**Files:**
- Create: `ui_model/navigation.yaml`, `assets/navigation/*.png`
- Modify: `ui_model/market.yaml`, `ui_model/shop_buy.yaml`, `ui_model/shop_sell.yaml`, `base_screenshot/scr/meta.json` (rect'ы досъёмок), `tests/test_model.py` / guard-тесты (по факту упавших)
- Test: guard-тесты существующего набора + новый `tests/test_navigation.py::test_shipped_navigation_validates`

**Interfaces:**
- Consumes: `validate_navigation`, `trace_navigation`, `load_navigation` (Tasks 3–5), досъёмки `scr_menu_closed.png` / `scr_menu_open.png` (в репо), досъёмки open-окон лавок и quick-slot (Task M1/M2 ниже).
- Produces: отгруженная `ui_model/navigation.yaml` (проходит `validate_navigation` и `trace_navigation` против отгруженных моделей окон); `market.yaml` без `market_pic` и перехода открытия; `recover_closed` с ≥2 сигнатурами, подтверждающимися на `scr_menu_closed` (например: template `hamburger_button` + template `quickslot_buy_icon` — оба видны на голом экране; выбор — разметчик по кадру); `DEFAULT_ANCHOR_TEMPLATES` пополнен опорами-заголовками open-окон лавок (Task 1 формат).

- [ ] **Step 1: Разметить meta.json** rect'ами: гамбургер (`scr_menu_closed`), якорь меню + иконка обзора (`scr_menu_open`), иконки quick-slot и open-заголовки лавок (досъёмки M1/M2). Прогнать `python -m l2core.model.converter` (кропы → `assets/navigation/`, diff-отчёт зафиксирован в выводе).
- [ ] **Step 2: Написать failing guard-тесты**: `test_shipped_navigation_validates` (валидатор+трассировка по отгруженным файлам); `recover_closed` подтверждается детектором на `scr_menu_closed`; отсутствие `market_pic`/`open_buy_trade`/`open_sale_trade` в моделях окон.
- [ ] **Step 3: Собрать `navigation.yaml` и правки моделей окон** по измеренным rect'ам (пороги template matching — замер по кадрам, как в отчётах 3.3/3.4).
- [ ] **Step 4: Run → PASS**: `python -m pytest` полностью зелёный.
- [ ] **Step 5: Commit** `feat(model): отгружена навигационная модель меню/quick-slot, HUD-путь удалён`

### Task 12: KNOWN_GAPS, FORMAT.md, AGENTS.md, docs/agents

**Files:**
- Modify: `ui_model/KNOWN_GAPS.yaml`, `ui_model/FORMAT.md`, `AGENTS.md`, `docs/agents/primitives.md`, `docs/agents/api-runtime.md`, `docs/agents/model-reading.md` (по факту затронутого)
- Test: `tests/test_validator.py::test_shipped_models_match_known_gaps` (guard — должен остаться зелёным)

- [ ] **Step 1: Обновить KNOWN_GAPS.yaml** по правилам файла: п. 11 — закрыт (правка модели; история сохранена, стиль п. 4); п. 6 — закрыт многоопорным якорем; п. 3 — дополнена развязка (межокнонные переходы открытия — рецепты navigation.yaml).
- [ ] **Step 2: FORMAT.md** — секция «Навигационная модель» (схема из Task 3, соглашение `open_<window>` ↔ `recover_closed`, `expect: absent`, `when`/`skip_if`, alternate).
- [ ] **Step 3: AGENTS.md** — структура пакета (`model/navigation.py`, `adapters/keycodes.py`, `detector/navchecks.py`, `primitives/key.py` 5.9, `primitives/navigation.py`), валидатор/трассировка navigation.yaml, досъёмки. `docs/agents/` — примитив press_key и исполнитель рецептов (primitives.md), навигация по рецептам (api-runtime.md).
- [ ] **Step 4: Run → PASS**: `python -m pytest` (guard KNOWN_GAPS зелёный).
- [ ] **Step 5: Commit** `docs: навигационная модель, press_key, рецепты — AGENTS.md/FORMAT.md/KNOWN_GAPS/docs`

## Ручные задачи (вне CI, не блокируют кодовые задачи)

- **M1 (блокирует часть Task 11)**: инструментальная досъёмка open-окон лавок без перекрытий + кропы заголовков «Торговая лавка» (развязка KNOWN_GAPS п. 6).
- **M2 (блокирует рецепты лавок)**: инструментальная досъёмка quick-slot: открыть shop_buy и shop_sell кликом по своим иконкам панели, снять панель и подтверждённые открытия. Рецепт `open_market` этого не требует.
- **M3**: живой dry-run (9.4): прогон `tools/live_dry_run.py` с `--model` — проба `open_market` через рецепт; после M2 — лавок.

## Self-Review

1. **Spec coverage**: navigation-recipes (Tasks 3–8, 11), action-primitives MODIFIED (Tasks 2, 6), ui-model MODIFIED (Tasks 9–11), window-state-detection MODIFIED (Task 1) — всё покрыто. Ручные M1–M3 — из tasks.md ченджа.
2. **Step scan**: шаги дают ровно одно действие; тела тестов приведены там, где сигнатуры их не определяют.
3. **Type consistency**: `RecipeStep`/`Recipe`/`NavigationModel` (Task 3) → `validate_navigation`/`trace_navigation` (4, 5) → `EvaluateChecks` (6) → `build_nav_evaluator` (7) → `run_recipe`/`RecipeResult` (8) → wiring/API (9) → scan (10) — имена и поля согласованы; `KeyResult`/`RecipeResult` повторяют поля `ClickResult` для `to_outcome`.
4. **Review Focus**: каждый из 6 пунктов имеет владельца — Tasks 1, 3, 6, 8, 9, 10.
5. **Proportion**: план ~в 1.5 раза длиннее спеки — обосновано объёмом интерфейсных решений (парсер/исполнитель); тел кода в плане нет.
