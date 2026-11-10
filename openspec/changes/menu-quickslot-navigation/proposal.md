# Proposal: menu-quickslot-navigation

## Why

Живой прогон 9.4 (2026-11-10) показал, что путь открытия окон торговли через
HUD-иконки сломан на живой игре: `open_market` вернул FAILED, потому что
`market_pic` на голом экране отсутствует (KNOWN_GAPS п. 11). Ответ владельца
(Q28): «Обзор лавок» открывается только через игровое «Меню» (клавиша X /
гамбургер), лавки — через иконки панели быстрого доступа (quick-slot). Сырьё
по меню доснято инструментально, элементы «Меню» и механика открытия в модели
отсутствуют — открытие окон невозможно без навигации «Меню/quick-slot».

Источник: одобренный дизайн
`docs/superpowers/specs/2026-11-10-menu-quickslot-navigation-design.md`
(builder-check пройден, замечания разрешены правками документа).

## What Changes

- Новая навигационная модель `ui_model/navigation.yaml` (не модель окна):
  HUD-элементы (гамбургер, иконки quick-slot), якорь «Меню» и элементы
  внутри него 1:1 от якоря, сигнатуры состояния `menu_open`, декларативные
  рецепты открытия окон (шаги `key`/`click` с verify/on_fail, условные
  пометки `when`/`skip_if`, отрицание `expect: absent`, `alternate`, поле
  `opens`).
- Новый примитив `press_key` (5.9): активация окна, `hid.key_press`,
  блокирующий `wait_until` на verify, retry → on_fail → recover; кейкод
  `KEY_X` в HID-адаптере.
- Исполнитель рецептов: резолв элементов (HUD — template matching по
  search_area; элементы меню — по якорю меню), исполнение шагов через
  примитивы, трёхзначный исход, слепых кликов нет, неудача → on_fail →
  alternate → общий recover.
- API `open_market` / `open_shop_buy` / `open_shop_sell` переведены на
  рецепты (verify перестаёт быть обязательным параметром у лавок —
  override сохранён); метод `_open_via_hud` удалён.
- Предшественник в скоупе: многоопорный якорь окон (развязка KNOWN_GAPS
  п. 6) — детектор держит кортеж опор, досъёмка open-окон лавок, импорт
  заголовков в модели.
- Чистка моделей: удалены `market_pic`, `open_buy_trade`, `open_sale_trade`
  и переход `recover_closed --click market_pic--> list__tab_home`; сигнатуры
  `recover_closed` пересобраны; HUD-кропы quick-slot перенесены в
  `navigation.yaml`; KNOWN_GAPS п. 11 закрыт, п. 3 дополнен развязкой.
- Инструментальные досъёмки: идентификация иконок quick-slot (какая лавка
  каким слотом открывается) с подтверждёнными открытиями окон.
- Кейкоды цифр/F-клавиш не добавляются (раскладка quick-slot свободная,
  открытие лавок — визуальным поиском иконки и кликом).

## Capabilities

### New Capabilities

- `navigation-recipes`: навигационная модель `ui_model/navigation.yaml`
  (элементы меню/quick-slot, сигнатуры `menu_open`, рецепты открытия),
  парсер/валидатор её инвариантов, исполнитель рецептов с
  OK/FAILED/UNKNOWN, участие рецептов в трассировке графа (6.5).

### Modified Capabilities

- `action-primitives`: новый примитив `press_key` (нажатие клавиши с
  verify/retry/on_fail, трёхзначный исход), кейкод `KEY_X` в HID.
- `ui-model`: HUD-элементы `market_pic`/`open_buy_trade`/`open_sale_trade`
  удалены из моделей окон; `recover_closed` пересобран без HUD-иконок;
  `navigation.yaml` — новый вид файла модели (не окно).
- `window-state-detection`: многоопорный якорь (`anchor_templates: window ->
  [опоры]`, перебор по порогу окна) — предшественник рецептов лавок.

## Impact

- **Репозиторий**: `E:\L2-CRNN-core-market` (модель, код ядра, тесты,
  `ui_model/FORMAT.md`, `AGENTS.md`, `docs/agents/`).
- **Артефакты**: новые кроп-шаблоны в `assets/navigation/` (версионируются);
  досъёмки в `base_screenshot/scr/` + meta.json (сырьё).
- **Тесты**: парсер/валидатор/трассировка `navigation.yaml`, примитив
  `press_key`, исполнитель рецептов на replay-кадрах, guard-тесты
  KNOWN_GAPS; детерминизм (replay-входы, без живого экрана).
- **Приёмка**: проба `open_market`/`open_shop_*` живого dry-run (9.4)
  переводится на рецепты; до инструментальной досъёмки quick-slot
  рецепты лавок заблокированы (рецепт market — нет).
- **Совместимость**: публичные сигнатуры API сохранены (verify у
  `open_shop_*` — необязательным, не удалённым).
