# Tasks: menu-quickslot-navigation

## 1. Предшественники

- [ ] 1.1 Многоопорный якорь: `anchor_templates: window -> [опоры]` в детекторе, `find_anchor` перебирает опоры по порогу окна; guard-тест: поведение market на эталонных кадрах не изменилось (TDD: сначала тест кортежа опор)
- [ ] 1.2 Инструментальная досъёмка: чистые open-окна shop_buy/shop_sell (без перекрытий) + кропы заголовков; импорт в модели через конвертер (diff-отчёт, ручные правки не трогаются)
- [ ] 1.3 Coverage-тест `ANCHOR_ABSENT_OPEN` обновлён: open-окна лавок на досъёмке якорь находят
- [ ] 1.4 Инструментальная досъёмка quick-slot: открыть shop_buy и shop_sell кликом по своим иконкам панели, снять кадры панели (шаблоны `quickslot_*_icon`) и подтверждённые открытия окон (verify рецептов); meta.json обновлён

## 2. Навигационная модель

- [ ] 2.1 Парсер `navigation.yaml` (секции `elements`, `signatures`, `recipes`; неизвестные ключи → `ModelFormatError` с файлом/секцией/деталью); модель окон его не грузит (TDD)
- [ ] 2.2 Валидатор `navigation.yaml`: ≥2 сигнатуры `menu_open`; непустые verify/on_fail на шагах и рецептах; шаги ссылаются на существующие элементы; on_fail — только словарные шаги исполнителя; `alternate` существует; шаблоны в `assets/navigation/`; элементы меню привязаны к `menu_anchor`; кейкод шага `key` — известен HID (TDD)
- [ ] 2.3 Трассировка (6.5): рецепты участвуют в обходе — окно достижимо из `recover_closed` через `open_<window>` (поле `opens`), неуспех рецепта имеет выход в recover (TDD)
- [ ] 2.4 Разметка `navigation.yaml`: элементы (гамбургер, якорь меню, иконка обзора в меню, иконки quick-slot из досъёмки 1.4), сигнатуры `menu_open`, рецепты `open_market` (+`open_market_via_hamburger`), `open_shop_buy`, `open_shop_sell` с нормализацией меню (`when`/`skip_if`, `expect: absent`)
- [ ] 2.5 Разметка meta.json: rect'ы гамбургера/якоря меню/иконок (из `scr_menu_closed`/`scr_menu_open` и досъёмок); кропы в `assets/navigation/`; `python -m l2core.model.converter` (diff-отчёт зафиксирован)
- [ ] 2.6 `ui_model/FORMAT.md`: секция навигационной модели (гранулярность, схема рецептов, семантика элементов)

## 3. Примитивы и исполнение

- [ ] 3.1 HID: кейкод `KEY_X` (TDD: fake hid получает корректный код)
- [ ] 3.2 Примитив `press_key` (5.9): активация, `key_press`, `wait_until` на verify, retry → on_fail → recover; FAILED при verify-таймауте, UNKNOWN при `HidError` (`unconfirmed`) (TDD)
- [ ] 3.3 Исполнитель рецептов: резолв элементов (HUD — template matching по `search_area`; элементы меню — по якорю `menu_anchor`), условные пометки `when`/`skip_if` на текущем кадре, исполнение шагов через примитивы; иконка не найдена → FAILED без клика; клик ушёл, окно не появилось → UNKNOWN; неудача шага → on_fail → следующий шаг; исчерпание → on_fail рецепта → alternate → общий recover (TDD, replay-кадры)

## 4. API и сценарии

- [ ] 4.1 API: `open_market`/`open_shop_buy`/`open_shop_sell` исполняют рецепты; verify у лавок необязателен (из рецепта, параметр — override); `_open_via_hud` удалён (TDD: фейковый исполнитель, исходы OK/FAILED/UNKNOWN)
- [ ] 4.2 Сценарий scan_market: путь открытия через рецепт (вместо `market_pic`) (TDD)
- [ ] 4.3 Живой dry-run (9.4): проба `open_market` через рецепт; рецепты лавок — после досъёмки 1.4 (ручной прогон, вне CI)

## 5. Чистка моделей и пробелов

- [ ] 5.1 `market.yaml`: удалить `market_pic` и переход `recover_closed --click market_pic--> list__tab_home`; пересобрать сигнатуры `recover_closed` без HUD-иконок (TDD: `recover_closed` подтверждается на `scr_menu_closed`)
- [ ] 5.2 `shop_buy.yaml`/`shop_sell.yaml`: удалить HUD-элементы `open_buy_trade`/`open_sale_trade`; HUD-кропы quick-slot перенесены в `navigation.yaml`
- [ ] 5.3 `KNOWN_GAPS.yaml`: п. 11 закрыт правкой модели (запись обновлена по правилам файла); п. 3 дополнен развязкой (межоконные переходы открытия — рецепты); guard-тест KNOWN_GAPS зелёный
- [ ] 5.4 Валидатор/парсер/трассировка прогнаны по всем моделям: `python -m pytest` зелёный

## 6. Документация (DoD ломающих задач)

- [ ] 6.1 `AGENTS.md`: команды/структура/инварианты дополнены (navigation.yaml, press_key 5.9, исполнитель рецептов)
- [ ] 6.2 `docs/agents/primitives.md`: примитив press_key; `docs/agents/api-runtime.md`: навигация через рецепты; заметки по модели — navigation.yaml
