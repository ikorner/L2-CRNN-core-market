# Сценарии (задачи 6.1–6.4)

> Детальный справочник по слою `l2core/scenarios/`, вынесенный из
> AGENTS.md (прогрессивная загрузка: AGENTS.md — ядро, этот файл — детали
> по слою). Правится вместе с кодом: обновление затронутых заметок
> `docs/agents/` — часть DoD ломающих задач (см. «Правила изменений» в
> AGENTS.md).

## scan_market и чтение строк таблицы (задача 6.1, spec market-scenarios,
spec ocr-reading «Чтение строк таблицы с Row и дедупликацией»)

`l2core/model/rows.py` — чтение строк списка (слой модели, как reading.py):
`Row` (frozen: cells по именам колонок модели [CellValue: text +
разобранное value по грамматике колонки], screen_y — y кромки строки в
КООРДИНАТАХ ОКНА от якоря [дедуп между кадрами в координатах окна: окно
может сдвинуться, слот — нет; абсолютный y = anchor.y + screen_y],
icon_hash — pHash иконки предмета, partial — краевые слоты вьюпорта [0 и
max_rows−1] потенциально обрезаны: без позиции скролла консервативно);
`read_rows(frame, ...)` → `RowsResult(rows, snapshot_id)`: нарезка
TableSegmentation (4.3), непустые ячейки — read_confident (4.5) с
грамматикой колонки, пустая ячейка — честный CellValue("", None) без OCR,
иконка — pHash левой части ячейки item до зоны текста; snapshot_id —
детерминированный id кадра (sha1 по байтам, идентичные кадры → тот же
id; рынок меняется — стратегия сверяет). pHash — numpy-only DCT (схема
imagehash.phash: серый BT.601 -> 32x32 ближайший сосед -> DCT-II -> низкие
8x8 -> биты >= медианы -> int64), сравнение — Хэмминг, дефолтный порог
DEFAULT_HAMMING_THRESHOLD=6/64 (допуск к мелким пережаткам отрисовки,
пины tests/test_rows.py; threshold=0 — строгое равенство). Дедуп между
кадрами скролла — `dedup_rows(rows, price_column="price",
hamming_threshold=...)`: ключ icon_hash (по Хэммингу) + цена + y, порядок
первого появления; колонки цены нет — ключом служат тексты всех ячеек.
read_rows ТОЛЬКО читает кадр: ни HID, ни скролла (обзор — разведка цен).

`l2core/scenarios/scan.py` — `scan_market(points, ...)`: КОМПОЗИЦИЯ
примитивов 5.x без внутреннего состояния (spec: состояние — у детектора
через инъекции, не в «голове» сценария). Обход параметризуется списком
точек `ScanPoint(verify, tab=None, filters=(), search=None)` извне (params
табов/фильтров частично не размечены — KNOWN_GAPS): открытие обзора
(recipe_runner: рецепт навигации open_market — menu-quickslot-navigation;
runner не задан — ScanError, открытие вслепую запрещено) ->
для каждой точки: таб -> фильтры (click_element, verify точки) -> поиск
(enter_text; OCR-регион поля в market.yaml не размечен — search_region/
search_field_element/search_button_element передаются снаружи, подшиваются
в копию модели; точка с search без search_region — ScanError) -> цикл
чтения: read_rows -> дедуп -> scroll_position (5.5) -> scroll_list вниз
до признака конца из модели (disabled-стрелка «вниз»), предохранитель
max_scrolls. Результат: ScanResult(points: PointScan[агрегат строк +
снимки RowsResult], cancelled, failures): needs_recover примитива фиксирует
отказ обхода (исполнение recover — 6.4, вызывающий слой), отмена — флаг.
ОТКРЫТИЕ ЧУЖИХ ЛАВОК ОТСУТСТВУЕТ (non-goal): guard-тесты
tests/test_scan_market.py::TestNoLavkaOpeningGuard следят, что код
сценария не содержит иконок открытия лавок и прямых hid.click; клики
только по element id точек обхода (табы/фильтры/поиск/HUD-открытие).

## setup_sell и start_sell/cancel_sell (задача 6.2, spec
market-scenarios «Сценарий выставления лавки продажи», инварианты 2/4/9)

`l2core/scenarios/setup_sell.py` — выставление лавки продажи:
`setup_sell(items)` со списком `SellItem(slot, price)`, `start_sell()`,
`cancel_sell()`; композиция примитивов 5.x без внутреннего состояния.

- **Слоты из геометрии сетки** (контракт design.md: координаты ячейки
  генерируются из геометрии сетки лавки, НЕ размечаются вручную):
  параметрическая `GridGeometry(origin, pitch, cell, cols, rows)` —
  опциональная секция `grid_geometry` модели окна (FORMAT.md, loader,
  валидатор `grid_in_window`), константы от якоря окна; генератор
  `slot_point(grid, index)` — центр ячейки, нумерация row-major.
  Источник констант: параметр `grid` сценария, иначе секция модели;
  нет ни того ни другого — SetupSellError (R4: не выдумывать). В
  отгруженной shop_sell секция не размечена (мета поле `slot` не
  описывает — KNOWN_GAPS п.9, Q26). Вне диапазона слота, повтор слота,
  цена < 1 — SetupSellError (fail fast).
- **setup_sell**: открытие лавки — рецепт навигации (recipe_runner,
  open_recipe по умолчанию "open_shop_sell"; open_recipe=None — лавка
  открыта вызывающим слоем) → для каждого item: свежий якорь (слепой drag
  запрещён) + target = якорь + slot_point → drag_item(verify=drag_verify)
  → enter_price (OCR-регион поля цены в shop_sell не размечен —
  price_region подшивается в копию модели, как search_region в 6.1; без
  него SetupSellError) → итоговый verify (final_verify, свежий кадр).
  Источник drag'а — точка предмета в ИНВЕНТАРЕ (вне окон модели):
  даёт инъекция source_locator(item), заново на каждый предмет.
  Отказ примитива — SetupSellResult.needs_recover (recover — 6.4,
  вызывающий слой), отмена — cancelled=True.
- **start_sell() — потенциально необратимое**: re-verify ПЕРЕД кликом
  (FrameTransaction.reverify, 3.6: свежий grab + confirm_checks по свежему
  якорю); не сошёлся — клик по кнопке «Начать продажу» НЕ выполняется
  (no blind click на необратимое), StartSellResult(reverified=False,
  outcome=None). Страж необратимых (5.7): guard.before/after вокруг
  клика — UNKNOWN фиксирует блокировку повтора до явного
  confirm_after_unknown стратегией (IrreversibleRepeatError проброшена);
  outcome = to_outcome(ClickResult) в StartSellResult.
- **cancel_sell()**: клик по кнопке отмены (элемент модели
  cansel_sell_button, через click_element с verify); element_id=None —
  честный fallback на ESC: пролог (activate OK + статус found, иначе
  исключение, HID не трогается) → ESC → verify на свежем кадре → retry
  → on_fail[].
- Тесты: tests/test_setup_sell.py (скриптованные экран/HID/OCR:
  набор цифр key_press'ами меняет «поле цены» экрана; пины slot-центров,
  re-verify блокировки клика, стража UNKNOWN, ESC-fallback, отмены,
  needs_recover); loader/валидатор grid_geometry — tests/test_model.py,
  tests/test_validator.py.

## monitor_shop — мониторинг своей лавки и событие sold (задача 6.3, spec
market-scenarios «Сценарий мониторинга своей лавки»)

`l2core/scenarios/monitor.py` — ShopMonitor: детекция продажи
ПЕРЕЧИТЫВАНИЕМ окна лавки (OCR показателей количества/прибыли со сравнением
снимков относительно базовой линии, не через окно сообщений игры) —
явное ИСКЛЮЧЕНИЕ из правила «сценарии без внутреннего состояния» (spec:
внутреннее состояние допускается только мониторам/наблюдателям — baseline;
докстринг модуля это фиксирует).

- **Показатели и источник**: OCR-регионы проданного количества/прибыли в
  отгруженной shop_sell не размечены (ocr_regions пуст — KNOWN_GAPS п.10,
  Q27): регионы передаёт вызывающий слой (`regions`, rect'ы в координатах
  окна от якоря; fallback — секция ocr_regions модели). Регион-количество
  (`counter_region`) обязателен, грамматика числовая (digits/price —
  валидация в конструкторе, MonitorError); регион-прибыли
  (`revenue_region`) опционален (payload revenue=None).
- **Семантика baseline/дельт**: `capture_baseline()` — показатели лавки
  после выставления/переразметки (стратегия вызывает после start_sell);
  poll() до линии отклоняется. Увеличение количества = продажа: событие
  sold, baseline := текущие показатели (повторная продажа даёт свежую
  дельту; накопительный счётчик не нужен — показатели окна суммарные).
  Уменьшение = переразметка лавки: события нет (YAGNI: только sold),
  baseline := текущие показатели (иначе продажи после переразметки не
  детектятся), факт — в диагностике. Деградация OCR: счётчик не прочитан —
  события нет, результат degraded с диагностикой, линия не сдвигается;
  прибыль не прочитана при читаемом счётчике — событие с revenue=None
  (честная деградация, флаг degraded).
- **Событие sold**: push-форма — callback-инъекция `on_sold(payload)`;
  `SoldPayload` — sold_count (суммарно), sold_delta, revenue,
  revenue_delta, snapshot_id (sha1 байтов кадра, тот же источник, что у
  RowsResult 6.1). Полная система событий (dispatcher/pull/on-шина) —
  7.3, здесь только генерация.
- **Наблюдатель, не исполнитель**: монитор HID не трогает вообще (ни
  кликов, ни скролла — только grab/find_anchor/OCR-инъекции); guard-тесты
  `tests/test_monitor_shop.py::TestNoChatNoActionsGuard` следят (AST:
  запрет импортов действий и вызовов click/scroll/drag/key_press; детекция
  не через чат). Компонент для observer-потока (7.2): poll() синхронен,
  замков нет — монитором владеет один поток; внешняя синхронизация границы
  — решение 7.2.
- Отмена (kill switch 8.1, инъекция `cancelled`): проверяется до чтения
  кадра — результат cancelled=True, экран не трогается.
- Тесты: tests/test_monitor_shop.py (17 тестов: baseline/poll, дельты при
  серии продаж, переразметка, OCR-деградация, отмена, конфигурация, guard).

## scenario_recover — сценарная точка входа recover (задача 6.4, spec
market-scenarios «Сценарий recover как композиция примитивов без
внутреннего состояния»)

`l2core/scenarios/recover.py` — ТОНКАЯ ОБВЯЗКА поверх цепочки 5.8
(primitives/recover.py, механика там не дублируется):
`scenario_recover(cause, target, *, recipe_runner, провайдеры...)`.

- **Контракт вызова**: `cause` — результат примитива 5.x/сценария 6.1–6.3
  со структурным типом `RecoverCause` (PrimitiveResult + needs_recover);
  БЕЗ needs_recover (успех/отмена) — `ScenarioRecoverError` (fail fast,
  цепочка не запускается; отмену стратегия ветвит до вызова). `target` —
  id восстанавливаемого окна; переоткрытие — рецепт навигации
  open_<target> через recipe_runner (сборка без навигационной модели —
  шаг пропускается с диагностикой). cause протягивается в цепочку
  (needs_recover → release_all первым HID-действием).
- **Провайдеры — готовые, от вызывающего слоя** (паттерн 6.1): screen/hid/
  cursor_pos/activate/window_status/find_anchor/goal_reached/
  current_window/close_point/recipe_runner/... сценарий адаптеры не создаёт,
  wiring на детекторы 3.4/3.5 не собирает (это 7.1). Без внутреннего
  состояния: каждый вызов — независимая композиция.
- **Результат — RecoverResult как есть**: success / escalated (событие
  уже ушло push-callback'ом on_escalation) / cancelled — сценарием не
  перехватывается. Паттерн интеграции: сценарии 6.1–6.3 фиксируют
  needs_recover и обрывают работу, recover вызывает стратегия через
  scenario_recover.
- Тесты: tests/test_scenario_recover.py (9 тестов: target протянут,
  провайдеры/последовательность шагов ScriptedHid, release_all от cause,
  прозрачная эскалация/отмена, fail fast, проброс RecoverError модели).
