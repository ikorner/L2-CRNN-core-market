## Superpowers ↔ OpenSpec Integration

### Design handoff

Superpowers brainstorming is the primary design exploration process.

After brainstorming is completed and the user approves the design:

1. Locate and read the complete Superpowers design document.
   Default location:
   docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md

2. Treat this document as the authoritative input for OpenSpec proposal generation.

3. Do NOT repeat brainstorming, reopen settled decisions, or independently redesign the feature.

4. Run OpenSpec propose using the existing design document as mandatory context.

5. Translate the approved design into OpenSpec artifacts:
   - proposal.md: intent, motivation, scope, goals and non-goals.
   - specs/: observable, testable functional requirements.
   - design.md: preserve the approved technical design and decisions.
   - tasks.md: implementation tasks derived from the approved design and requirements.

6. Preserve all approved decisions, constraints, trade-offs and non-goals.
   Do not silently omit or reinterpret them.

7. If information is missing or contradictory:
   - First inspect the project and existing OpenSpec specs.
   - Ask the user only if the missing information affects the intended behavior or architecture.
   - Do not invent requirements.

8. Before finishing, verify that every important design decision is represented in the appropriate OpenSpec artifact.

### Conflict resolution

- User-approved decisions take precedence.
- Existing OpenSpec specs describe the current system behavior.
- The approved Superpowers design describes the intended change.
- OpenSpec artifacts formalize that change.
- If the design conflicts with existing specifications, explicitly identify the conflict and resolve it before implementation.

### Implementation boundary

Do not start implementation until OpenSpec artifacts have been reviewed and approved.

---

## Проект: l2core — ядро торгового бота Lineage 2

Драйвер трёх внутриигровых окон торговли: «Обзор лавок» (market),
«Торговая лавка (Купить)» (shop_buy, лавка-выкуп), «Торговая лавка (Продажа)»
(shop_sell). Всё знание об UI — в машиночитаемой UI-модели (`ui_model/`,
YAML); ядро — её интерпретатор плюс примитивы и сценарии. Стратегии торговли
— сменные модули поверх стабильного API, в ядро не входят.

Авторитетные источники решений: `openspec/changes/core-market-driver/design.md`
(решения не переоткрываются) и спецификации в `openspec/changes/core-market-driver/specs/`.

### Команды

Окружение: Python 3.10+, Windows (Win32 API, pywin32). Зависимости и pytest —
в `pyproject.toml`.

```bash
# Установка (репозиторий, редактируемый режим, с dev-зависимостями)
pip install -e ".[dev]"

# Тесты (единственная проверяемая команда на текущем этапе)
python -m pytest

# Конвертер meta.json → UI-модель: импорт кропов в assets/, карточки
# элементов/состояний, diff-отчёт на stdout. Ручные правки в ui_model/*.yaml
# и файлы в assets/ не перезаписываются (ручное побеждает, отчёт фиксирует).
python -m l2core.model.converter

# Валидатор UI-модели: инварианты спеки ui-model (сигнатуры состояний,
# путь в recover, семантика переходов, rect'ы в окне, шаблоны в assets/).
# Отгруженные модели имеют известные пробелы — они явно записаны в
# ui_model/KNOWN_GAPS.yaml (ruling R10), тесты следят за их актуальностью.
python -m l2core.model.validator ui_model/market.yaml ui_model/shop_buy.yaml ui_model/shop_sell.yaml

# Экспорт констант геометрии таблицы (table_geometry) в датасет-пайплайн
# (L2-CRNN-Dataset-Builder): Python-модуль констант с пометкой версии модели
# (--version, git-описание ui_model/ или ручная) и diff-отчётом об отличиях
# от предыдущей выгрузки (stdout + <output>.diff). Детерминирован, без меток
# времени. Значения закреплены golden-тестами tests/test_table_geometry.py.
python -m l2core.model.export_geometry ui_model/market.yaml \
    --version <версия-модели> --output <путь>/table_geometry.py

# Валидация config.yaml против схемы l2core/config.py
python -c "from l2core.config import load_config; load_config('config.yaml')"

# Golden-набор OCR: нарезка/пересоздание эталонных кропов из сырья по индексу
# assets/ocr_golden/index.yaml (rect'ы 1:1, источник истины — индекс)
python tools/gen_ocr_golden.py

# OCR-бенчмарк на golden-наборе (ручной, вне CI): точность и время на CPU и
# GPU; без crnn_best.pt — понятная ошибка, код выхода 2
python tools/benchmark_ocr.py --model <путь>/crnn_best.pt

# Запуск бота: точка входа появится вместе с публичным API (раздел 6 tasks.md);
# документировать/выдумывать команду запуска до этого нельзя.
```

### Структура пакета

Зависимости только вниз: стратегии → сценарии → примитивы → интерпретатор
UI-модели → Screen/Detector/OcrClient/HidPort → адаптеры
(MssScreen/ReplayScreen, TemplateMatcher, TorchOcr, HidArduino).

- `l2core/model/` — интерпретатор UI-модели (YAML: окна, элементы, состояния,
  переходы, OCR-регионы, table_geometry). Грамматики OCR-регионов (4.4,
  grammars.py): чистые функции разбора значений по грамматике (`digits`,
  `price` с разделителями `,`/`.` и суффиксами `к`/`кк`, `text`), валидация
  по диапазону (дефолт price 1..2^63-1, digits 0..2^63-1; опциональные
  `min`/`max`/`units` в грамматике — FORMAT.md) и механизм перечитывания
  при несовпадении: `read_region(ocr, frame, rect, grammar)` через контракт
  OcrClient — одно перечитывание с увеличением региона, при повторе —
  статус `invalid`, `confidence=0.0` (сигнал «не сошлось»). Слой чтения
  (4.5, reading.py): `read_confident` — цикл до `reread_count`
  перечитываний с увеличением региона (внешняя попытка k стартует с
  региона +k·pad от исходного; внутренняя перечитка read_region — +pad
  при несовпадении грамматики, двойного расширения нет) до порога
  `confidence_threshold` из конфига, исчерпание → `OcrReadError` с
  диагностикой (region_id, threshold, attempts: rect/text/confidence/
  status); `read_stable` — стабилизация поля ввода от мигающей каретки:
  чтение кадров от источника (`frame_source()`, совместимо с Screen.grab)
  до двух подряд одинаковых текстов, лимит `max_frames`, исчерпание →
  `OcrReadError`; стабильный текст совпавший с regex-паттерном
  `placeholder` региона (опциональное поле ocr_regions, FORMAT.md) или
  пустой → status `placeholder`, value=None (плейсхолдер ≠ введённый
  текст).
- `l2core/detector/` — детекция окна и состояний, статусы окна
  (`not_found / occluded_or_unstable / found`), перехватчик модалок.
  Детектор якоря (3.3, anchor.py): numpy-only `detect_window()` (яркие
  линии рамки — первый дешёвый этап для «Обзора лавок», ~27 мс/кадр) +
  подтверждение/поиск template matching'ом (заголовок лавок, целое окно
  обзора); кэша позиции между кадрами нет — свежий поиск на каждом grab.
  Детектор состояний (3.4, states.py): классификация (window, params) по
  кадру+якорю; состояние подтверждается строгим совпадением ВСЕХ сигнатур
  (валидатор 2.3 гарантирует ≥2), оценка в иерархии «пиксель-пробы →
  template → OCR» с остановкой состояния на первом провале; OCR-клиент
  (`read(region_img) -> (text, confidence)`) — опциональная инъекция, без
  него состояния с OCR-сигнатурами дают unknown+диагностику (деградация,
  не исключение); конфликт/неподтверждение → unknown с журналированием
  конфликтующих признаков; состояния с префиксом `recover_` (глобальный
  recover 5.8) классификацией не занимаются. Порог template-сигнатур
  0.9 — замер 3.4 (позитивы ≥0.998, макс. ложный дискриминирующий 0.829).
  Детектор статусов (3.5, status.py): статусы `not_found /
  occluded_or_unstable / found` + признак «контент прогрузился». Компонент
  поверх находилки окна игры (3.2) и детекторов якоря/состояний: not_found —
  окно игры недоступно (причина из 3.2) или якорь устойчиво отсутствует;
  occluded_or_unstable — результаты детекции якоря различаются между двумя
  кадрами подряд (anchor_lost / anchor_appeared / anchor_moved сверх
  допуска position_tolerance=8 px, допущение); found — якорь стабилен.
  Перекрытие тултипом/окном из одного када не детектируется (честный минимум
  — нестабильность между кадрами). Контентные сигнатуры — сигнатуры
  не-recover состояний, кроме элемента-опоры якоря (StateDetector.
  check_content, exclude_elements передаёт статус-детектор из своего
  anchor_templates — должен совпадать с опорами AnchorDetector); заголовок
  не считается контентом, флаг в формат модели не добавлен (отчёт 3.5).
  Граница инварианта «кэша позиции нет»: поиск якоря на каждом кадре свежий,
  внутреннее состояние — только факт результата предыдущего кадра для
  сравнения стабильности (монитору история разрешена, design.md решение 8).
  Перехватчик неизвестных модалок (3.6, modal.py): консервативная эвристика
  `modal.unknown` (окно игры есть, ни один якорь не найден, в центральной
  области клиента — крупный яркий прямоугольный регион) → PNG-снимок +
  JSONL-строка в log_dir + событие modal.unknown; recover-исполнителя нет
  (5.8) — точка входа on_unknown. На 39 кадрах сырья не стреляет
  (ни шума, ни позитивов — сырьё модалок не содержит; R4, отчёт 3.6,
  досъёмка реальных модалок — OWNER_QUESTIONS Q20). Транзакция кадра
  (3.6, transaction.py, design.md решение 18): FrameTransaction(Screen) —
  явный контекстный менеджер, один grab анализируется один раз (кэш по
  идентичности кадра, не по времени), refresh()/reverify(predicate) —
  свежий grab + re-verify перед необратимым; вложенный вход — RuntimeError.
  Инфраструктура для примитивов 5.x, исполнителей здесь нет.
- `l2core/primitives/` — примитивы действий (wait_until, activate_game,
  click_element, enter_text/enter_price, scroll_list, drag_item).
- `l2core/scenarios/` — сценарии (scan_market, setup_sell, monitor_shop).
- `l2core/adapters/` — MssScreen/ReplayScreen/RecordingScreen (screen.py,
  формат сессии — session.py), win32-шлюз и находилка окна игры (win32.py,
  DPI-aware объявление процесса), TemplateMatcher (template.py),
  вендоренная копия OCR-кода (ocr_vendor/, задача 4.1), OCR-адаптер
  (ocr.py: OcrClient/TorchOcr/RemoteOcr, задача 4.2), сегментация таблицы
  (table_segmentation/, задача 4.3), HidArduino.

### Вендоринг сегментации таблицы (задача 4.3, design.md решение 12)

Байт-идентичный блок `rows_and_columns` / `extract_cells` / `trim_to_text`
(+ их хелперы и дефолтные константы) из L2-CRNN-Dataset-Builder/
build_dataset.py — `l2core/adapters/table_segmentation/build_dataset.py`.
Источник — коммит 606a65ebb474a1b8ea13ad2483aaa915b89a168d (шапка файла).
Не скопированы: `detect_window` (в ядре — свой детектор якоря), разметка/
сборка датасета, CLI; адаптации — только импортный уровень (убраны импорты
невендоренных частей). Правки вендора запрещены: менять в источнике, затем
перевендоривать (эталоны перегенерирует `tools/gen_table_segmentation_fixtures.py`,
запуск вручную вне CI, проверяет коммит-источник).

Вендор читает геометрию из module-level globals; дефолты — калибровка
датасета 1136x833, устаревшая для рантайма (ruling R5). Рантайм работает
только через wrapper `TableSegmentation` (segmentation.py): при создании
подменяет глобалы константами `table_geometry` модели (через объектную
модель l2core.model), эффект процесс-wide. Маппинг: col_seps[0] → CONTENT_X0,
col_seps[1:] → COL_SEPS, ключ колонки «item» → «name» (обрезка зоны текста
без иконки привязана к литералу «name» в вендоре), полоса заголовка
row_y0−18..row_y0, ROW_H = row_pitch (в новой калибровке нет разделительных
линий между строками), TABLE_BOTTOM = row_y0 + max_rows·pitch (не биндинг
при кропе по якорю), зоны item/price → NAME_TEXT_X/PRICE_TEXT_X. Детали и
обоснования — в отчёте task-4.3-report.md.

Guard-тесты `tests/test_table_segmentation_vendor.py` сравнивают вендор с
эталонами `tests/fixtures/table_segmentation/` (exact: структуры, хэши,
байты кропов на сохранённых входах). Guard запрета прямых импортов
(по аналогии с R13): `tests/test_table_segmentation_contract.py` — импорт
вендоренного build_dataset разрешён только внутри `l2core/adapters/` и
явным allowlist'ом тестов (guard вендора и тест wrapper'а). Интеграционная
нарезка реальных скринов меты константами модели —
`tests/test_table_segmentation.py` (7 строк, тайлинг от row_y0, зоны
колонок из 2.4). Потребитель read_rows — задача 6.1.

### Вендоринг OCR-кода (задача 4.1, инвариант 5)

Байт-идентичная копия обучающего кода CRNN-OCR — `l2core/adapters/ocr_vendor/`
(`crnn_train.py`: константы 32×320/BLANK, `preprocess` с хелперами, `CRNN`,
`ctc_greedy_decode` из L2-CRNN-MODEL-TRAIN/train.py; `model_io.py`:
`load_model` — загрузчик чекпоинта/алфавита из autolabel.py). Источник —
коммит e720a8df8078337740df5c3ad8dc2b1eb4387529, зафиксирован в шапках
файлов. Единственная адаптация тел — квалификация `train.CRNN` → `CRNN`
в `load_model`; всё остальное — импортный уровень (см. шапки). Правки
вендоренного кода запрещены: менять в источнике, затем перевендоривать.

Guard-тесты `tests/test_ocr_vendor.py` сравнивают вендор с эталонными
векторами `tests/fixtures/ocr_vendor/` (exact-сравнение: эталоны
сгенерированы тем же окружением, операции детерминированы). Эталоны
генерирует оригинальным кодом `tools/gen_ocr_vendor_fixtures.py` (вручную,
вне CI; проверяет коммит-источник). Для запуска guard-тестов исходный
репозиторий не нужен. Обёртка инференса (TorchOcr/OcrClient) — задача 4.2,
сюда не входит.

### OCR-адаптер (задача 4.2)

`l2core/adapters/ocr.py`: протокол `OcrClient` (`read(region_img) ->
(text, confidence)`, runtime_checkable — подмена без наследования),
`TorchOcr` (in-process инференс поверх ocr_vendor) и заглушка `RemoteOcr`
(NotImplementedError — удалённый сервис, задача будущего). Точка подмены —
фабрика `create_ocr_client(kind, ...)`: аргумент (далее — конфиг) выбирает
реализацию без правки ядра. Модуль импортируется без torch (ленивый импорт
при создании TorchOcr); модель грузится один раз при создании, устройство —
авто-детект cuda → cpu (инъекция `cuda_available` для тестов обоих путей;
выбор устройства вендоренного load_model mps/cpu для ядра не используется).
Confidence — средняя max-softmax по таймстепам, round 4 (формула
`confidence_from_logits` оригинального autolabel.py — та же метрика, что
в пайплайне автоматической разметки; замер 4.2, отчёт task-4.2-report.md).
Грамматики (4.4), перечитывания (4.5), бенчмарк (4.6) — не здесь (YAGNI).
Тесты: `tests/test_ocr.py` (фейковый чекпоинт fake_crnn.pt, exact-пины).

Guard-тест запрета прямых импортов вендора (4.2b, ruling R13, spec
ocr-reading требование 2): `tests/test_ocr_contract.py` — AST-проход по
l2core/ (кроме самого ocr_vendor/), tests/ и tools/; импорт
`l2core.adapters.ocr_vendor` разрешён только внутри `l2core/adapters/`
(сейчас — только `ocr.py`), иначе тест падает с именем файла и строки
(абсолютные, from-import и относительные формы; строки/комментарии
с текстом «ocr_vendor» ложных срабатываний не дают). Единственное
исключение — `tests/test_ocr_vendor.py` (guard самого вендора, задача 4.1),
зафиксировано явным allowlist'ом.

### Golden-набор OCR и бенчмарк (задача 4.6)

`assets/ocr_golden/` — эталонные кропы с известным текстом: index.yaml
(карточки: id, файл, скриншот-источник, rect 1:1 в пикселях источника,
ожидаемый текст, грамматика digits/price/text, units-суффиксы, критерий
известности и кто/когда подтвердил) + кропы `<id>.png`. Критерий известности
(R4): текст читается человеком со скриншота без сомнений и подтверждён
визуальной проверкой кропа; description меты — аннотации, НЕ ground truth.
Нарезка/пересоздание — `python tools/gen_ocr_golden.py` (из сырья по rect
индекса). Тесты tests/test_ocr_golden.py: индекс валиден, кропы совпадают
с повторной нарезкой из сырья (guard протухания), guard чтения через
OcrClient — только при наличии реальной модели (иначе skip с parked-
диагностикой; карточки с символами вне алфавита чекпоинта пропускаются
с отчётом — алфавит реальной модели до получения файла честно неизвестен),
smoke бенчмарка на fake-чекпоинте (CPU; GPU-ветка честно «skipped» без GPU,
ruling R6).

Бенчмарк (ручной, вне CI): `python tools/benchmark_ocr.py [--model путь]
[--devices cpu cuda] [--repeat N]` — точность (exact match эталонного
текста) и среднее время по кропам отдельно на CPU и GPU; без чекпоинта —
понятная ошибка и код выхода 2 (crnn_best.pt вне git, paths.model).
Реальные числа и досъёмка числовых кропов (цены, «0 из N», адена) —
на машине владельца с crnn_best.pt (docs/OWNER_QUESTIONS.md Q24).

### click_element (задача 5.3, spec action-primitives, инварианты 1/2/3/9)

`l2core/primitives/click.py` — оркестратор клика по элементу модели; все
внешние эффекты — инъекции (HID, экран/FrameTransaction 3.6, activate,
статус, поиск якоря, оценка проверок run_checks, rng, часы):
пролог → цикл попыток (1 + timing.click_retry_count) → on_fail[] → сигнал
recover.

- **Пролог (binding ruling ревью 5.2): клик допустим только при
  activate_game OK (5.2) И статусе окна found (3.5).** Иначе — исключение
  (ActivateGameError проброшена / ClickPreconditionError), HID не трогается
  — слепых кликов нет (инвариант 9).
- Verify (инвариант 2) через обязательную типизированную VerifySpec
  (проверки перехода модели, VerifySpec.from_transition) на свежем кадре;
  клики без verify запрещены — параметр обязателен, guard-тест
  `tests/test_click.py::TestVerifyContract` следит за сигнатурой.
- Hover: наведение курсора на click_point (варианта/центра rect, HUD — только
  через инъекцию hud_locator, без него клик невозможен) → hover-пауза
  (timing.hover_pause) → re-verify присутствия элемента (hover мог изменить
  вид/перекрыть) → только потом click.
- Hover-пауза без sleep (инвариант 1): фиксированная задержка через
  wait_until по инъецированным часам с предикатом «прошло время»
  (интервал — timing.verify_interval); результат игнорируется — задержка
  гарантирована механизмом дедлайна.
- Тултип: детекции из кадра нет (модель элемент-тултип не содержит) —
  честная механика «сдвиг курсора на TOOLTIP_SHIFT_PX (24) + пауза +
  повтор в пределах retry», не детекция.
- on_fail[]: «press_esc», «click_free_area» (точка — free_area_point или
  центр config.resolution, честное допущение); перед on_fail пролог
  повторяется (activate + found): иначе click_free_area был бы слепым
  кликом при деградировавшем окне (инвариант 9) — при провале on_fail
  пропускается, сразу needs_recover=True с причиной в failures;
  неизвестный шаг — ошибка модели. Исчерпание retry + on_fail →
  ClickResult(needs_recover=True) — исполнение recover — задача 5.8.
- Джиттер/hold_ms из behavior-профиля через инъецируемый rng (детерминизм
  с seed, инвариант 7) — вендорный random HidDevice.click не используется.

### enter_text / enter_price (задача 5.4, spec action-primitives)

`l2core/primitives/enter.py` — ввод текста/цены: пролог (как в 5.3:
activate OK + статус found, иначе EnterPreconditionError/ActivateGameError,
HID не трогается) → выбор способа ввода по профилю → фокус → набор →
OCR verify → восстановление раскладки → submit → retry (1 +
timing.click_retry_count) → on_fail[] → сигнал recover (EnterResult,
без трёхзначной обёртки — 5.7, YAGNI).

- Способ ввода (клавиатура/экранный нумпад) — по behavior.numpad_fraction
  через rng (детерминизм с seed). Клавиатура: символ → HID keycode (usage
  table: цифры 0x1E..0x27, a-z → 0x04.., пробел 0x2C; константа LEFT_CTRL
  0xE0 задана локально — в вендорном hid.py её нет), заглавные — с SHIFT
  (key_down/press/up), пауза key_delay_min/max_ms (wait_until по
  инъецированным часам — инвариант 1). Нумпад: клики по элементам модели
  через click_element (маппинг символов — параметр numpad_map; семантика
  экранной клавиатуры сырьём не подтверждена — KNOWN_GAPS, Q25). Не-ASCII
  (кириллица) вне 1-байтного протокола вендора — fail fast (инъекция key_tap
  для расширения).
- Фокус поля — клик по элементу модели (field_element_id) через
  click_element с присутствие-верификацией; без него фокус-клика нет (поле
  сфокусировано вызывающим слоем).
- Verify содержимого: read_stable (4.5) по OCR-региону поля (rect от якоря
  на свежем кадре) до submit; enter_price сравнивает значение, разобранное
  грамматикой price, с ожидаемой ценой (OCR «1,500,000» == введённые
  «1500000»).
- Восстановление раскладки (spec): несовпадение OCR → backspace ×
  len(наблюденного OCR текста) + switch_layout + повтор ввода + повторная
  OCR-проверка (раз на попытку). Та же семантика «стереть-видимое-и-набрать»
  у retry: перед повторным набором попытки стирают содержимое, видимое OCR
  в конце предыдущей (печать поверх нестёртого делала бы verify
  принципиально несходимым). switch_layout — инъекция, дефолт Ctrl+Shift
  (key_down/press/up), сочетание подтверждается у владельца (Q25); чего OCR
  не видит (placeholder/invalid), стирание не достаёт (очистка поля/
  select-on-focus — Q8/Q25, досъёмка модели).
- Submit: Enter (KEY_ENTER) или кнопка «Поиск» (элемент модели, клик через
  click_element) — выбор по behavior.enter_fraction через rng; submit=False
  — ввод без отправки. Verify submit: присутствие кнопки после клика; у
  Enter наблюдаемого признака в модели нет (список обновился — Q25, задача
  модели/стратегии). OCR verify идёт до submit (после Enter поле может
  очиститься).
- Ограничения (R4, честно паркуется — Q25): семантика нумпада, сочетание
  раскладки, признак результата поиска, кириллица.

### scroll_list / scroll_position (задача 5.5, spec market-scenarios)

`l2core/primitives/scroll.py` — скролл списка и чтение позиции скролла.

- **Механика**: wheel есть в протоколе вендора (`HidDevice.scroll`,
  CMD_SCROLL) — fallback на клики по стрелкам не нужен; единица n = один
  щелчок колеса = HID-команда `scroll(1, ±1, delay)` (вниз +1, вверх −1).
- **Пролог** — как в 5.3/5.4: activate OK + статус found, иначе
  ScrollPreconditionError/ActivateGameError, HID не трогается (инвариант 9).
- **Область списка**: элемент-область в модели отсутствует (KNOWN_GAPS) —
  курсор наводится на центр таблицы по table_geometry (середина
  col_seps[0]..[-1] × row_y0..row_y0+max_rows·pitch от якоря).
- **Verify каждой итерации** (инвариант 2): wait_until по свежим кадрам
  (refresh транзакции 3.6) до смены хэша нижней видимой строки ИЛИ признака
  края по направлению; таймаут/интервал — timing.wait_default_timeout /
  verify_interval; sleep нет (инвариант 1). retry = 1 + click_retry_count,
  затем повторный пролог → on_fail[] (те же шаги, что в 5.3) → needs_recover.
- **Признак конца из модели** (предпочтителен по спеке): disabled-стрелки
  `prokrutka_vverh_not_active` / `prokrutka_vniz_not_active` (кропы
  scr_153722_*_not_active, вариант записи «active»), оценка template-проверок
  через инъекцию run_checks (matcher/assets — слой детекции). Элементов нет —
  at_start/at_end = None, честный fallback на сравнение строк (зафиксировано
  в KNOWN_GAPS): замкнутый список всё равно закрыт стрелкой.
- **Позиция**: `scroll_position(frame, ...)` → ScrollPosition{at_start,
  at_end, bottom_row_hash}; хэш — sha1 по обрезанным ячейкам последней
  непустой строки нарезки TableSegmentation (4.3) на константах модели,
  байтовое сравнение между кадрами, не OCR; None — строк нет/окно вне кадра.
- **Результат**: ScrollResult(requested/scrolled/reached_end/position/
  wheels/cancelled/on_fail_executed/failures), success = всё n проскроллено
  ИЛИ край достигнут; трёхзначная обёртка — 5.7 (YAGNI).

### drag_item (задача 5.6, spec action-primitives, инварианты 1/2/9)

`l2core/primitives/drag.py` — drag-перенос предметов (в лавку и обратно);
оркестратор по паттерну 5.3-5.5, общие хелперы — из click.py
(CheckRunner/JitterRng/VerifySpec/_fixed_pause/_move_to/_run_on_fail).

- **Граница ответственности**: drag_item принимает ГОТОВЫЕ экранные точки
  `source_point`/`target_point` — координаты ячеек сеток инвентаря/лавки
  генерируются из геометрии сетки вызывающим слоем (контракт design.md:
  «slot: координаты ячейки генерируются из геометрии сетки лавки, не
  размечаются вручную» — 6.2), якорь вызывающий берёт свежий непосредственно
  перед вызовом; drag точки внутри попытки пересчитать не может (курсор уже
  тащит предмет) — сдвиг окна между расчётом точек и drop ловит verify,
  retry со свежими точками — решение вызывающего слоя. Verify — обязательная
  типизированная VerifySpec (guard-тест TestVerifyContract следит за
  сигнатурой), оценка через инъекцию run_checks.
- **HID-drag собран из низкоуровневых примитивов вендора** (готовой команды
  drag в протоколе нет, вендорить нельзя): move_rel к source →
  mouse_down (BTN_LEFT) → удержание → пакетная траектория traj
  ((dx, dy, delay_ms) за точку, исполнение на плате, один пакет CMD_TRAJ)
  → пауза на цели → mouse_up. Отмена между press и release отпускает
  кнопку (mouse_up) до возврата — мышь зажатой не остаётся.
- **Траектория (behavior-профиль)**: source/target + джиттер
  click_jitter_px; 1..3 промежуточные точки (drag_midpoints_min/max) с
  джиттером move_jitter_px; последний шаг — ровно в target; пауза шага —
  move_step_delay_min/max_ms через rng (детерминизм с seed; порядок
  rng-розыгрышей зафиксирован пинами tests/test_drag.py::planned).
  Число точек сверх ёмкости пакета HID-протокола (TRAJ_MAX_POINTS) —
  DragElementError до HID (fail fast).
- **Пролог** — как 5.3-5.5: activate OK + статус found, иначе
  DragPreconditionError/ActivateGameError, HID не трогается (инвариант 9);
  перед press каждой попытки — свежий якорь, якоря нет — попытка потрачена
  без press.
- **Паузы без sleep (инвариант 1)**: удержание после press (press_delay
  профиля, мс→сек) и пауза на цели перед release (timing.hover_pause) —
  _fixed_pause по инъецированным часам.
- **Обратимость/re-verify (3.6)**: перенос обратим (предмет можно
  перетащить обратно) — re-verify «перед необратимым» применяется к
  start_sell (6.2), НЕ к drag; drag обязан verify ПОСЛЕ release и не мешает
  reverify сценария (транзакция кадра не вложенная: drag открывает свою
  FrameTransaction, reverify start_sell — вне примитива).
- on_fail[] — общая механика 5.3 (press_esc / click_free_area) с повторным
  прологом перед шагами; при деградации окна on_fail пропускается →
  needs_recover=True. Результат DragResult (без трёхзначной обёртки — 5.7).

### wait_until и запрет sleep (задача 5.1, инвариант 1)

`l2core/primitives/wait.py` — единственная sanctioned-точка блокирующего
ожидания в ядре (инвариант 1: «Никаких sleep — только
wait_until(predicate, timeout)»). Сигнатура:
`wait_until(predicate, *, timeout=None, interval=None, config=None,
cancelled=None, clock=time.monotonic, sleep=time.sleep) -> WaitResult`.

- Подстановка таймингов (договорённость 4.5, защита от рассинхрона
  дефолтов): явные `timeout`/`interval` побеждают; если None — значения
  из переданного `config` (timing.wait_default_timeout /
  timing.verify_interval); литеральных дефолтов в модуле нет, оба отсутствуют
  → ValueError. Подставляет вызывающий слой.
- Семантика: первый poll немедленно, дальше poll каждые `interval`;
  сон перед poll урезан остатком до дедлайна — последний poll не позже
  timeout, дольше не ждёт. Elapsed на таймауте — ровно timeout. `cancelled`
  (точка инъекции kill switch 8.1, сам kill switch не здесь) проверяется
  до первого poll'а и после каждого неуспешного; отмена — результат с
  флагом `cancelled=True`, не исключение.
- Возврат `WaitResult(success, cancelled, polls, elapsed, last_value)` —
  факты и диагностика; обёртка в трёхзначный OK/FAILED/UNKNOWN — задача 5.7
  (здесь не строится, структура не мешает).
- Часы и сон — инъекции (`clock`, `sleep`), в проде time.monotonic /
  time.sleep; тесты на фейковом времени без реального ожидания
  (инвариант 7).

Guard-тест запрета sleep — `tests/test_sleep_contract.py` (по аналогии с
R13/4.2b): AST-проход по l2core/ ищет вызовы `.sleep(...)` (time/asyncio),
`sleep(...)` (from-import) и `.wait(...)` на threading.Event/Condition;
сообщение — файл:строка. Ссылка `time.sleep` без вызова не нарушение
(инъекция sleeper'а). Allowlist явный: `adapters/hid.py` (сон у железа —
перезагрузка Arduino), `adapters/ocr_vendor/`, `adapters/table_segmentation/`
(байт-идентичный вендор), `primitives/wait.py` (sanctioned-точка ожидания).

### activate_game (задача 5.2, spec core-api-events, safety)

`l2core/primitives/activate.py`: активация окна игры перед действием —
поиск окна (GameWindowFinder, критерии из config.window), проверка фокуса
(GetForegroundWindow == hwnd через шлюз), при необходимости SetForeground,
verify через wait_until («фокус получен», тайминги timing из конфига),
retry: попыток = 1 + timing.click_retry_count; исчерпание → ActivateGameError
(reason=focus_not_acquired, диагностика) — сигнал эскалации, recover-цепочка
— 5.8. Окно не найдено/свёрнуто/без клиентской области → ActivateGameError
с reason находилки (3.2), SetForeground не вызывается (инвариант 9). Отмена
через cancelled (kill switch 8.1) → результат с cancelled=True, не исключение.
Протокол Win32Gateway расширен методами `get_foreground_window()` /
`set_foreground(hwnd)` (реализация — PyWin32Gateway, в тестах fake).

Проверка перекрытия — честный минимум: фокус. Надёжной дешёвой проверки
перекрытия клиентской области другим окном Win32 не даёт; перекрытие
покрывается статус-детектором 3.5 (occluded_or_unstable) и перехватчиком
модалок 3.6 — вне примитива. Контракт «activate перед каждым действием»
оформляется в 5.3+ (click_element), здесь — готовый примитив + сигнал.

### Template matching (задача 3.3, решение 13)

Вид поиска — NCC, эквивалент OpenCV `TM_CCOEFF_NORMED`. Реализация по
умолчанию — `OpenCvTemplateMatcher` (`default_matcher()`): замер на всех 39
скриншотах `base_screenshot/scr/` показал полную эквивалентность решений с
numpy-only альтернативой (0 расхождений find_anchor по 3 окнам × 39 кадров),
но numpy-only в 2.3–4 раза медленнее на кадре 1920x1080 (jak 198x29:
~237 мс против ~58 мс; целое окно 1008x674: ~183 мс против ~80 мс;
полный проход детекции: 26.7 с против 9.8 с; CPU, один поток) — ощутимо
хуже на горячем пути. `NumpyTemplateMatcher` остаётся встроенной
альтернативой (FFT + интегральные изображения), guard-тесты следят за
паритетом решений и score. Числа и протокол — в отчёте 3.3.

### Формат файла сессии (контракт replay, задача 9.2)

Сессия — директория: `session.json` (манифест: `format: l2core-session`,
`version: 1`, meta без меток времени, события) + `frames/*.png` (кадры BGR
uint8, без потерь; одинаковые кадры дедуплицируются по содержимому).
События: `frame` (кадр изменился без действия; первое событие обязано быть
frame — начальное состояние) и `action` (`action` — нормализованный словарь
действия, `image` — кадр-после или null, `result` — verify-результат или
null). Replay-модель эффектов: действие → записанный после него кадр
(ReplayScreen не кликает, а продвигает курсор; несовпадение действия —
`ActionMismatchError`, конец записи — `SessionExhaustedError`).

Программная запись сессии без живого экрана (R7, путь 9.2):
`SessionWriter(path, meta=...)` → `record_frame(arr)` /
`record_action(action, frame_after=arr|None)` / `set_result(status, detail)`
→ `save()` (или контекстный менеджер). Чтение: `load_session(path)` →
`.events`, `.read_image(ref)`. Живой прогон пишет сессию через
`RecordingScreen(inner, path)` (декоратор поверх любого Screen).
- `l2core/api/` — публичный синхронный API + события для модулей стратегий.
- `l2core/config.py` — схема и типизированный загрузчик `config.yaml`.
- `tests/` — pytest; `ui_model/` — YAML UI-модель; `assets/` — шаблоны и
  golden-кропы (версионируются вместе с моделью); `base_screenshot/scr/` —
  сырьё (39 скриншотов + meta.json), не зависимость ядра.

### Инварианты (нарушать нельзя)

1. **Никаких `sleep` — только `wait_until(predicate, timeout)`**; таймауты из
   конфига (`timing.wait_default_timeout`, `timing.verify_interval`).
2. **Verify на каждое действие**: retry (n из конфига) → `on_fail[]` из
   перехода модели → общий recover.
3. **Никаких абсолютных координат вне слоя якоря**: все координаты элементов
   1:1 от якоря своего окна; якорь ищется на каждом кадре, кэша позиции нет.
4. **Трёхзначный результат критичных действий: OK / FAILED / UNKNOWN.**
   UNKNOWN = эффект не подтверждён (лаг, перекрытие). Потенциально необратимые
   действия (начать продажу, подтвердить) при UNKNOWN не повторяются — решение
   за стратегией.
5. **Байт-идентичность вендоренного OCR-препроцесса**: вендорится
   замороженная копия preprocess + CTC greedy decode + загрузчик чекпоинта из
   `L2-CRNN-MODEL-TRAIN` (guard-тесты на эталонных векторах и кропах
   `assets/ocr_golden/`). Правки вендоренного кода запрещены.
6. **TDD**: сначала тест (красный), потом код (зелёный).
7. **Детерминизм тестов**: ни тест, ни код под тестом не зависят от времени
   суток, случайных чисел без фиксированного seed и внешних сервисов;
   replay-входы — записанные кадры, не живой экран.
8. **Код и комментарии — на русском.**
9. **Слепых кликов нет**: заблокированный экран, оверлеи, невозможность
   активации окна → осмысленная ошибка / событие эскалации.

### Процесс обновления модели после патчей игры

Патч игры ломает UI → чинится модель, не код (в общем случае):

1. Доснять скрины нового состояния, обновить meta.json (сырьё в
   `base_screenshot/scr/`).
2. Через конвертер (задача 2.2) импортировать кропы в `assets/` и собрать
   карточки элементов; конвертер пишет diff-отчёт и НЕ перезаписывает ручные
   правки модели.
3. Разметить/исправить rect'ы, сигнатуры, переходы (verify, on_fail) в
   `ui_model/`; переход без verify/on_fail не считается завершённым.
4. Прогнать валидатор модели (задача 2.3) и `python -m pytest` — golden-тесты
   геометрии и эталонные кропы `assets/ocr_golden/` обязаны проходить.
5. Если патч меняет геометрию таблицы — перекалибровать `table_geometry` и
   экспортировать константы в датасет-пайплайн (общий источник с обучением,
   версия + diff-отчёт).
6. Если модели недостаточно (новая динамика, новый примитив) — менять код
   по слоям выше, инварианты сохранять.

### Правила изменений

- Обновление AGENTS.md — часть DoD ломающих задач: изменение команд,
  структуры, инвариантов или процесса без правки этого файла = задача не
  закрыта.
- Процессный раздел «Superpowers ↔ OpenSpec Integration» выше — отдельный
  контракт проекта; проектные изменения его не трогают.
- Документация не описывает несуществующий код: команда запуска, API и
  сценарии появляются в README/AGENTS.md только вместе с реализацией.
- `crnn_best.pt` — вне git (путь в конфиге, `.gitignore`); шаблоны и
  golden-кропы в `assets/` — версионируются.
