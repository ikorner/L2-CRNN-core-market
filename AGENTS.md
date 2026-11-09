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

# Трассировка use-cases по графу переходов (6.5, spec market-scenarios):
# синтезированный узел unknown (достижим отовсюду — аксиома рёбер деградации)
# обязан иметь выход в рабочее состояние через recover_-состояния
# (unknown_no_exit / recover_no_return по рёбрам переходов модели); шаги
# on_fail — только именованные шаги исполнителя (нет координатных
# литералов в переходах). Правила парсера/валидатора не дублирует, а
# ссылается. CLI нет — запуск в тестах CI (tests/test_trace.py).

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

# Запуск бота: публичный API есть (create_api, задача 7.1), готовый
# цикл бота/CLI — нет (kill switch 8.1, события 7.3, потоки 7.2);
# выдумывать команду запуска нельзя.
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
  recover 5.8) классификацией не занимаются. Публичная оценка произвольных
  проверок — `StateDetector.evaluate(checks, frame, anchor)` (та же
  семантика/пороги, что у сигнатур; точка переиспользования run_checks
  сборки API 7.1); шаблоны проверок грузятся лениво с кэшем на экземпляре —
  verify-наборы переходов могут ссылаться на элементы вне сигнатур
  состояний. Порог template-сигнатур
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
timing.click_retry_count) → on_fail[] → сигнал recover (EnterResult;
трёхзначная обёртка OK/FAILED/UNKNOWN — адаптер outcome.py, 5.7).

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
  wheels/cancelled/unconfirmed/on_fail_executed/failures), success = всё n
  проскроллено ИЛИ край достигнут; трёхзначная обёртка — outcome.py (5.7).

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
  → пауза на цели → mouse_up. Механика между press и release — под
  try/finally: кнопка отпускается при любом выходе (отмена, транспортный
  HidError, исключение cancelled()). HidError механики — попытка потрачена
  с причиной в failures (drag обратим, retry безопасен), HidError самого
  release глушится и фиксируется в failures (состояние кнопки неизвестно →
  needs_recover → release_all в recover 5.8); mouse_down намеренно вне try
  (press не подтверждён — release не вызываем, очистка у 5.8).
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
  needs_recover=True. Результат DragResult; трёхзначная обёртка —
  outcome.py (5.7).

### Трёхзначный результат и страж необратимых (задача 5.7, spec
action-primitives, инвариант 4)

`l2core/primitives/outcome.py` — адаптер ПОВЕРХ результатов примитивов
5.1–5.6 (сами примитивы не переписывались): `ActionOutcome {OK, FAILED,
UNKNOWN}` (str-Enum, в лог диагностикой) + `PrimitiveResult` (общий
контракт success/cancelled/unconfirmed) + `to_outcome(result)` + страж
`IrreversibleGuard`. CANCELLED в enum не входит: отмена — домен kill
switch (8.1, README-замечание 1.4), не итог действия; `to_outcome`
отменённый результат не пропускает (ValueError, fail fast) — стратегия
ветвит `cancelled` до адаптации.

- **UNKNOWN vs FAILED** (честное различие): FAILED — verify выполнялся и
  провалился либо HID-эффект не применён вовсе (якорь потерян до клика,
  hover re-verify не сошёлся): эффект отсутствует/опровергнут. UNKNOWN —
  HID-эффект, скорее всего, применён (клик ушёл / текст набран / wheel
  послан / drag отпущен / HID-сбой оставил состояние неизвестным), но
  подтвердить невозможно (якорь потерян на свежем кадре ПОСЛЕ действия,
  OCR-чтение поля невозможно после набора). Различие несёт поле
  `unconfirmed: bool = False`, добавленное в ClickResult/EnterResult/
  ScrollResult/DragResult: True выставляется только в точках «эффект ушёл,
  подтверждение невозможно» (клик→verify без якоря; drag: verify без
  якоря, HID-сбой механики/release; enter: OcrReadError/_TypeFailed после
  набора; scroll: wheel ушёл, якоря не было ни на одном poll'ах verify —
  якорь был, но строка не менялась = опровергнутый эффект, FAILED).
  Маппинг: success→OK; unconfirmed→UNKNOWN; иначе→FAILED; cancelled→
  ValueError. Пины — tests/test_outcome.py (27 тестов, включая реальные
  прогоны примитивов на UNKNOWN/FAILED-ветках).
- **Страж необратимых**: спека запрещает ядру повторять необратимое при
  UNKNOWN, «решение за стратегией» = выбор момента повтора и ответственность
  за него. `IrreversibleGuard` делает запрет исполняемым: стратегия
  помечает необратимое идентификатором и обязует себя before/after вокруг
  исполнения (`wrap()` — обёртка с before/after, защита от пропуска).
  `after(id, UNKNOWN, detail)` блокирует повтор; `before(id)` при блоке —
  `IrreversibleRepeatError` (действие не доходит до исполнителя); снятие —
  только явное `confirm_after_unknown(id, detail)` (записывает
  GuardResolution в журнал resolutions; снятие без блока — ValueError).
  OK/FAILED страж не задействуют; `after` принимает только ActionOutcome
  (результат примитива/отмена — TypeError: домены не смешиваются). Страж
  не исполняет действия и не решает за стратегию — только не даёт молча
  повторить то, что могло уже свершиться.

### Общий recover-цепочка (задача 5.8, spec action-primitives,
инвариант 9)

`l2core/primitives/recover.py` — оркестратор-исполнитель сигнала
needs_recover примитивов 5.1–5.6 поверх существующих механик (пассивные
шаги — общей механикой on_fail click.py; примитивы не переписываются):

    ESC → клик в свободную область → закрыть окно (если открыто не то) →
    переоткрыть через иконку HUD (1 клик + wait verify открытия) →
    перекалибровка якоря → эскалация (событие + пауза)

- **Не ритуал, а путь к цели**: цель — инъекция `goal_reached(frame)`
  (wiring на детекторах 3.4/3.5); проверяется на входе и после каждого
  шага (wait_until по свежим кадрам) — достигнута, цепочка завершается
  досрочно. Шаги, неприменимые к ситуации (окон нет, открыто целевое,
  HUD не локализован), — пропуск с диагностикой; каждый шаг однократен
  (повторы — сама цепочка). Диагностика по шагам — `steps:
  Tuple[RecoverStep]` (имя/итог/detail).
- **Пассивные шаги первыми**; автоперетаскивание окон запрещено (клик по
  заголовку для подъёма — только по решению стратегии, recover не
  исполняет). Пролог — одна активация окна игры на входе и повторная
  перед каждым активным шагом: без окна игры HID-шаги слепы (инвариант
  9) → фиксация + эскалация. Пролог 5.3–5.5 (статус found целевого
  окна) сознательно не применяется: recover вызывается при деградации,
  целевое окно законно отсутствует.
- **Закрыть окно**: открытое окно — инъекция `current_window(frame)`;
  точку кнопки закрытия даёт `close_point(window_id, frame)` (wiring
  считает от якоря той лавки по модели: market — main_market_close_button,
  shop_sell — close_sell_button); точки нет (shop_buy) — документированный
  fallback: ESC как закрытие. Verify: окно закрылось устойчиво (не то
  окно не видно подряд stability_frames свежих кадров).
- **Переоткрыть через HUD**: точку иконки (market_pic / open_buy_trade /
  open_sale_trade из market.yaml) даёт hud_locator-инъекция (конвенция
  5.3); РОВНО ОДИН клик + wait verify открытия. Неоткрытие за таймаут →
  штатный FAILED (фиксация состояния: статус 3.5 + capture_state —
  снимок/модалка) + событие escalation, БЕЗ повторных кликов по HUD
  («Лавку нельзя создать в текущей зоне»: предусловия за таймаут не
  меняются).
- **Перекалибровка якоря**: кэша позиции нет (инвариант 3) — шаг =
  подтверждение стабильности: find_anchor целевого окна подряд
  stability_frames свежих кадров, затем финальная проверка цели.
- **Эскалация**: фиксация состояния → событие escalation
  (callback-инъекция on_escalation, push-форма; диспетчер событий — 7.3,
  здесь точка входа) → пауза wait_until по инъекции `resume()`
  (pause_timeout=None — бесконечно до явного продолжения стратегией).
- **release_all**: вызывающий слой передаёт результат примитива-причины
  (`cause`); при needs_recover recover делает best-effort
  hid.release_all() ПЕРВЫМ HID-действием (до пассивных шагов — залипшая
  кнопка из 5.6 исказила бы любой клик); HidError глушится с фиксацией
  в failures.
- **Отмена**: cancelled (kill switch 8.1) проверяется на входе, перед
  шагами и во всех wait_until: результат cancelled=True, событие
  escalation НЕ отправляется. RecoverResult совместим с
  PrimitiveResult (to_outcome: success→OK, иначе→FAILED; unconfirmed
  всегда False — recover сам необратимого эффекта не производит).
  Исполнение сценариями — 6.4; YAGNI: никакого event dispatcher/API/
  kill switch.

### scan_market и чтение строк таблицы (задача 6.1, spec market-scenarios,
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
(клик по market_pic, verify из перехода модели recover_closed ->
list__tab_home; перехода нет — ScanError, открытие вслепую запрещено) ->
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

### setup_sell и start_sell/cancel_sell (задача 6.2, spec
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
- **setup_sell**: открытие лавки через HUD (клик по open_element —
  open_sale_trade из модели market, verify снаружи: межокнные переходы
  формат не выражает — KNOWN_GAPS п.3; open_element=None — лавка открыта
  вызывающим слоем) → для каждого item: свежий якорь (слепой drag
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

### monitor_shop — мониторинг своей лавки и событие sold (задача 6.3, spec
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

### scenario_recover — сценарная точка входа recover (задача 6.4, spec
market-scenarios «Сценарий recover как композиция примитивов без внутреннего
состояния»)

`l2core/scenarios/recover.py` — ТОНКАЯ ОБВЯЗКА поверх цепочки 5.8
(primitives/recover.py, механика там не дублируется):
`scenario_recover(cause, target, *, hud_model, hud_element, провайдеры...)`.

- **Контракт вызова**: `cause` — результат примитива 5.x/сценария 6.1–6.3
  со структурным типом `RecoverCause` (PrimitiveResult + needs_recover);
  БЕЗ needs_recover (успех/отмена) — `ScenarioRecoverError` (fail fast,
  цепочка не запускается; отмену стратегия ветвит до вызова). `target` —
  id восстанавливаемого окна, `hud_element` — id HUD-иконки его открытия
  (market.yaml): выбор пары — решение вызывающего слоя, маппинг сценарий
  не выдумывает. cause протягивается в цепочку (needs_recover →
  release_all первым HID-действием).
- **Провайдеры — готовые, от вызывающего слоя** (паттерн 6.1): screen/hid/
  cursor_pos/activate/window_status/find_anchor/goal_reached/
  current_window/close_point/hud_locator/... сценарий адаптеры не создаёт,
  wiring на детекторы 3.4/3.5 не собирает (это 7.1). Без внутреннего
  состояния: каждый вызов — независимая композиция.
- **Результат — RecoverResult как есть**: success / escalated (событие
  уже ушло push-callback'ом on_escalation) / cancelled — сценарием не
  перехватывается. Паттерн интеграции: сценарии 6.1–6.3 фиксируют
  needs_recover и обрывают работу, recover вызывает стратегия через
  scenario_recover.
- Тесты: tests/test_scenario_recover.py (8 тестов: target протянут,
  провайдеры/последовательность шагов ScriptedHid, release_all от cause,
  прозрачная эскалация/отмена, fail fast, проброс RecoverError модели).

### Синхронный блокирующий API для стратегий (задача 7.1, spec
core-api-events, инварианты 1/2/4/9)

`l2core/api/core.py` — L2CoreAPI, ТОНКИЙ ФАСАД: методы — композиции
существующих сценариев/примитивов поверх сборки провайдеров ApiWiring
(`l2core/api/wiring.py`), механика не дублируется. Карта: open_market/
close_market/open_shop_buy/open_shop_sell (навигация), set_tab/set_filter/
search/scroll_list (обзор), read_rows (6.1), setup_sell/start_sell/
cancel_sell (6.2 + страж 5.7), read_shop_state (разовое чтение показателей
— паттерн монитора 6.3 без baseline, деградация флагами), get_state
(window/state/params/status 3.4/3.5), recover (6.4), health (агрегат:
окно игры 3.2, окно торговли, OCR, HID — через провайдеров).

- **Thread-owner**: владелец — поток конструктора (или owner_thread); каждый
  публичный метод проверяет threading.current_thread() — чужой поток →
  RuntimeError (spec: «стратегия MUST вызывать блокирующий API только из
  своего потока»; enforcement в точке вызова).
- **Навигация**: open_market — verify/on_fail из перехода модели
  (recover_closed -> list__tab_home, как 6.1); open_shop_buy/open_shop_sell
  требуют verify (межокнные переходы не выражены — KNOWN_GAPS п.3);
  set_tab/set_filter — verify из перехода модели, иначе явный (fail fast).
  Открытие через HUD НЕ click_element 5.3: его пролог требует якорь окна,
  которого при закрытом обзоре нет (та же латентная граница открытия в
  scan 6.1) — механика паттерна 5.8: пролог активации → hud_locator →
  РОВНО ОДИН клик → блокирующее ожидание сигнатуры окна на свежих кадрах
  (wait_until, тайминги конфига): сигнатура сошлась — OK; verify выполнялся,
  но не сошёлся за таймаут — FAILED; якоря не было ни разу — UNKNOWN
  (клик ушёл, не подтверждён). Повторных кликов по таймауту нет.
  Навигация возвращает NavigationResult (success/outcome/cancelled/
  unconfirmed/on_fail_executed/failures, needs_recover).
- **read_shop_state**: регионы counter/revenue — параметр regions →
  wiring.shop_regions → ocr_regions модели (в отгруженной пусто —
  KNOWN_GAPS п.10); грамматики показателей числовые (ApiError иначе);
  якоря нет/регион не сошёлся — ShopStateResult(degraded=True, failures),
  не исключение; HID не трогает (наблюдатель).
- **Отмена**: методы принимают cancelled (аргумент побеждает wiring.
  cancelled); мгновенные вызовы без флага в результате (read_rows/
  get_state/read_shop_state) на отмене — ApiError; остальные — флагом.
- **Страж**: api.guard — IrreversibleGuard сборки (5.7): start_sell при
  UNKNOWN блокирует повтор до guard.confirm_after_unknown стратегией.
- **wiring.py — единая точка сборки провайдеров**: find_anchor(frame,
  window_id) / window_status(frame, window_id) (детекторы 3.3/3.5),
  run_checks — через StateDetector.evaluate (3.4, публичная точка
  переиспользования _evaluate; шаблоны проверок переходов грузятся лениво
  — verify может ссылаться на элементы вне сигнатур состояний),
  current_window (первый найденный якорь), close_point (market →
  main_market_close_button, shop_sell → close_sell_button; shop_buy нет —
  None, fallback 5.8: ESC), hud_locator (template matching в search_area
  HUD-элемента + click_point варианта, порог hud_threshold=0.9),
  goal_reached(target) — якорь цели найден, health-провайдеры
  (game_window — находилка 3.2; ocr_health дефолт True — локальный OCR жив
  по построению; hid_health — is_connected, если есть).
- **create_api(config, ...)** — фабрика живой сборки: load_config по пути,
  модели из ui_model/, MssScreen/HidDevice/TorchOcr (create_ocr_client)/
  PyWin32Gateway (ensure_dpi_aware, get_cursor_pos, activate_game 5.2);
  адаптеры подменяемы (тесты); регионы KNOWN_GAPS — параметрами.
- Win32Gateway расширен get_cursor_pos() (позиция курсора для относительных
  дельт HID; см. activate_game 5.2).
- YAGNI: никакого event dispatcher (7.3), executor/observer (7.2), kill
  switch (8.1) — здесь только синхронный фасад.
- Тесты: tests/test_api.py (контракт всех методов на скриптованном мире:
  делегирование, OK/FAILED/UNKNOWN, thread-owner, отмена, страж),
  tests/test_wiring.py (математика провайдеров, create_api с подменой
  адаптеров на реальных моделях).

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
  факты и диагностика; WaitResult — не результат действия (нет HID-эффекта),
  в обёртку outcome.py (5.7) не адаптируется, структура не мешает.
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
`set_foreground(hwnd)` / `get_cursor_pos()` (реализация —
PyWin32Gateway, в тестах fake).

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
- `l2core/api/` — публичный синхронный API для модулей стратегий (задача
  7.1): L2CoreAPI — тонкий фасад, композирующий сценарии 6.1–6.4 и
  примитивы 5.x поверх провайдеров сборки ApiWiring (l2core/api/wiring.py —
  build_wiring из детекторов 3.3–3.5 и create_api с живыми адаптерами:
  MssScreen/HidDevice/TorchOcr/PyWin32Gateway). События/потоки 7.2–7.3 —
  не здесь (YAGNI).
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
   за стратегией; механизм — outcome.py (5.7): поле `unconfirmed` результатов
   примитивов, адаптер `to_outcome`, страж `IrreversibleGuard`
   (UNKNOWN → повтор запрещён до явного `confirm_after_unknown`).
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
