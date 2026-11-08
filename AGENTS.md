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
