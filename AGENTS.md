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

# Живой dry-run прогон (ручной, вне CI; приёмка 9.4): требует запущенную и
# видимую игру; проба по фасаду (health/get_state/open_market/read_rows/
# scroll_list/close_market), действия — намерения без HID, kill switch
# (хоткей + таймаут неактивности) активен на прогоне; --model побеждает
# paths.model; без чекпоинта — понятная ошибка, код выхода 2. Это зонд
# приёмки, не цикл бота.
python tools/live_dry_run.py --model <путь>/crnn_best.pt

# Инструментальная досъёмка кадров в base_screenshot/scr/ (ручная, вне CI;
# игра запущена и видима): захват одного кадра + заглушка в meta.json
# (разметка rect'ов кропов — отдельно). Имя — без расширения.
python tools/capture_frame.py <имя-кадра> "описание"

# Запуск бота: публичный API есть (create_api, задача 7.1), kill switch есть
# (KillSwitch 8.1, компонент для сборки вызывающим слоем), готовый
# цикл бота/CLI — нет (события 7.3, потоки 7.2 есть);
# выдумывать команду запуска нельзя.
```

### Структура пакета

Зависимости только вниз: стратегии → сценарии → примитивы → интерпретатор
UI-модели → Screen/Detector/OcrClient/HidPort → адаптеры
(MssScreen/ReplayScreen, TemplateMatcher, TorchOcr, HidArduino).

- `l2core/model/` — интерпретатор UI-модели (YAML: окна, элементы, состояния,
  переходы, OCR-регионы, table_geometry) + чтение строк таблицы (rows.py) +
  навигационная модель (navigation.py: элементы/сигнатуры/рецепты
  открытия окон, формат — FORMAT.md, «Навигационная модель»).
- `l2core/detector/` — детекция окна и состояний (anchor/states/status,
  статусы `not_found / occluded_or_unstable / found`; якорь многоопорный —
  кортеж опор на окно), перехватчик модалок, транзакция кадра
  (FrameTransaction), оценщик проверок навигации (navchecks.py — меню и
  quick-slot без якоря окна торговли).
- `l2core/primitives/` — примитивы действий (wait_until, activate_game,
  click_element, enter_text/enter_price, scroll_list, drag_item, outcome,
  recover, press_key 5.9) + исполнитель рецептов навигации (navigation.py:
  шаги key/click с when/skip_if и alternate, открытие окон при закрытом
  окне).
- `l2core/scenarios/` — сценарии (scan_market, setup_sell, monitor_shop,
  scenario_recover).
- `l2core/api/` — публичный синхронный API для модулей стратегий (L2CoreAPI,
  wiring, engine/L2CoreBot, events, killswitch, dryrun, actionlog).
- `l2core/adapters/` — MssScreen/ReplayScreen/RecordingScreen (screen.py,
  формат сессии — session.py), win32-шлюз и находилка окна игры (win32.py,
  DPI-aware объявление процесса), TemplateMatcher (template.py),
  вендоренная копия OCR-кода (ocr_vendor/), OCR-адаптер (ocr.py),
  сегментация таблицы (table_segmentation/), HidArduino, кейкоды клавиш
  (keycodes.py — вне вендоренного hid.py: KEY_X и словарь KEYCODES).
- `l2core/config.py` — схема и типизированный загрузчик `config.yaml`.
- `tests/` — pytest; `ui_model/` — YAML UI-модель (окна + common +
  navigation.yaml); `assets/` — шаблоны и golden-кропы (версионируются
  вместе с моделью); `base_screenshot/scr/` — сырьё (47 скриншотов +
  meta.json), не зависимость ядра.

Детальные заметки по слоям (механики, контракты, guard-тесты, отчёты) —
`docs/agents/`, читаются по необходимости, содержимое прежнее (вынесено из
этого файла без потерь; нумерация задач сохранена для сквозных ссылок):

- `model-reading.md` — грамматики OCR-регионов (4.4), слой чтения
  read_confident/read_stable (4.5).
- `detection.md` — детектор якоря (3.3, +template matching), состояний
  (3.4), статусов (3.5), перехватчик модалок и транзакция кадра (3.6).
- `adapters.md` — вендоринг OCR-кода (4.1), OCR-адаптер (4.2), вендоринг
  сегментации таблицы (4.3), golden-набор OCR и бенчмарк (4.6), формат
  файла сессии replay (9.2).
- `primitives.md` — примитивы 5.1–5.9: wait_until, activate_game,
  click_element, enter_text/enter_price, scroll_list, drag_item, outcome
  (OK/FAILED/UNKNOWN + страж необратимых), recover-цепочка, press_key
  (нажатие клавиши с verify) и исполнитель рецептов навигации.
- `scenarios.md` — scan_market и read_rows/дедуп (6.1), setup_sell/
  start_sell/cancel_sell (6.2), monitor_shop (6.3), scenario_recover (6.4).
- `api-runtime.md` — L2CoreAPI и wiring (7.1), потоки L2CoreBot (7.2),
  события on/wait_event/журнал (7.3), kill switch (8.1), dry-run (8.2),
  JSONL-лог действий (8.3), эскалации при блокировках экрана (8.4).

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

- Обновление AGENTS.md и затронутых заметок `docs/agents/` — часть DoD
  ломающих задач: изменение команд, структуры, инвариантов или процесса без
  правки соответствующих файлов = задача не закрыта.
- Процессный раздел «Superpowers ↔ OpenSpec Integration» выше — отдельный
  контракт проекта; проектные изменения его не трогают.
- Документация не описывает несуществующий код: команда запуска, API и
  сценарии появляются в README/AGENTS.md только вместе с реализацией.
- `crnn_best.pt` — вне git (путь в конфиге, `.gitignore`); шаблоны и
  golden-кропы в `assets/` — версионируются.
