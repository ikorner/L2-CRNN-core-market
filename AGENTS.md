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
  переходы, OCR-регионы, table_geometry).
- `l2core/detector/` — детекция окна и состояний, статусы окна
  (`not_found / occluded_or_unstable / found`), перехватчик модалок.
  Детектор якоря (3.3, anchor.py): numpy-only `detect_window()` (яркие
  линии рамки — первый дешёвый этап для «Обзора лавок», ~27 мс/кадр) +
  подтверждение/поиск template matching'ом (заголовок лавок, целое окно
  обзора); кэша позиции между кадрами нет — свежий поиск на каждом grab.
- `l2core/primitives/` — примитивы действий (wait_until, activate_game,
  click_element, enter_text/enter_price, scroll_list, drag_item).
- `l2core/scenarios/` — сценарии (scan_market, setup_sell, monitor_shop).
- `l2core/adapters/` — MssScreen/ReplayScreen/RecordingScreen (screen.py,
  формат сессии — session.py), win32-шлюз и находилка окна игры (win32.py,
  DPI-aware объявление процесса), TemplateMatcher (template.py), TorchOcr/RemoteOcr(позже),
  HidArduino.

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
