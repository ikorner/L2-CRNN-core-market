# Живой dry-run прогон (9.4) — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ручной раннер-зонд `tools/live_dry_run.py`, собирающий живую dry-run
сборку (create_api + KillSwitch) и прогоняющий фиксированную пробу по фасаду
L2CoreAPI — приёмка задачи 9.4.

**Architecture:** Тонкий раннер поверх существующего фасада (механика не
дублируется): `run_probe(api)` — чистая композиция вызовов фасада, тестируема
на stub-API; `main()` — argparse + живая сборка (KillSwitch, TorchOcr по
флагу --model, create_api(dry_run=True)) и печать сводки.

**Tech Stack:** Python 3.10+, pytest; только stdlib + существующий l2core.

**Spec:** `docs/superpowers/specs/2026-11-10-live-dry-run-design.md`
(утверждён). Открытый контракт задачи: `openspec/changes/core-market-driver/tasks.md` п. 9.4.

## Global Constraints

- Код и комментарии — на русском (инвариант 8 AGENTS.md).
- TDD: тест (красный) раньше кода (зелёный) (инвариант 6).
- Детерминизм тестов: stub-API в тестах, без экрана/HID/Win32 (инвариант 7).
- Никаких sleep вне wait_until ядра (инвариант 1) — раннер только вызывает
  фасад, ожидания внутри ядра.
- Shipped `config.yaml` == дефолты схемы — НЕ редактируем; чекпоинт OCR
  передаётся флагом `--model`.
- Раннер — ручной инструмент вне CI (по аналогии с tools/benchmark_ocr.py);
  загрузка модуля из тестов — через `importlib.util.spec_from_file_location`
  (прецедент: tests/test_ocr_golden.py, BENCHMARK_SCRIPT).
- Отсутствие чекпоинта → понятное сообщение в stderr, код выхода 2
  (конвенция benchmark_ocr.py).

## Review Focus

- **Игра не запущена / окно не найдено** — health.game_window=False, шаги
  честно падают без исключений раннера; сводка печатается. (Live-only,
  семантика зафиксирована тестом на ошибку шага + сводку.)
- **Kill во время прогона** — cancelled между шагами прерывает пробу,
  сводка выполненных шагов печатается, код выхода 0.
  Тест: `test_probe_cancelled_between_steps`.
- **Чекпоинт отсутствует** — exit 2, понятное сообщение с путём.
  Тест: `test_resolve_model_missing_returns_none`.
- **Шаг бросает исключение** (например ActivateGameError при свёрнутой
  игре) — ошибка фиксируется в шаге, проба продолжается.
  Тест: `test_probe_continues_after_step_error`.
- **Обзор не открылся** (типичный исход dry-run) — табличные шаги
  пропускаются, close_market всё равно вызывается.
  Тест: `test_probe_skips_table_when_not_found`.

---

### Task 1: Ядро пробы — `run_probe` + сводка (TDD)

**Files:**
- Create: `tests/test_live_dry_run.py`
- Create: `tools/live_dry_run.py` (в этом шаге — только проба и сводка, без main)

**Interfaces:**
- Consumes: `L2CoreAPI` методы: `health() -> HealthResult`,
  `get_state() -> CoreState`, `open_market(**kw) -> NavigationResult`,
  `read_rows(**kw) -> RowsResult`, `scroll_list(direction, n, **kw) ->
  ScrollResult`, `close_market(**kw) -> NavigationResult`. Поля, которые
  читает проба: `HealthResult(ok, game_window, game_detail, trading_window,
  ocr, hid)`, `CoreState(window, status, state, detail)`,
  `NavigationResult(success, outcome, cancelled, unconfirmed, needs_recover)`,
  `RowsResult(rows, snapshot_id)` (читается len(rows), snapshot_id),
  `ScrollResult(scrolled, reached_end, wheels)` — все из
  `l2core/api/core.py`, `l2core/model/rows.py`, `l2core/primitives/scroll.py`.
- Produces (Task 2 опирается):
  ```python
  @dataclass(frozen=True)
  class StepReport:
      name: str
      detail: str
      error: Optional[str] = None
      @property
      def ok(self) -> bool: ...

  @dataclass(frozen=True)
  class ProbeReport:
      steps: Tuple[StepReport, ...]
      killed: bool

  def run_probe(api, *, out: Callable[[str], None] = print,
                cancelled: Optional[Callable[[], bool]] = None,
                scroll_n: int = 2) -> ProbeReport
  ```
  Модуль грузится тестом через importlib.util (прецедент BENCHMARK_SCRIPT).

- [ ] **Step 1: Написать failing тесты `tests/test_live_dry_run.py`**

Stub-API (в файле теста): класс `StubApi` с методами фасада, пишущими
имена вызовов в `self.calls` и возвращающими реальные frozen-даатклассы
(`HealthResult(ok=True, game_window=True, game_detail="ok",
trading_window=None, ocr=True, hid=True)` и т.п.; `NavigationResult(
success=..., outcome=ActionOutcome.OK/FAILED, ...)`; `RowsResult(rows=(),
snapshot_id="s1")`; `ScrollResult(...)` по фактическим полям модуля
scroll.py). Поведение настраивается: статус состояния (`found`/`not_found`),
исключение на заданном методе (`fail_on`).

Тесты:
```python
def test_probe_sequence_and_summary():
    # get_state после open вернул window="market", status="found"
    # -> calls == ["health", "get_state", "open_market", "get_state",
    #              "read_rows", "scroll_list", "read_rows", "close_market"]
    # report.killed is False; каждый шаг ok, детали непустые

def test_probe_skips_table_when_not_found():
    # get_state всегда status="not_found"
    # -> "read_rows"/"scroll_list" нет в calls; "close_market" есть

def test_probe_continues_after_step_error():
    # open_market бросает RuntimeError("нет окна")
    # -> шаг open_market в отчёте с error, шаги после него выполнены,
    #    close_market вызван

def test_probe_cancelled_between_steps():
    # cancelled возвращает True после первого шага
    # -> report.killed is True; calls == ["health"]; сводка напечатана
```

- [ ] **Step 2: Запустить тесты, убедиться в красном**

Run: `python -m pytest tests/test_live_dry_run.py -v`
Expected: FAIL (модуль tools/live_dry_run.py отсутствует / ImportError).

- [ ] **Step 3: Реализовать пробу в `tools/live_dry_run.py`**

Модульный docstring: назначение (живой dry-run прогон 9.4), пример команды,
честная семантика исходов. Датклассы `StepReport`/`ProbeReport` и
`run_probe` — сигнатуры из Interfaces.

Логика `run_probe`: список шагов `(name, callable)`; шаги таблицы
добавляются только если второй `get_state()` дал `window == "market"` и
`status == "found"`. Перед каждым шагом: `if cancelled is not None and
cancelled(): killed=True, break`. Исключение шага → `StepReport(name,
detail="", error=f"{type(e).__name__}: {e}")`, проба продолжается.
Каждый завершённый шаг печатается через `out(f"  [шаг] деталь")` или
`out(f"  [шаг] ОШИБКА: ...")`; `run_probe` в конце печатает сводку
`out(...)` со списком шагов. Детали форматируются из полей результатов
(health: `ok/game_window/trading_window/ocr/hid`; state: `window/status/
state`; open/close: `success/outcome/cancelled/needs_recover`; read_rows:
`len(rows)/snapshot_id`; scroll: `scrolled/reached_end/wheels`).

- [ ] **Step 4: Запустить тесты, убедиться в зелёном**

Run: `python -m pytest tests/test_live_dry_run.py -v`
Expected: PASS (4 теста).

- [ ] **Step 5: Прогнать весь набор на регрессию**

Run: `python -m pytest`
Expected: PASS (без новых падений).

- [ ] **Step 6: Commit**

```bash
git add tests/test_live_dry_run.py tools/live_dry_run.py
git commit -m "feat(tools): проба живого dry-run прогона (run_probe, 9.4)"
```

### Task 2: CLI-сборка — `main()` с KillSwitch и create_api

**Files:**
- Modify: `tools/live_dry_run.py` (добавить `main`, `_resolve_model`)
- Modify: `tests/test_live_dry_run.py` (тесты `_resolve_model` и main)

**Interfaces:**
- Consumes: Task 1 (`run_probe`, `ProbeReport`); `l2core.config.load_config`,
  `l2core.api.wiring.create_api`; `l2core.api.killswitch.KillSwitch`;
  `l2core.adapters.win32.PyWin32Gateway, Win32HotkeySource`;
  `l2core.adapters.ocr.create_ocr_client`; `l2core.api.dryrun.DryRunHid`.
- Produces:
  ```python
  def _resolve_model(config_model: str, override: Optional[str]) -> Optional[Path]
      # override побеждает config_model; Path не существует -> None

  def main(argv: Optional[Sequence[str]] = None) -> int
      # 0 — прогон завершён (FAILED-шаги — честные исходы dry-run);
      # 2 — нет чекпоинта / конфиг не читается (понятное сообщение в stderr)
  ```

- [ ] **Step 1: Написать failing тесты**

```python
def test_resolve_model_override_wins(tmp_path):
    # существующий файл a.pt, override указывает на него, config_model —
    # несуществующий -> результат == a.pt

def test_resolve_model_missing_returns_none(tmp_path):
    # ни override, ни config_model не существуют -> None
```

Тесты main НЕ пишем: живую сборку (Win32, mss, torch) main поднимает
целиком — это live-путь, покрываемый самим прогоном 9.4 (вне CI).

- [ ] **Step 2: Запустить, убедиться в красном**

Run: `python -m pytest tests/test_live_dry_run.py -k resolve -v`
Expected: FAIL (функции нет).

- [ ] **Step 3: Реализовать `main()` в `tools/live_dry_run.py`**

argparse: `--config` (default `"config.yaml"`), `--model` (default None).
Порядок:
1. `cfg = load_config(args.config)`; `model = _resolve_model(
   cfg.paths.model, args.model)`; `model is None` → stderr-сообщение с
   двумя путями, return 2.
2. `gateway = PyWin32Gateway()`; `ks = KillSwitch(cfg.kill_switch,
   hotkey_source=Win32HotkeySource(cfg.kill_switch.hotkey),
   set_cursor_pos=gateway.set_cursor_pos, on_kill=<печать reason>)`.
3. `ocr = create_ocr_client("torch", model_path=str(model))`;
   `api = create_api(cfg, gateway=gateway, ocr=ocr,
   cancelled=ks.cancelled, note_activity=ks.note_activity, dry_run=True)`.
4. `ks.attach(hid=api._w.hid)` (DryRunHid: release_all при kill — no-op
   с логом; доступ к wiring — прецедент README `api._w.hid.intentions`).
5. Печать вводной строки: dry-run, хоткей, путь чекпоинта, log.dir.
6. `ks.start()`; `try: report = run_probe(api, cancelled=ks.is_killed)`
   `except KeyboardInterrupt: печать «прервано пользователем»`;
   `finally: ks.stop()`.
7. Сводка main: число намерений `len(api._w.hid.intentions)` (если
   `isinstance(api._w.hid, DryRunHid)`), `report.killed` / причина
   `ks.reason`, путь `cfg.log.dir`. return 0.
   Внизу модуля: `if __name__ == "__main__": raise SystemExit(main())`.

- [ ] **Step 4: Запустить тесты, убедиться в зелёном**

Run: `python -m pytest tests/test_live_dry_run.py -v`
Expected: PASS (6 тестов).

- [ ] **Step 5: Дымовая проверка CLI без чекпоинта**

Run: `python tools/live_dry_run.py --model nonexistent.pt`
Expected: понятное сообщение в stderr, код выхода 2
(`echo $?` в bash → 2).

- [ ] **Step 6: Commit**

```bash
git add tests/test_live_dry_run.py tools/live_dry_run.py
git commit -m "feat(tools): CLI сборка живого dry-run прогона (KillSwitch + create_api)"
```

### Task 3: Документация — AGENTS.md и README

**Files:**
- Modify: `README.md` (раздел «Dry-run», путь уточнить по факту заголовка)
- Modify: `AGENTS.md` (список команд, рядом с benchmark_ocr.py)

**Interfaces:**
- Consumes: команда из Task 2: `python tools/live_dry_run.py --config
  config.yaml --model E:\L2-CRNN-MODEL-TRAIN\runs\20261105-213625\crnn_best.pt`.
- Produces: задокументированная ручная команда прогона (вне CI).

- [ ] **Step 1: Дописать README (раздел «Dry-run»)**

Три-четыре строки: команда прогона; что происходит (экран читается
по-настоящему, действия — намерения в логе `l2core.dry_run` и JSONL
ActionLogger); ожидаемые исходы (FAILED/UNKNOWN на навигации — честно,
OK если окно уже открыто); требование запущенной игры и хоткей остановки.

- [ ] **Step 2: Дописать AGENTS.md (команды)**

В блок команд, рядом с benchmark: команда прогона с пометкой «ручной
прогон на живой игре (приёмка 9.4), вне CI; требует запущенной игры;
без чекпоинта — ошибка и код выхода 2». Одной записью в стиле соседних.

- [ ] **Step 3: Проверка согласованности**

Run: `python -m pytest`
Expected: PASS (guard-тестов на документацию нет, но регрессия не нужна —
комментарий-проверка, что ничего не сломано).

- [ ] **Step 4: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: команда живого dry-run прогона (9.4) в README и AGENTS.md"
```

### Task 4: Живой прогон и закрытие 9.4

**Files:**
- Modify: `openspec/changes/core-market-driver/tasks.md` (одна строка 9.4)

- [ ] **Step 1: Запустить живой прогон**

Run: `python tools/live_dry_run.py --model "E:\L2-CRNN-MODEL-TRAIN\runs\20261105-213625\crnn_best.pt"`
(Игра должна быть запущена и видима; остановка — хоткей ctrl+shift+k.)
Expected: сводка с определёнными состояниями на живых кадрах, число
намерений > 0, ключевые кадры в `logs/actions/frames/`.

- [ ] **Step 2: Зафиксировать результат**

Заметка в строке 9.4: галочка `[x]` + дата и итог (какие состояния
определились, сколько намерений, куда смотреть). Если прогон выявил
проблему (состояния не определяются) — НЕ закрывать галочку, описать
находку в ответе пользователю.

- [ ] **Step 3: Commit**

```bash
git add openspec/changes/core-market-driver/tasks.md
git commit -m "chore(openspec): 9.4 закрыт — живой dry-run прогон выполнен"
```

## Self-Review

1. **Spec coverage:** дизайн → задачи: раннер (1, 2), CLI+kill switch (2),
   документация (3), живой прогон + 9.4 (4). Non-goal (запись сессии,
   цикл бота) не планируется — ок.
2. **Step scan:** каждый шаг — одно действие с проверкой; тела кода не
   расписаны, где сигнатуры и тесты их определяют.
3. **Type consistency:** `run_probe`/`StepReport`/`ProbeReport`/`_resolve_model`/
   `main` единообразны между задачами 1 и 2; поля результатов сверены с
   модулями (core.py, rows.py, scroll.py).
4. **Review Focus:** пять пунктов, каждый привязан к тесту или помечен
   live-only с обоснованием.
5. **Proportion:** план короче дизайна + кода вместе; код-блоки — только
   тестовые имена и команды.
