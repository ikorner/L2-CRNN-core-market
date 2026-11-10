# API, потоки, события, безопасность (задачи 7.1–7.3, 8.1–8.4)

> Детальный справочник по слою `l2core/api/`, вынесенный из AGENTS.md
> (прогрессивная загрузка: AGENTS.md — ядро, этот файл — детали по слою).
> Правится вместе с кодом: обновление затронутых заметок `docs/agents/` —
> часть DoD ломающих задач (см. «Правила изменений» в AGENTS.md).

## Синхронный блокирующий API для стратегий (задача 7.1, spec
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
- **YAGNI**: никакого event dispatcher (7.3), kill
  switch (8.1) — здесь только синхронный фасад. Потоки — L2CoreBot (7.2,
  engine.py ниже), не здесь.
- Тесты: tests/test_api.py (контракт всех методов на скриптованном мире:
  делегирование, OK/FAILED/UNKNOWN, thread-owner, отмена, страж),
  tests/test_wiring.py (математика провайдеров, create_api с подменой
  адаптеров на реальных моделях).

## Модель потоков: executor, observer, dispatcher (задача 7.2, spec
core-api-events, сценарий «Observer не отправляет действия»)

`l2core/api/engine.py` — L2CoreBot (orchestrator потоков) и компоненты
StateObserver / EventDispatcher; очередь — `l2core/api/events.py`
(EventQueue/Event, типы `sold`, `window_closed`, `modal`, `recovered`,
`escalation`, `killed` — инфраструктура типы знает, источники
подключаемы); HID-owner guard — `l2core/api/hid_owner.py`
(вендоренный `adapters/hid.py` не трогается — обёртка снаружи).

- **Action executor — поток стратегии (вариант Б из design-развилки)**:
  API синхронен и блокирующий (7.1) и по спеке вызывается только из
  потока стратегии — отдельный executor-поток с очередью команд при
  однопоточной стратегии потребителя не имеет (async-слой без выгоды).
  Владение HID — ИСПОЛНЯЕМОЕ: wiring.hid оборачивается OwnerGuardedHid
  (владелец — поток конструктора L2CoreBot); HID-команда из чужого
  потока → HidOwnershipError до железа (инвариант 9). Обёртка fail
  closed: публичного доступа к сырому HidDevice нет (`inner` отсутствует,
  `__getattr__` пробрасывает только метаданные) — обхода проверки
  владельца через публичную поверхность нет; паритет с публичными
  методами вендорного HidDevice следит guard-тест
  tests/test_engine.py::TestHidOwnerContract (новая команда вендора —
  падение с именем метода). Граница владения: гарантия — для доступа
  через guard-сборку (bot.wiring.hid); исходный wiring, собранный
  снаружи вручную (ApiWiring(...)/build_wiring), — зона ответственности
  вызывающего слоя (эталонная сборка: create_api → wiring → L2CoreBot).
  Многопоточные стратегии — не-цель v1.
- **State observer — фоновый поток, только чтение**: grab, детекция
  окна/статуса/состояния (3.4/3.5), перехватчик модалок (3.6), монитор
  лавки (6.3). HID observer не получает ВООБЩЕ (в конструктор не
  принимается) — сценарий спеки выполнен по построению (интеграционно:
  wiring.hid=RecordingHid после прогона движка чист). Публикует в
  EventQueue: `window_closed` (открытое окно исчезло), `modal`
  (перехватчик сработал), `sold` (монитор детектировал продажу),
  `escalation` (устойчивая деградация 8.4). Пауза
  между циклами — wait_until с предикатом stop/cancelled (инвариант 1,
  kill switch 8.1 через cancelled); интервал — `timing.poll_interval`
  (новый ключ конфига, дефолт 1.0 с). Внутреннее состояние observer'а —
  допустимое исключение (монитор, design 8): последний снапшот
  (`observer.snapshot`: окно/статус/состояние/деталь) и факт открытого
  окна для детекции закрытия. Исключение итерации не роняет поток
  (лог, цикл продолжается — наблюдатель переживает деградации).
- **Event dispatcher — поток доставки**: EventQueue → callbacks
  (`dispatcher.subscribe/unsubscribe`); callback получает Event
  (type/payload/seq). Callbacks исполняются В ПОТОКЕ DISPATCHER'А
  (отклонение от буквы design 8 «в поток стратегии»: поток стратегии
  блокирован в API-вызовах, event pump противоречит блокирующей модели;
  зафиксировано в отчёте 7.2) — ключевое требование «не в observer-
  потоке» выполнено, pull-форма, исполняемая строго в потоке стратегии,
  — `wait_event` (7.3) поверх той же очереди. Исключение callback'а
  изолируется: лог, остальные доставлены; поток не умирает от чужого
  кода. Callback не должен вызывать блокирующий API (чужой поток →
  RuntimeError по thread-owner 7.1).
- **L2CoreBot**: создаётся в потоке стратегии с готовым ApiWiring;
  `bot.api` — фасад поверх guard-сборки (thread-owner 7.1),
  `bot.events` — очередь, `bot.push(type, payload)` — внешние источники
  (escalation от recover через on_escalation wiring'а, killed от 8.1);
  `start()` порождает оба потока, `stop(timeout)` — stop-флаги + join
  (idempotent; повторный start — EngineError, жизненный цикл один);
  контекстный менеджер. Доступ к API — только через `bot.api`: прямое
  использование исходного wiring с сырым HidDevice обходит guard
  (граница зафиксирована в докстринге engine.py).
- Kill switch (8.1) подключается параметром `kill_switch`: движок
  передаёт guard-сборку HID (release_all при kill — санкционированное
  исключение guard'а владельца, см. killswitch.py) и публикует событие
  killed; cancelled подставляется в копию wiring автоматически (явный
  wiring.cancelled побеждает). Без явного `cancelled` observer проверяет
  kill-флаг и завершает цикл сам. AST-guard'а на observer нет — сценарий
  «Observer не отправляет действия» покрыт интеграционно
  (tests/test_engine.py).
- Тесты: tests/test_engine.py (очередь: порядок/seq/неизвестный тип/
  конкурентность; guard: делегирование/чужой поток/явный владелец;
  dispatcher: порядок/изоляция исключений/unsubscribe/join; observer:
  window_closed/sold/modal/стабильность/отмена до кадра; движок:
  start/stop/restart-rejected/контекст/thread-owner/guard в wiring/
  observer HID не тронул). Потоки в тестах короткоживущие (интервалы
  0.01–0.05 с), join в каждом тесте — pytest не висит.

## События двух форм и журнал (задача 7.3, spec core-api-events,
инвариант 4)

`l2core/api/events.py`: EventJournal (кольцевой буфер, лимит
`DEFAULT_JOURNAL_LIMIT=1000` — константа, события редки; timestamp из
инъецированных часов, в проде реальное время) + EventAPI поверх (queue,
dispatcher). Журнал подключается к очереди: `EventQueue(journal=...)` —
запись на push, ДО потребления.

- **push `on(type, callback)` -> EventSubscription** (unsubscribe idempotent,
  контекстный менеджер): доставка — **dispatcher-поток** (ruling R14,
  ратификация отклонения от design 8 «доставка в поток стратегии» — буква
  требовала event pump'а против блокирующей модели API; правка спеки —
  commits 50cbb5e): не producer-поток, исключения callback'ов изолированы.
  Callback НЕ вызывает блокирующий API (dispatcher-поток ≠ поток владельца,
  RuntimeError по thread-owner).
- **pull `wait_event(type, timeout)` -> Event | None**: блокировка по
  журналу (события не теряются, забранные dispatcher'ом); строго из потока
  стратегии (owner_thread, чужой поток — RuntimeError, R14); возвращает
  первое событие типа, опубликованное ПОСЛЕ входа в вызов (прошлое —
  только через журнал); таймаут — None.
- **журнал `get_events(since_seq)`** — записи JournalEntry(event,
  timestamp) с seq >= since_seq; ведёт ядро (spec), доступен стратегии.
- **публикация**: единый канал — EventQueue.push. Источники: observer 7.2
  (sold/window_closed/modal), `api.recover()` по итогу цепочки 5.8/6.4
  (успех -> recovered `{"target", "steps"}`, эскалация -> escalation
  `{"target", "detail", "steps"}`, отмена — без событий), `bot.push` /
  `EventAPI.push` — внешним источникам. Типы: sold,
  window_closed, modal, recovered, escalation, killed (источник — kill
  switch 8.1 через движок).
- **API-справочник — docs/API.md** (контракт payload'ов): sold ->
  SoldPayload 6.3, window_closed -> `{"window"}`, modal -> ModalReport 3.6,
  recovered/escalation -> dict по справочнику. Пины:
  tests/test_event_api.py (28 тестов: журнал/лимит/since_seq, push-доставка/
  отписка/изоляция, pull-фильтр/таймаут/thread-owner/доставка после
  потребления dispatcher'ом, payload-поля, recovered/escalation от
  api.recover, интеграция с L2CoreBot).
- L2CoreAPI: методы `on/wait_event/get_events` и свойство `events`
  делегируют EventAPI (параметр ctor `events`; сборка без него — ApiError,
  fail fast); движок подключает: `bot.event_api` (== `bot.api.events`).

## Kill switch: локальный хоткей + таймаут неактивности (задача 8.1,
spec safety, решение 16 design.md)

`l2core/api/killswitch.py` — KillSwitch: слушатель в собственном потоке;
срабатывание — хоткей ИЛИ таймаут тишины (линию продлевают вызовы API:
каждый публичный метод фасада через `wiring.note_activity` фиксирует
активность executor'а). `kill(reason)` идемпотентен и необратим, эффекты
ровно одни: (1) флаг killed — `is_killed()` передаётся в сборку как
`cancelled` (wiring.cancelled — дефолт всех блокирующих вызовов 7.1:
wait_until/примитивы прерываются, результат — cancelled=True, outcome=None,
CANCELLED-домен, не UNKNOWN); (2) `hid.release_all()` best-effort; (3) увод
курсора в безопасную точку (`set_cursor_pos`, параметр побеждает
`kill_switch.safe_point` из конфига; оба не заданы — курсор не трогается +
диагностика; дефолт (8, 8) — угол экрана вне клиентской области игры);
(4) событие `killed` (`{"reason": "hotkey" | "inactivity"}`) через on_kill.

- **Канал хоткея** — инъекция `hotkey_source` (контракт start/stop/
  pressed); прод-реализация — `Win32HotkeySource` (adapters/win32.py):
  низкоуровневый хук WH_KEYBOARD_LL в собственном потоке (нажатие
  фиксирует сам хук, не опрос — короткое нажатие не пропадает между
  poll'ами); слушаем локальную клавиатуру, НЕ arduino. Парсинг строки
  хоткея («ctrl+shift+k»: a-z/0-9/f1-f12 + модификаторы) fail fast без
  Win32; хук — живой код (pragma no cover), ленивый импорт как в 3.2.
- **Guard владельца и kill**: слушатель — чужой поток, а владелец HID —
  поток стратегии, поэтому `OwnerGuardedHid(hid, kill_switch=ks.is_killed)`
  даёт САНКЦИОНИРОВАННОЕ ИСКЛЮЧЕНИЕ: release_all при поднятом kill-флаге
  выполняется из любого потока (идемпотентен, только отпускает кнопки);
  любая другая HID-команда из чужого потока при активном kill по-прежнему
  HidOwnershipError (fail closed). L2CoreBot(kill_switch=ks) собирает это
  сам: guard-сборка + attach + push killed в свою очередь; cancelled
  подставляется в копию wiring автоматически — приоритет: явный
  wiring.cancelled сборки > параметр cancelled > kill_switch.is_killed
  (headline-гарантия спеки «прерывание блокирующих вызовов» не зависит от
  ручной передачи cancelled в build_wiring). Без явного cancelled observer
  проверяет kill-флаг и завершает цикл сам. Движок kill НЕ останавливает:
  dispatcher должен дожить до доставки события killed; join потоков — за
  стратегией (killed -> bot.stop()).
- Ожидание в цикле слушателя — wait_until (инвариант 1), часы/сон —
  инъекции; `poll_once()` — детерминированная итерация проверок для тестов
  без потоков. Жизненный цикл один: повторный start — KillSwitchError,
  stop идемпотентен (kill-флаг не снимается).
- Win32Gateway дополнен `set_cursor_pos(x, y) -> bool` (PyWin32Gateway —
  win32gui.SetCursorPos, ленивый импорт).
- Конфиг: `kill_switch.safe_point` — пара неотрицательных int [x, y] |
  null (валидатор по схеме 1.3; shipped config.yaml == дефолты схемы).
- Тесты: tests/test_killswitch.py (38 тестов: хоткей/таймаут/эффекты/
  идемпотентность/жизненный цикл, прерывание wait_until и полного
  api.open_market из потока стратегии, дефолт wiring.cancelled движком
  (без ручного cancelled) и приоритет явного, активность через API,
  guard-исключение, интеграция с L2CoreBot, конфиг safe_point, парсинг
  хоткея).
  YAGNI: dry-run (8.2) и JSONL-лог (8.3) сюда не входят.

## Dry-run режим без HID (задача 8.2, spec safety «Dry-run режим без HID»)

`l2core/api/dryrun.py` — DryRunHid: самостоятельная реализация командного
интерфейса HidDevice без железа и без инъекции реального устройства.
Каждое действие (move_rel/click/scroll/keys/traj/release_all/connect/
close) фиксируется как намерение — структурированно в лог
`l2core.dry_run` и в список `intentions` в памяти (опционально внешний,
инъекция для тестов/стратегии); возвраты — как у успешного no-op, чтобы
примитивы проходили штатный путь до verify. Запросы состояния
(is_connected/ping/get_version) — не действия: не логируются, адаптер
отчитывается «живым» (True/(0, 0)) — в dry-run HID-путь доступен по
построению, health() без изменений. `TRAJ_MAX_POINTS` — паритет вендора
(валидация ёмкости traj), guard-тест паритета командной поверхности с
HidDevice — tests/test_dryrun.py (новая команда вендора в dry-run не
пропадёт молча).

- **Точка включения — create_api** (параметр `dry_run=None`): эффективный
  флаг — параметр, иначе `config.dry_run` (ключ уже в схеме 1.3).
  Включённый режим поднимает DryRunHid вместо живого HidDevice (arduino
  не трогается вообще — подключение/проверка прошивки пропускаются);
  явно переданный hid побеждает режим (инъекция адаптера для тестов).
  Флаг виден стратегии: `wiring.dry_run` и `api.dry_run`.
- **Экран читается по-настоящему**: MssScreen и детекторы работают как
  обычно, verify проверяет реальные кадры. Эффекта «клика» нет, поэтому
  результаты примитивов честно уходят в FAILED (verify выполнялся,
  эффект опровергнут) или UNKNOWN (якоря на кадрах не было); success
  возможен, только если эффект и так присутствовал на экране (окно уже
  открыто). Семантика для первых запусков на живой игре: детекция
  работает, действия бы не сработали — намерения смотрим в логе.
- **Thread-owner дисциплина не дублируется**: DryRunHid — устройство,
  guard владения — OwnerGuardedHid (7.2), L2CoreBot применяет его к
  wiring.hid независимо от режима (проверка владельца — одна точка;
  санкционированный release_all при kill switch 8.1 сохраняется: в
  dry-run это no-op с логом).
- Тесты: tests/test_dryrun.py (23 теста: намерения/лог/release_all/
  no-op-всех-команд/запросы/паритет traj и командной поверхности,
  create_api — флаг конфига/победа параметра/победа явного hid,
  композиция с OwnerGuardedHid, поведенческий open_market: FAILED/UNKNOWN/
  OK-уже-открыто, накопление намерений).
  YAGNI: JSONL-лог (8.3) и live-прогон (9.4) сюда не входят.

## JSONL-лог действий (задача 8.3, spec replay-logging «JSONL-лог каждого
действия», инварианты 1/7)

`l2core/api/actionlog.py` — ActionLogger: каждое действие ядра фиксируется
JSONL-записью (спека: кадры до/после, действие, verify-результат; размер
ограничивается ротацией, сохраняются только ключевые кадры).

- **Формат записи**: `{seq, ts, key, action: {type, target, args},
  frames: {before, after}, verify: {success, outcome, unconfirmed,
  cancelled, attempts, failures}}`. seq — сквозной счётчик (возобновляется
  по существующим файлам: перезапуск процесса журнал не обнуляет), ts —
  инъецированные часы (в проде монотонные секунды), action нормализуется
  `normalize_action` сессий 3.1. outcome — str-Enum 5.7 (JSON-сериализация
  по значению: «OK»/«FAILED»/«UNKNOWN»).
- **Хранилище** (детерминизм, конвенции 3.1/3.6): `log_dir/actions-NNNN.
  jsonl` (NNNN — счётчик ротации, меток времени в именах нет) + `log_dir/
  frames/<sha1>.png` — снимки по хэшу содержимого (дедуп бесплатно:
  одинаковый кадр — один файл).
- **Политика ключевых кадров** (спека «только ключевые кадры»): запись
  пишется всегда, PNG — только для ключевых: не-OK исходов (FAILED/UNKNOWN
  — для разбора) и необратимых действий (start_sell — контроль точки
  невозврата). Обычные успешные записи идут без кадров; отмена (kill
  switch 8.1) — не ключевая. Конфигурируемо: `log.policy {all_failures,
  irreversible}`.
- **Ротация**: запись дописана, размер текущего jsonl превысил
  `log.max_bytes` → следующий счётчик (запись не разрывается между
  файлами). Файлов старше `log.keep_files` не держим: ключевые записи
  удаляемого файла МИГРИРУЮТ в новейший (оригинальные seq/ts/frames) —
  ключевые кадры переживают ротацию вместе со своими записями; неключевые
  записи старших файлов удаляются вместе с файлами («удаление старых
  записей»). Сиротские PNG (ссылок из сохранённых записей нет — остатки
  прерванных сессий) чистятся при ротации.
- **Точка интеграции — методы фасада, не примитивы** (примитивы 5.x не
  переписываются): каждый действие-метод L2CoreAPI (open_*/close_market/
  set_tab/set_filter/search/scroll_list/setup_sell/start_sell/cancel_sell/
  recover — НЕ чтение: read_rows/read_shop_state/get_state/health)
  обёрнут `_log_action`: кадр «до» = grab на входе, «после» = grab на
  выходе. ГРАНИЦА (зафиксирована в отчёте 8.3): это кадры на границе
  ВЫЗОВА API, а не непосредственно вокруг HID-команды; точечная прибавка
  action_logger в каждый примитив отвергнута по минимальной инвазивности
  (метод API = одно действие ядра с одним verify-итогом). Fail-fast
  ошибки контракта (ApiError до действия) в журнал не пишутся — это не
  действия.
- **Точка включения — create_api** (параметр `action_logger=None`):
  при `config.log.enabled` (дефолт true) поднимает ActionLogger из
  `config.log` (dir/max_bytes/keep_files/policy; ts — от часов сборки);
  явный action_logger побеждает конфиг. Поле `ApiWiring.action_logger`
  (None — логирование off, поведение сборок без него не меняется). Запись
  синхронна в потоке вызова (поток стратегии, thread-owner 7.1) —
  конкурентности нет по построению.
- **Конфиг** (схема 1.3): секция `log` — `enabled`, `dir` (log_dir,
  дефолт `logs/actions`), `max_bytes` (ротация), `keep_files` (хранение),
  вложенная `policy {all_failures, irreversible}`; shipped config.yaml ==
  дефолты схемы.
- Тесты: tests/test_actionlog.py (38 тестов: формат записи, политика
  ключевых кадров, дедуп кадров по содержимому, ротация/keep_files/
  миграция ключевых записей/чистка сирот, возобновление счётчиков,
  детерминизм имён, конфиг (дефолты/YAML/отказоустойчивость), интеграция
  фасада (FAILED с кадрами, OK без кадров, start_sell необратимый,
  отмена), create_api (enabled/disabled/явный logger)).
  YAGNI: live-прогон (9.4) и чтение/анализ лога сюда не входят.

## Осмысленные ошибки при блокировках экрана и оверлеях (задача 8.4,
spec safety «Осмысленные ошибки при блокировках экрана и оверлеях»,
инвариант 9)

Аудит цепочек отказа + закрытие наблюдательного пробела. Действия
(прологи 5.2–5.6: activate OK + статус found, иначе исключение с reason,
HID не тронут) и recover-эскалация (6.4 → событие escalation 7.3) были
реализованы ранее; добавлено превращение УСТОЙЧИВОЙ деградации в
событие для стратегии (state observer 7.2, engine.py):

- **Устойчивая деградация статуса 3.5**: `occluded_or_unstable` N кадров
  подряд (anchor_lost/anchor_moved/anchor_appeared — перекрытие тултипом/
  окном, «заблокированный экран» в терминах ядра) → observer публикует
  ОДНО событие `escalation` за эпизод: `{"target": <окно>, "detail":
  <reason 3.5>, "steps": ()}` (формат payload 7.3); возврат к found/
  not_found снимает эпизод (новый эпизод — новое событие). Единичные
  всплески (межкадровый джиттер) — не эскалация.
- **Устойчивая неизвестная модалка** (3.6, modal.unknown не уходит N
  циклов) → дополнительно одно `escalation` `{"target": None, "detail":
  ..., "steps": ()}` за эпизод (событие modal продолжает идти каждый цикл).
- Пороги — константы `_OCCLUDED_ESCALATION_FRAMES` / `_MODAL_ESCALATION_
  CYCLES` (по 3) и параметры конструктора StateObserver
  (`occluded_escalation_frames` / `modal_escalation_cycles`, точка подмены
  тестов); ключей конфига нет (события редки, YAGNI).
- Матрица отказов (состояние/причина → исключение с reason / событие /
  результат needs_recover; и диагностируемость reason-ов ActivateGameError
  и WindowStatus) — контракт в docs/API.md, раздел «Ошибки и эскалации».
- Тесты: tests/test_engine.py::TestObserverSafetyEscalation (5 тестов:
  одиночная эскалация за эпизод, транзиенты без эскалации, rearm после
  восстановления/снятия модалки; «не кликая» — observer HID не получает
  по построению, hid движка чист в интеграционных тестах 7.2).
  YAGNI: live-прогоны (9.4/9.5) сюда не входят.
