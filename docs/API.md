# API-справочник: события ядра (задача 7.3)

Справочник — контракт полезной нагрузки событий (spec core-api-events:
«полезная нагрузка фиксируется в API-справочнике»). Пины payload'ов —
`tests/test_event_api.py::TestPayloadContract` и поведенческие тесты
источников; расхождение кода со справочником — дефект.

## Дисциплина потоков (ruling R14)

Ратифицировано отклонение от буквы design 8 «доставка в поток стратегии»:

- **push (`on`)**: callback исполняется **dispatcher-потоком** (не observer-
  потоком, не потоком стратегии). Буквальная доставка в поток стратегии
  требовала бы event pump'а в нём — противоречие блокирующей модели API.
  Исключение callback'а изолируется: остальные подписчики получают событие.
- **pull (`wait_event`)**: исполняется **строго в потоке стратегии**
  (thread-owner check; чужой поток — RuntimeError).
- Callback push-формы **не должен вызывать блокирующий API**: его поток —
  dispatcher, не поток владельца, API поднимет RuntimeError (thread-owner).
  Подписка/отписка (`on`, `subscription.unsubscribe`) из любого потока
  допустимы (не блокируют).

## Две формы

```python
api = bot.api                       # L2CoreAPI (тот же объект: bot.event_api)

# push: подписка, доставка dispatcher-потоком
sub = api.on("sold", lambda event: ...)
...
sub.unsubscribe()                   # idempotent; также with api.on(...) as sub:

# pull: блокировка до события/таймаута, строго из потока стратегии
event = api.wait_event("sold", timeout=30.0)   # Event | None (None — таймаут)

# журнал: записи ядра с seq >= since_seq
for entry in api.get_events(since_seq=0):      # JournalEntry(event, timestamp)
    ...

# публикация внешними источниками (kill switch 8.1 — единый канал)
bot.push("killed", {"reason": ...})            # == bot.event_api.push(...)
```

`Event`: `type` (str), `payload` (по таблице ниже), `seq` (монотонный номер
публикации, порядок доставки == порядок публикации).

Семантика `wait_event`: возвращает первое событие заданного типа,
опубликованное **после входа в вызов**; события «из прошлого» (в т.ч. уже
забранные dispatcher'ом из очереди) не возвращаются — для истории есть
журнал. Ожидание строится по журналу (пишется на push, до потребления), так
что pull не теряет события независимо от dispatcher'а.

## Типы событий и полезная нагрузка

| Тип | Источник | Payload |
|---|---|---|
| `sold` | монитор лавки (observer, 6.3) | `SoldPayload` |
| `window_closed` | observer (7.2) | `{"window": str}` — исчезнувшее окно |
| `modal` | перехватчик модалок (observer, 3.6) | `ModalReport` |
| `recovered` | `api.recover()` по успеху цепочки | `{"target": str, "steps": tuple[str, ...]}` |
| `escalation` | `api.recover()` по эскалации цепочки; observer при устойчивой деградации (8.4) | `{"target": str \| None, "detail": str, "steps": tuple[str, ...]}` |
| `killed` | kill switch (8.1, `L2CoreBot(kill_switch=...)`) | `{"reason": "hotkey" \| "inactivity"}` |

### `sold` — `SoldPayload` (l2core/scenarios/monitor.py, 6.3)

| Поле | Тип | Смысл |
|---|---|---|
| `sold_count` | int | суммарно продано (показатель окна) на момент снимка |
| `sold_delta` | int | приращение относительно линии, давшей событие |
| `revenue` | int \| None | суммарная прибыль; None — регион не настроен/не прочитан (деградация, не ноль) |
| `revenue_delta` | int \| None | приращение прибыли; None вместе с `revenue` |
| `snapshot_id` | str | sha1 байтов кадра-источника (тот же источник, что у RowsResult 6.1) |

### `window_closed` — dict

`{"window": <id окна торговли>}` — открытое окно исчезло со всех кадров
observer'а (market / shop_buy / shop_sell).

### `modal` — `ModalReport` (l2core/detector/modal.py, 3.6)

| Поле | Тип | Смысл |
|---|---|---|
| `detected` | bool | True (событие публикуется только при срабатывании) |
| `state` | str | `"modal.unknown"` |
| `region` | (x, y, w, h) \| None | регион кандидата в пикселях кадра |
| `reason` | str \| None | диагностическая причина срабатывания |
| `snapshot_path` | Path \| None | PNG-снимок кандидата (None без log_dir) |

### `recovered` / `escalation` — dict

Публикуются `api.recover()` по итогу recover-цепочки 5.8/6.4 (успех →
`recovered`; исчерпание цепочки/неоткрытие окна → `escalation`; отмена —
событий нет) и state observer'ом при устойчивой деградации (safety 8.4,
см. «Ошибки и эскалации» ниже). `target` — восстанавливаемое окно
(None при observer-эскалации от модалки: целевого окна нет); `steps` —
имена шагов цепочки (кортеж; у observer-эскалации пустой — цепочки не
было); `detail` — причина эскалации (`escalation_detail` у recover /
reason статуса 3.5 или модалки у observer). Событие дублирует не push-callback `on_escalation`
wiring'а (точка расширения вызывающего слоя), а журнально-очередной канал:
push-callback и событие — разные каналы, подключаются независимо.

### `killed` — dict

Источник — kill switch (8.1, l2core/api/killswitch.py): локальный хоткей
(`config.kill_switch.hotkey`) или глобальный таймаут неактивности
(`config.kill_switch.inactivity_timeout`). Публикуется движком
(`L2CoreBot(kill_switch=...)`, канал — push в очередь бота; подключение
ручной сборки — `ks.attach(on_kill=...)` / `bot.push("killed", ...)`).

| Поле | Тип | Смысл |
|---|---|---|
| `reason` | str | `"hotkey"` — аварийная остановка хоткеем; `"inactivity"` — таймаут неактивности |

Семантика: до публикации kill switch уже выставил cancelled-флаг (все
блокирующие вызовы прерываются с cancelled=True — CANCELLED-домен, не
UNKNOWN), выполнил `hid.release_all()` (best-effort) и увёл курсор в
`config.kill_switch.safe_point`. Kill идемпотентен: событие — ровно одно
за жизнь объекта KillSwitch. Остановку движка (join потоков) выполняет
стратегия: получила `killed` → `bot.stop()` (dispatcher должен остаться
живым до доставки самого события).

## Ошибки и эскалации (safety, 8.4)

Спека safety: осмысленная ошибка или событие эскалации при заблокированном
экране, оверлеях или невозможности активации окна; слепых кликов нет —
на каждой строке HID не тронут до подтверждения предусловий.

### Матрица отказов: что получает стратегия

| Состояние / причина | Сигнал стратегии | HID |
|---|---|---|
| Окно игры не найдено / свёрнуто / без клиентской области (3.2: `window_not_found` / `window_minimized` / `no_client_area`) | исключение `ActivateGameError` с `reason` (при действии); `get_state()` → `status=not_found`, `detail` = reason; `health().game_window.ok=False` с reason | не тронут (SetForeground не вызывался) |
| Фокус не получен: `SetForeground` исчерпал retry (`focus_not_acquired`) | исключение `ActivateGameError(reason="focus_not_acquired", detail)` — сигнал эскалации; recover-цепочка — решение вызывающего слоя | не тронут |
| Окно торговли не найдено/не стабильно при действии (статус ≠ `found`) | исключение `ClickPreconditionError` / `EnterPreconditionError` / `ScrollPreconditionError` / `DragPreconditionError` (5.3–5.6, пролог activate+found) | не тронут |
| Устойчивая деградация `occluded_or_unstable` (`anchor_lost` / `anchor_moved` / `anchor_appeared`, перекрытие тултипом/окном; ≥3 кадров подряд) | событие `escalation` от observer: `{"target": <окно>, "detail": <reason 3.5>, "steps": ()}` — одно за эпизод; при действии в этот момент — precondition-исключение | observer HID не получает; действия отклонены прологом |
| Неизвестная модалка-оверлей (3.6: `modal.unknown`) | событие `modal` (`ModalReport`, каждый цикл); устойчивая (≥3 циклов) → дополнительно одно `escalation` `{"target": None, "detail": ..., "steps": ()}` за эпизод | не тронут |
| Отказ примитива/сценария (`needs_recover`) | результат с `needs_recover=True` + диагностика `failures` → `api.recover()` → событие `recovered` или `escalation` | `release_all` первым шагом цепочки |
| UNKNOWN необратимого действия (`start_sell`) | страж 5.7: повтор блокирован (`IrreversibleRepeatError`) до явного `guard.confirm_after_unknown` | решение за стратегией |
| Kill switch (хоткей/таймаут неактивности) | событие `killed` + cancelled-флаг во всех блокирующих вызовах | `release_all()` best-effort, курсор в safe_point |

Пороги устойчивости observer'а — константы `_OCCLUDED_ESCALATION_FRAMES` /
`_MODAL_ESCALATION_CYCLES` (по 3), параметры конструктора `StateObserver`
(`occluded_escalation_frames` / `modal_escalation_cycles`) — точка подмены
для тестов и нестандартных сборок; ключей конфига нет (события редки).

### Диагностируемость reason-ов

- `ActivateGameError.reason` — один из `window_not_found`,
  `window_minimized`, `no_client_area` (находилка 3.2) либо
  `focus_not_acquired` (исчерпан retry SetForeground); `detail` — человеку
  читаемая диагностика (foreground/hwnd при focus_not_acquired).
- `WindowStatus.reason` — `anchor_not_found` / `anchor_lost` /
  `anchor_appeared` / `anchor_moved` (3.5) или reason находилки окна игры
  (окно торговли вне кадра из-за недоступности игры).
- Событие `escalation` от observer несёт `target` (окно или None при
  модалке) и `detail` (reason статуса/модалки) — достаточно для
  разбора без записанных кадров (модалка дополнительно пишет снимок,
  3.6).

## Журнал событий

Ведётся ядром на публикации (до потребления очереди): кольцевой буфер,
глубина `DEFAULT_JOURNAL_LIMIT` = 1000 записей (константа: события редки —
десятки в час; вытеснение старых записей штатно). Запись `JournalEntry`:
`event` (Event) + `timestamp` (момент публикации; в проде реальное время,
диагностика гонок, не контракт payload). Доступ: `api.get_events(since_seq)`
— записи с `seq >= since_seq`.

## Пример цикла стратегии (pull)

```python
while True:
    event = api.wait_event("sold", timeout=60.0)
    if event is None:
        continue                       # таймаут: нет продаж, цикл продолжается
    payload = event.payload            # SoldPayload по справочнику
    log.info("продано +%s (всего %s)", payload.sold_delta, payload.sold_count)
```
