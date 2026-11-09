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
| `escalation` | `api.recover()` по эскалации цепочки | `{"target": str, "detail": str, "steps": tuple[str, ...]}` |
| `killed` | kill switch (8.1) через `bot.push` | `dict` (определяется 8.1; инфраструктура готова) |

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
событий нет). `target` — восстанавливаемое окно; `steps` — имена шагов
цепочки (кортеж); `detail` (escalation) — причина эскалации
(`escalation_detail`). Событие дублирует не push-callback `on_escalation`
wiring'а (точка расширения вызывающего слоя), а журнально-очередной канал:
push-callback и событие — разные каналы, подключаются независимо.

### `killed` — dict

Источник — kill switch (8.1, вне этой задачи); инфраструктура типа и канал
публикации (`bot.push`) готовы. Формат payload определит 8.1.

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
