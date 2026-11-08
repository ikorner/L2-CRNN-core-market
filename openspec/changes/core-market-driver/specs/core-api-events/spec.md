# Spec: core-api-events

## ADDED Requirements

### Requirement: Синхронный блокирующий API для стратегий
Ядро SHALL предоставлять стратегиям синхронные блокирующие вызовы: навигация (`open_market/close_market`, `open_shop_buy`, `open_shop_sell`), управление обзором (`set_tab`, `set_filter`, `search`, `scroll_list`), чтение (`read_rows`), лавка продажи (`setup_sell`, `start_sell`, `cancel_sell`, `read_shop_state`), сервисные (`get_state`, `recover`, `health`); стратегия MUST вызывать блокирующий API только из своего потока.

#### Scenario: Открытие обзора
- **WHEN** стратегия вызывает `open_market()`
- **THEN** выполняется клик по иконке HUD и блокирующее ожидание сигнатуры окна, по таймауту — FAILED/UNKNOWN

#### Scenario: Чтение строк обзора
- **WHEN** стратегия вызывает `read_rows()`
- **THEN** возвращается список `Row` видимых строк со `snapshot_id` текущего кадра

### Requirement: События в двух формах — push (callback) и pull (wait_event)
Ядро SHALL поддерживать подписку на события через `on(callback)` (доставка в поток стратегии) и блокирующее `wait_event(type, timeout)` (pull); стратегия SHALL иметь возможность выбрать стиль сама; типы событий SHALL включать `sold`, `window_closed`, `modal`, `recovered`, `escalation`, `killed`, полезная нагрузка фиксируется в API-справочнике.

#### Scenario: Pull-ожидание события продажи
- **WHEN** стратегия вызывает `wait_event("sold", timeout)`
- **THEN** вызов блокируется до события `sold` или таймаута, полезная нагрузка соответствует справочнику

#### Scenario: Push-подписка
- **WHEN** стратегия подписалась через `on(callback)` и произошло событие `modal`
- **THEN** callback вызван в потоке стратегии с полезной нагрузкой события

### Requirement: Модель потоков: executor, observer, dispatcher
Система SHALL разделять `action executor` (единственный владелец HID, исполняющий действия стратегии), `state observer` (фоновый поток, читающий экран и не отправляющий действия, питающий события) и `event dispatcher` (очередь событий с доставкой callbacks); поток стратегии SHALL быть единственным вызывающим блокирующий API.

#### Scenario: Observer не отправляет действия
- **WHEN** state observer анализирует кадры в фоне
- **THEN** ни один вызов HID не происходит из потока observer

### Requirement: Фокус и активность окна перед каждым действием
Перед каждым действием ядро SHALL проверять, что окно игры активно и не перекрыто; при необходимости SHALL выполняться активация через Win32 (SetForeground, retry + эскалация); неудача SHALL вести в recover/паузу, не в слепые клики.

#### Scenario: Окно не активно
- **WHEN** перед действием обнаружено неактивное/перекрытое окно
- **THEN** выполняется активация через Win32, при неудаче — retry и эскалация

### Requirement: Журнал событий ведётся ядром
Ядро SHALL вести журнал событий (продано, окно закрыто, модалка, recover-эскалация, kill switch) с полезной нагрузкой, доступный стратегии через API событий.

#### Scenario: Событие попадает в журнал
- **WHEN** произошло событие `recovered`
- **THEN** оно доступно через pull-ожидание и callback-подписку
