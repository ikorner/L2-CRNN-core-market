# Spec Delta: action-primitives

## ADDED Requirements

### Requirement: Примитив press_key нажимает клавишу с verify и трёхзначным исходом
Система SHALL предоставлять примитив `press_key(key, verify, on_fail, ...)` (5.9): пролог активации окна игры, `hid.key_press`, блокирующий `wait_until` на verify свежих кадров, retry (n из конфига) → `on_fail[]` → общий recover. Исходы по инварианту 4: verify не сошёлся за таймаут → FAILED (эффект отсутствует, повтор безопасен); HID-сбой при нажатии → UNKNOWN (`unconfirmed`). Никаких `sleep` — только `wait_until` с таймаутами из конфига.

#### Scenario: Нажатие с подтверждением результата
- **WHEN** `press_key("X", verify=<сигнатуры menu_open>)` и меню открылось на свежих кадрах
- **THEN** результат OK, действие подтверждено verify

#### Scenario: Клавиша нажата, эффект отсутствует
- **WHEN** `press_key("X", ...)` отработал, но verify не сошёлся за таймаут из конфига
- **THEN** результат FAILED, после retry и шагов `on_fail` — общий recover

#### Scenario: HID-сбой при нажатии
- **WHEN** `hid.key_press` бросил `HidError`
- **THEN** результат UNKNOWN с `unconfirmed=True` (нажатие могло уйти, могло нет)

### Requirement: HID-адаптер знает кейкод клавиши X
HID-адаптер SHALL предоставлять кейкод `KEY_X` (Virtual-Key `X`) для `key_press`. Кейкоды цифр и F-клавиш не добавляются (раскладка quick-slot свободная — открытие лавок кликом по найденной иконке).

#### Scenario: Нажатие X доступно через адаптер
- **WHEN** примитив запрашивает `key_press` с `KEY_X`
- **THEN** адаптер отправляет корректный Virtual-Key код клавиши X
