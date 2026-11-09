# l2core — ядро торгового бота Lineage 2

Драйвер трёх внутриигровых окон торговли: «Обзор лавок» (market — только
разведка цен), «Торговая лавка (Купить)» (shop_buy — лавка-выкуп) и
«Торговая лавка (Продажа)» (shop_sell). Всё знание об UI вынесено в
машиночитаемую UI-модель (`ui_model/`); ядро — её интерпретатор плюс
примитивы и сценарии. Стратегии торговли (что покупать/продавать и по каким
ценам) — сменные модули поверх стабильного API и в этот репозиторий не входят.

Статус: в разработке. Готовы адаптеры, детекторы, OCR, примитивы, сценарии
и синхронный публичный API (задача 7.1); события/модель потоков (7.2–7.3) и
готовые стратегии — ещё нет (план —
`openspec/changes/core-market-driver/tasks.md`).

## Требования

- Windows 10/11 (Win32 API: активация окна, клиентская область, DPI).
- Python 3.10+.
- Lineage 2 в **оконном режиме**, клиентская область 1920×1080 (референс).
- Запуск **с правами не ниже игры** (иначе SetForeground и ввод не
  сработают — см. «Типичные проблемы»).

## Установка

```bash
git clone <repo> && cd L2-CRNN-core-market
pip install -e ".[dev]"
```

Зависимости: mss (захват экрана), numpy, opencv-python (template matching),
torch (OCR in-process, CPU-режим обязан работать), pygetwindow + pywin32
(окно/фокус), PyYAML (конфиг и UI-модель). Чекпоинт CRNN `crnn_best.pt` в git
не входит: положите файл рядом с корнем репозитория либо укажите путь в
`config.yaml` (`paths.model`).

## Конфигурация

Все настройки — в `config.yaml` (схема и дефолты: `l2core/config.py`).
Файл может быть частичным: неупомянутые значения берутся из дефолтов.
Неизвестные ключи и значения вне диапазонов отклоняются с `ConfigError`.

Секции:

- `timing` — таймаут `wait_until` по умолчанию, интервал verify, пауза
  после hover, число повторов клика.
- `ocr` — порог confidence (0.85) и число перечитываний.
- `behavior` — поведенческий профиль (антидетект): доли способов ввода
  цены (экранный нумпад/клавиатура), Enter против кнопки «Поиск», джиттеры
  клика и траектории, человеческие задержки нажатий.
- `paths` — путь к чекпоинту CRNN и порт HID-адаптера arduino.
- `kill_switch` — хоткей мгновенной остановки (`ctrl+shift+k`, слушается
  локально, не через arduino), глобальный таймаут неактивности (сек) и
  безопасная точка курсора `safe_point: [x, y]` при kill.
- `resolution` — референсное разрешение клиентской области (1920×1080).
- `dry_run` — см. ниже.

Проверка конфига без запуска:

```bash
python -c "from l2core.config import load_config; load_config('config.yaml')"
```

## Dry-run

```yaml
dry_run: true
```

Экран читается, состояние окна определяется, но все действия только
логируются — обращений к HID нет. Режим предназначен для первых запусков на
живой игре: проверяете детекцию и стабильность без риска случайных кликов.

Реализация: `create_api` поднимает `DryRunHid` (l2core/api/dryrun.py) вместо
живого `HidDevice` (arduino не подключается). Явный параметр
`create_api(..., dry_run=True)` побеждает флаг конфига. Намерения действий
пишутся в лог `l2core.dry_run` и доступны в `api._w.hid.intentions`; флаг
режима — `api.dry_run`.

Так как действие не выполняется, verify (честный, по реальным кадрам) не
видит эффекта: результаты примитивов уходят в FAILED/UNKNOWN — это
ожидаемый сигнал режима, а не сбой. success возможен, только если эффект и
так присутствовал на экране (например, окно уже открыто).

## Запуск

Публичный API — синхронный фасад `L2CoreAPI` (`l2core/api/`), собираемый
фабрикой `create_api` (живые адаптеры: MssScreen, HidDevice, TorchOcr,
PyWin32Gateway, детекторы). Модуль стратегии создаёт API в своём потоке и
вызывает блокирующие методы только из него (enforcement — проверка
потока-владельца, чужой поток → RuntimeError):

```python
from l2core.api import create_api

api = create_api("config.yaml")          # владелец — поток конструктора
state = api.get_state()                  # (window, state, params, status)
opened = api.open_market()               # клик по HUD + ожидание сигнатуры
if opened.needs_recover:
    api.recover(opened, "market", hud_element="market_pic")
rows = api.read_rows()                   # RowsResult со snapshot_id
api.scroll_list("down", 3)
api.close_market()
```

Навигация возвращает `NavigationResult` с трёхзначным `outcome`
(OK / FAILED / UNKNOWN — адаптер 5.7): UNKNOWN означает «эффект, скорее
всего, ушёл, подтвердить нельзя» — повтор потенциально необратимого до
явного `api.guard.confirm_after_unknown(...)` запрещён. Регионы, которых нет
в отгруженной UI-модели (поле поиска, поле цены, показатели лавки),
передаются в `create_api` параметрами `search_region` / `price_region` /
`shop_regions` (см. KNOWN_GAPS и докстринги `l2core/api/`).

Готовый цикл бота как CLI/сервис — вне текущего объёма; репозиторий пока —
библиотека + тесты (API 7.1, потоки 7.2, события 7.3, kill switch 8.1 —
реализованы, сборку выполняет вызывающий слой):

```bash
python -m pytest
```

## Kill switch

При срабатывании хоткея (`ctrl+shift+k`) или таймаута неактивности:
мгновенный стоп цикла, прерывание блокирующих вызовов с возвратом
UNKNOWN/CANCELLED, освобождение HID (`release_all`), увод курсора в
безопасную точку (`kill_switch.safe_point`, по умолчанию угол экрана —
вне клиентской области игры), событие стратегии `killed`
(`{"reason": "hotkey" | "inactivity"}`, см. docs/API.md).

Компонент — `l2core/api/killswitch.py` (`KillSwitch`); хоткей слушается
низкоуровневым Win32-хуком WH_KEYBOARD_LL (не через arduino; источник
клавиш — инъекция, тесты на фейках). Каноническая сборка — через движок
(`build_wiring` возвращает ApiWiring; `create_api` — фасад без движка,
для него `cancelled` передаётся явно):

```python
from pathlib import Path
from l2core.adapters.hid import HidDevice
from l2core.adapters.screen import MssScreen
from l2core.adapters.template import default_matcher
from l2core.adapters.win32 import (GameWindowFinder, PyWin32Gateway,
                                   Win32HotkeySource, ensure_dpi_aware)
from l2core.api.engine import L2CoreBot
from l2core.api.killswitch import KillSwitch
from l2core.api.wiring import build_wiring
from l2core.model import load_model

config = load_config("config.yaml")
gateway = PyWin32Gateway()
ensure_dpi_aware(gateway)
ks = KillSwitch(config.kill_switch,
                hotkey_source=Win32HotkeySource(config.kill_switch.hotkey),
                set_cursor_pos=gateway.set_cursor_pos)
models = {name: load_model(f"ui_model/{name}.yaml")
          for name in ("market", "shop_buy", "shop_sell")}
hid = HidDevice(config.paths.arduino_port); hid.connect()
wiring = build_wiring(
    config=config, models=models, screen=MssScreen(), hid=hid, ocr=...,
    matcher=default_matcher(), assets_dir=Path("assets"),
    cursor_pos=lambda: gateway.get_cursor_pos() or (0, 0),
    activate=..., game_finder=GameWindowFinder(gateway, title=config.window.title))
bot = L2CoreBot(wiring, kill_switch=ks)  # hid + killed + cancelled движок
api = bot.api                            # тот же guard-wiring (7.1)
ks.start(); bot.start()
# стратегия получила killed -> bot.stop() (join потоков — за стратегией)
```

Kill идемпотентен и необратим. Активность потока стратегии фиксируют
вызовы API (`wiring.note_activity`) — они продлевают таймаут неактивности.
Движок с `kill_switch` подставляет `ks.is_killed` в wiring.cancelled сам
(явный `wiring.cancelled` сборки побеждает) — прерывание блокирующих
вызовов не зависит от ручной передачи cancelled.

## Типичные проблемы

- **SetForeground не активирует окно** — запустите бота с правами не ниже
  игры (например, игра и бот оба «от имени администратора»). Оверлей или
  заблокированный экран: ядро вернёт диагностируемую ошибку/эскалацию и не
  будет кликать вслепую.
- **Координаты «плывут» / клики мимо** — процесс должен быть DPI-aware
  (объявляется ядром, задача 3.2); убедитесь, что масштабирование Windows для
  монитора = 100% или что игра, и бот видят одну и ту же координатную сетку.
- **OCR возвращает низкий confidence** — проверьте `ocr.confidence_threshold`
  и `ocr.reread_count`; мигающая каретка в поле ввода лечится стабилизацией
  кадра (два одинаковых чтения).
- **Игра свёрнута или окно перекрыто** — драйвер работает только с видимым
  неперекрытым окном в оконном режиме; статус окна
  `occluded_or_unstable` означает «подождать/поднять окно», а не кликать.

## Разработка

Инварианты проекта (никаких sleep — только `wait_until`, verify на каждое
действие, координаты только от якоря, трёхзначный результат OK/FAILED/UNKNOWN,
TDD, детерминизм тестов, код на русском) и процесс обновления модели после
патчей игры — в `AGENTS.md`.
