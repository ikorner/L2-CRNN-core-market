# Формат YAML UI-модели (задача 2.1)

UI-модель — декларативное описание окон торговли: «где и как выглядит»
(design.md, решение 1). Потребители: конвертер (2.2), валидатор (2.3),
table_geometry (2.4), детектор состояний (3.4–3.5), грамматики OCR (4.4),
сценарии (6.x). Читаемость для человека-разметчика важнее лаконичности схемы.

## Гранулярность файлов (зафиксированное решение)

- `ui_model/common.yaml` — общий словарь: варианты элементов, роли,
  грамматики OCR. Не является моделью окна.
- `ui_model/<window>.yaml` — одна модель на окно, `<window>` — одно из
  `market`, `shop_buy`, `shop_sell`.

Каждая модель окна обязана содержать секции `window`, `elements`, `states`,
`transitions`, `ocr_regions`; `table_geometry` — для окон со списком
(обязательность по конкретному окну проверяет валидатор 2.3, парсер
принимает модель и без неё).

Неизвестные ключи на любом уровне отклоняются парсером — опечатка
разметчика не должна молчать.

## Координаты

Все координаты — пиксели референсного разрешения 1:1, без масштаба.

- `rect` обычного элемента — от якоря своего окна (решение 4).
- `search_area` HUD-элемента (`hud: true`) — допустимая область поиска
  в клиентской области игры, а не фиксированное смещение: панели игры
  перетаскиваются, HUD ищется template matching'ом по этой области.
- `click_point` — в системе координат родительского поля элемента
  (от якоря окна для `rect`, от угла `search_area` для HUD).
- Шаблоны (`template`) — пути относительно каталога `assets/` ядра.

## window

```yaml
window:
  id: market            # market | shop_buy | shop_sell
  title: "Обзор лавок"  # подпись окна (поиск якоря, диагностика)
  size: [w, h]          # размер окна в референсе (вписывание rect'ов)
```

## elements

Словарь `id -> элемент`. Роли: `click_target`, `ocr_region`,
`state_signature`, `icon_sample`. Роли-сырьё meta.json (`decor`,
`ocr_ground_truth`) в модель ядра не переносятся — это данные датасета.

```yaml
elements:
  tab_home:
    rect: [x, y, w, h]     # от якоря окна
    role: click_target
    variants:              # кроп-шаблон на вариант
      normal: {template: market/tab_home_normal.png}
      active: {template: market/tab_home_active.png, click_point: [x, y]}
  market_pic:              # HUD-элемент навигации
    role: icon_sample
    hud: true
    search_area: [x, y, w, h]  # область поиска в клиентской области игры
    variants:
      normal: {template: hud/market_pic.png, click_point: [x, y]}
```

Варианты (`normal`, `hover`, `pressed`, `disabled`, `active`,
`placeholder`, `focused`, `filled`) — словарь из `common.yaml`.

## states

Параметризованные состояния `(window, params)` (решение 9): сигнатуры
отделены от значений параметров. Признаки-сигнатуры образуют иерархию
`pixel_probe → template → ocr`; минимум два независимых признака на
состояние проверяет валидатор 2.3.

```yaml
states:
  list_home:
    params:              # свободный словарь строк
      tab: home
    signatures:
      - {kind: pixel_probe, points: [{at: [x, y], color: [r, g, b]}]}
      - {kind: template, element: tab_home, variant: active}
      - {kind: ocr, region: window_title, expect: "Обзор лавок"}
```

Проверка (check) — один из трёх видов, обязательные поля по виду:

- `pixel_probe`: `points` — список `{at: [x, y], color: [r, g, b]}`;
- `template`: `element` (id элемента), `variant` (вариант элемента);
- `ocr`: `region` (id OCR-региона), `expect` (ожидаемый текст).

## transitions

Простые переходы декларативно: `(state, action) -> (expected, verify,
timeout, on_fail[])`. Действие — `click` (поле `element`) или `key`
(поле `key`). `timeout` опционален: не задан — берётся из конфига при
исполнении. Семантика шагов `on_fail` определяется кодом recover
(примитивы), парсер имена не ограничивает.

```yaml
transitions:
  - from: list_home
    action: click
    element: tab_equipment
    expected: list_equipment
    timeout: 5.0
    verify:
      - {kind: template, element: tab_equipment, variant: active}
    on_fail: [press_esc, reopen_market]
```

## ocr_regions

Словарь `id -> регион`; `grammar` — имя грамматики из `common.yaml`.

```yaml
ocr_regions:
  window_title:
    rect: [x, y, w, h]   # от якоря окна
    grammar: text
  price_cell:
    rect: [x, y, w, h]
    grammar: price
```

## table_geometry

Константы нарезки таблицы (общие с пайплайном обучения, решение 12):

```yaml
table_geometry:
  col_seps: [x1, x2, ...]   # x разделителей колонок, от якоря окна
  row_y0: y                 # y первой строки
  row_pitch: p              # шаг строк
  max_rows: n
  columns:                  # зоны текста внутри ячейки колонки
    item:  {zone: [x, y, w, h], grammar: text}
    price: {zone: [x, y, w, h], grammar: price}
```

## common.yaml

```yaml
variants: [normal, hover, pressed, disabled, active, placeholder, focused, filled]
roles: [click_target, ocr_region, state_signature, icon_sample]
grammars:
  digits: {kind: digits}
  price: {kind: price, separators: [",", "."], suffixes: {"к": 1000, "кк": 1000000}}
  text: {kind: text}
```

Виды грамматик: `digits` (только цифры), `price` (разделители и суффиксы
`к`/`кк`), `text` (свободный текст). Парсер может работать и без явного
`common.yaml` — тогда используется встроенный словарь, совпадающий с
поставляемым `common.yaml` (проверяется тестом).

## Ошибки формата

Любая ошибка структуры — `l2core.model.ModelFormatError` с именем файла,
секцией и деталью (сообщение на русском). Падение загрузки обязательно;
«починить модель» — задача разметчика, не рантайма.
