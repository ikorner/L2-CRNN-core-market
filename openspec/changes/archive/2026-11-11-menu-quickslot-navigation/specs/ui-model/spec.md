# Spec Delta: ui-model

## MODIFIED Requirements

### Requirement: UI-модель описывает три торговых окна и HUD-элементы декларативно
Система SHALL хранить UI-модель в YAML-файлах в каталоге `ui_model/` репозитория: одна модель на окно (market, shop_buy, shop_sell) плюс общий словарь и навигационная модель `navigation.yaml`; каждая модель окна MUST содержать секции `window`, `elements`, `states`, `transitions`, `ocr_regions` и, для окон со списком, `table_geometry`. HUD-элементы навигации `market_pic`, `open_buy_trade`, `open_sale_trade` удалены из моделей окон (на голом экране живой игры их нет — KNOWN_GAPS п. 11); навигационные элементы («Меню», quick-slot) живут в `navigation.yaml`.

#### Scenario: Модель окна не содержит удалённых HUD-элементов
- **WHEN** загружаются `ui_model/market.yaml`, `ui_model/shop_buy.yaml`, `ui_model/shop_sell.yaml`
- **THEN** элементы `market_pic`, `open_buy_trade`, `open_sale_trade` отсутствуют, переход `recover_closed --click market_pic--> list__tab_home` отсутствует

#### Scenario: Навигационные элементы живут в navigation.yaml
- **WHEN** загружается `ui_model/navigation.yaml`
- **THEN** элементы гамбургера, якоря меню, иконки обзора в меню и иконок quick-slot доступны исполнителю рецептов

## ADDED Requirements

### Requirement: recover_closed пересобран без HUD-иконок
Сигнатуры глобального состояния `recover_closed` модели market SHALL подтверждать «все окна закрыты» без опорных HUD-иконок `market_pic`/`open_buy_trade` (≥ 2 независимых признака, набор — разметчик по сырью `scr_menu_closed` и др.).

#### Scenario: recover_closed подтверждается на голом экране живой игры
- **WHEN** детектор проверяет состояние на кадре `scr_menu_closed` (меню закрыто, окон нет)
- **THEN** `recover_closed` подтверждается сигнатурами модели

### Requirement: Формат ui_model описывает навигационную модель
`ui_model/FORMAT.md` SHALL описывать `navigation.yaml`: гранулярность (файл не является моделью окна), схему рецептов (шаги, обязательный verify, on_fail именованные шаги, `when`/`skip_if`, `expect: absent`, `alternate`, `opens`) и семантику элементов (HUD — `search_area`; элементы меню — от якоря `menu_anchor`).

#### Scenario: Формат документирует схему рецептов
- **WHEN** разметчик читает `ui_model/FORMAT.md`
- **THEN** секция навигационной модели задаёт схему `elements`/`signatures`/`recipes` без противоречий с валидатором
