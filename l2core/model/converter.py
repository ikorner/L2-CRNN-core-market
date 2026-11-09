"""Конвертер meta.json → UI-модель: импорт кропов в assets/, карточки
элементов и состояний, diff-отчёт. Файловая работа и YAML только —
никакой детекции и OCR.

Политика слияния «машина vs ручное» (единая, для всех секций):
- ручное побеждает: элемент/состояние с тем же id, что в существующей
  модели, не изменяется и не дополняется; при расхождении машинных
  данных с ручными diff-отчёт фиксирует конфликт (ручное побеждает);
- машина дополняет отсутствующее: новые элементы и состояния добавляются;
- секции transitions/ocr_regions/table_geometry машина не генерирует
  (мета их не знает) и переносит из ручной модели как есть;
- кропы в assets/ не перезаписываются: байтово совпадающий файл
  пропускается, расхождающийся — конфликт (файл-получатель остаётся,
  отчёт фиксирует расхождение).

Отображение «мета → модель»:
- окно: id из meta.window; title — из проектного словаря окон
  (AGENTS.md: «Обзор лавок», «Торговая лавка (Купить)»,
  «Торговая лавка (Продажа)»), мета тайтлов не хранит; size — размер
  кропа-примера всего окна;
- якорь окна: левый верхний угол самого большого кропа окна
  (кроп-пример открытого окна). rect'ы элементов пересчитываются от
  него. Кропы, не вписывающиеся в прямоугольник окна, импортируются
  как HUD-элементы (search_area = абсолютный rect кропа) и помечаются
  в отчёте как требующие ручной разметки (окно на их скриншоте могло
  быть в другом положении);
- элемент: id = часть meta element после префикса окна
  (obzor_lavok/buy/sell), санитизированная (префикс el_, если id не
  начинается с буквы); роль = первая переносимая роль по приоритету
  click_target > state_signature > ocr_region > icon_sample. Роли-сырьё
  meta (decor, ocr_ground_truth) в модель ядра не переносятся: кропы
  только с такими ролями пропускаются и попадают в отчёт;
- вариант: meta variant (normal/active); шаблон — импортированный кроп
  assets/<window>/<crop_id>.png (id кропа уникален и даёт прослеживаемость
  к мете);
- состояния: из скриншотов с непустым state, группировка по
  (state, state_params); id = имя состояния после точки + «__» +
  отсортированные params («к: значение»); без params — просто имя.
  Сигнатуры — template-признаки по кропам скриншота: активные варианты,
  если они есть, иначе все кропы. Мета фиксирует лишь факт наблюдения
  элемента в состоянии, а не семантику сигнатур — сокращение и
  усиление сигнатур, verify и on_fail — ручная разметка (2.3–2.5);
- transitions/ocr_regions/table_geometry: мета не знает — не
  генерируются.

Детерминизм: порядок окон фиксирован (market, shop_buy, shop_sell),
все списки и словари отсортированы, вывод не зависит от времени.

CLI: python -m l2core.model.converter [--meta ПУТЬ] [--scr-dir ПУТЬ]
[--model-dir ПУТЬ] [--assets ПУТЬ] — импорт + diff-отчёт на stdout.
"""

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Порядок окон фиксирован — совпадает с l2core.model.loader.WINDOW_IDS.
WINDOW_IDS = ("market", "shop_buy", "shop_sell")

# Подписи окон: мета их не хранит, источник — AGENTS.md проекта.
WINDOW_TITLES = {
    "market": "Обзор лавок",
    "shop_buy": "Торговая лавка (Купить)",
    "shop_sell": "Торговая лавка (Продажа)",
}

# Роли-сырьё meta: в модель ядра не переносятся (данные датасета).
RAW_ROLES = ("decor", "ocr_ground_truth")

# Переносимые роли по убыванию приоритета (элемент имеет одну роль).
ROLE_PRIORITY = ("click_target", "state_signature", "ocr_region", "icon_sample")

# Варианты, известные UI-модели (совпадают с common.yaml).
KNOWN_VARIANTS = (
    "normal",
    "hover",
    "pressed",
    "disabled",
    "active",
    "placeholder",
    "focused",
    "filled",
)


class MetaFormatError(Exception):
    """Некорректное сырьё meta.json: структура или ссылки."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"meta.json: {detail}")


@dataclass
class DiffReport:
    """Результат прогона конвертера: что сделано и что требует внимания."""

    assets_added: list = field(default_factory=list)
    assets_identical: list = field(default_factory=list)
    assets_conflicts: list = field(default_factory=list)
    elements_added: dict = field(default_factory=dict)  # window -> [id]
    elements_kept: dict = field(default_factory=dict)  # window -> [id]
    elements_conflicts: dict = field(default_factory=dict)  # window -> [id]
    states_added: dict = field(default_factory=dict)  # window -> [id]
    states_kept: dict = field(default_factory=dict)  # window -> [id]
    states_conflicts: dict = field(default_factory=dict)  # window -> [id]
    written_models: list = field(default_factory=list)
    issues: list = field(default_factory=list)  # неоднозначности сырья

    def render(self) -> str:
        """Отчёт на русском, для stdout и задачного отчёта."""
        lines = ["Diff-отчёт конвертера meta.json -> UI-модель", ""]
        lines.append(
            f"Кропы в assets/: добавлено {len(self.assets_added)}, "
            f"уже совпадают {len(self.assets_identical)}, "
            f"конфликтов (не перезаписано) {len(self.assets_conflicts)}"
        )
        for name in self.assets_added:
            lines.append(f"  + {name}")
        for name in self.assets_conflicts:
            lines.append(f"  ! {name} (файл отличается — оставлен, ручное побеждает)")
        for window in WINDOW_IDS:
            added = self.elements_added.get(window, [])
            kept = self.elements_kept.get(window, [])
            conflicts = self.elements_conflicts.get(window, [])
            lines.append(
                f"Элементы {window}: добавлено {len(added)}, "
                f"уже было {len(kept)}, конфликтов {len(conflicts)}"
            )
            for element_id in conflicts:
                lines.append(f"  ! {element_id} (ручное побеждает)")
            states_added = self.states_added.get(window, [])
            states_kept = self.states_kept.get(window, [])
            states_conflicts = self.states_conflicts.get(window, [])
            lines.append(
                f"Состояния {window}: добавлено {len(states_added)}, "
                f"уже было {len(states_kept)}, конфликтов {len(states_conflicts)}"
            )
            for state_id in states_conflicts:
                lines.append(f"  ! состояние {state_id} (ручное побеждает)")
        if self.written_models:
            lines.append("Записаны модели: " + ", ".join(self.written_models))
        if self.issues:
            lines.append("Примечания (неоднозначности сырья):")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        return "\n".join(lines)


# --- Разбор сырья ---


def load_meta(meta_path: Path) -> dict:
    """Прочитать meta.json; битый файл -> MetaFormatError."""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetaFormatError(f"не читается: {exc}") from exc
    for key in ("screenshots", "crops"):
        if not isinstance(meta.get(key), dict):
            raise MetaFormatError(f"отсутствует секция {key!r}")
    return meta


def _element_name(meta_element: str, window: str) -> str:
    """Часть meta element после префикса окна; санитизация id элемента."""
    prefix, _, name = meta_element.rpartition(".")
    if not prefix or not name:
        raise MetaFormatError(f"element без префикса окна: {meta_element!r}")
    if not name[0].isalpha():
        name = f"el_{name}"
    return name


def _transferable_role(roles: list) -> str:
    """Первая переносимая роль по приоритету; None, если все роли — сырьё."""
    for role in ROLE_PRIORITY:
        if role in roles:
            return role
    return None


def _window_anchor(cards: list) -> tuple:
    """Якорь окна: (x, y, w, h) самого большого кропа окна.

    cards отсортированы по crop_id, поэтому при равенстве площадей
    выбирается первый по алфавиту — детерминизм.
    """
    anchor = max(cards, key=lambda card: card["rect"][2] * card["rect"][3])["rect"]
    return tuple(anchor)


def _contains(outer: tuple, rect: list) -> bool:
    """Вписывается ли rect (x, y, w, h) в прямоугольник outer."""
    ox, oy, ow, oh = outer
    x, y, w, h = rect
    return ox <= x and oy <= y and x + w <= ox + ow and y + h <= oy + oh


# --- Машинная модель окна (словари в терминах YAML) ---


def build_window_model(window: str, meta: dict) -> tuple:
    """Собрать машинную модель окна: (словарь секций, карточки кропов).

    Возвращает только то, что знает мета: window/elements/states.
    """
    cards = []
    for crop_id, card in sorted(meta["crops"].items()):
        if card.get("window") != window:
            continue
        rect = card.get("rect")
        if not (isinstance(rect, list) and len(rect) == 4):
            raise MetaFormatError(f"кроп {crop_id}: битый rect {rect!r}")
        cards.append({"crop_id": crop_id, **card})

    if not cards:
        raise MetaFormatError(f"окно {window!r} отсутствует в мете")

    anchor = _window_anchor(cards)
    ax, ay, aw, ah = anchor

    elements = {}
    states = {}
    issues = []
    imported = []  # карточки кропов, вошедших в модель (для assets/)

    for card in cards:
        crop_id = card["crop_id"]
        name = _element_name(card["element"], window)
        variant = card.get("variant")
        if variant not in KNOWN_VARIANTS:
            issues.append(
                f"{window}/{name}: вариант {variant!r} неизвестен модели — "
                f"кроп {crop_id} пропущен"
            )
            continue
        role = _transferable_role(card.get("roles", []))
        if role is None:
            issues.append(
                f"{window}/{name}: только роли-сырьё meta "
                f"({card.get('roles')}) — кроп {crop_id} не перенесён в модель"
            )
            continue

        element = elements.setdefault(
            name,
            {
                "rect": [card["rect"][0] - ax, card["rect"][1] - ay]
                + card["rect"][2:],
                "role": role,
                "variants": {},
            },
        )
        if element["variants"]:
            # Элемент уже видели: rect берём от первого наблюдения.
            if element["rect"] != [
                card["rect"][0] - ax,
                card["rect"][1] - ay,
            ] + card["rect"][2:]:
                issues.append(
                    f"{window}/{name}: rect кропа {crop_id} отличается от "
                    f"первого наблюдения — оставлен первый"
                )
        template = f"{window}/{crop_id}.png"
        if variant not in element["variants"]:
            element["variants"][variant] = {"template": template}
        imported.append(card)

    # Классификация по геометрии: не вписавшиеся в окно -> HUD.
    window_rect = [ax, ay, aw, ah]
    for card in imported:
        name = _element_name(card["element"], window)
        element = elements[name]
        if card["rect"] == window_rect and element["rect"] == [0, 0, aw, ah]:
            continue  # сам кроп-пример окна — не HUD
        if not _contains(window_rect, card["rect"]):
            own_role = element["role"]  # роль элемента, а не последнего кропа
            variants = element["variants"]
            element.clear()
            element["role"] = own_role
            element["hud"] = True
            element["search_area"] = list(card["rect"])
            element["variants"] = variants
            issues.append(
                f"{window}/{name}: кроп {card['crop_id']} не вписывается в "
                f"прямоугольник окна — импортирован как HUD-элемент "
                f"(search_area = абсолютный rect); вероятно, окно на этом "
                f"скриншоте в другом положении, разметить вручную"
            )

    # Состояния: скриншоты с непустым state, группировка по params.
    by_state = {}
    for shot_id, shot in sorted(meta["screenshots"].items()):
        if shot.get("window") != window:
            continue
        state_name = shot.get("state")
        if not state_name:
            continue
        params = shot.get("state_params") or {}
        key = (state_name, tuple(sorted(params.items())))
        by_state.setdefault(key, []).append(shot_id)

    for (state_name, params_items), shot_ids in sorted(by_state.items()):
        local = state_name.rpartition(".")[2] or state_name
        state_id = local
        if params_items:
            state_id += "__" + "__".join(f"{k}_{v}" for k, v in params_items)
        crop_ids = []
        for shot_id in shot_ids:
            crop_ids.extend(meta["screenshots"][shot_id].get("crops", []))
        crop_ids = sorted(set(crop_ids) & {c["crop_id"] for c in imported})

        active = [c for c in crop_ids if meta["crops"][c]["variant"] == "active"]
        chosen = active if active else crop_ids
        signatures = [
            {
                "kind": "template",
                "element": _element_name(meta["crops"][c]["element"], window),
                "variant": meta["crops"][c]["variant"],
            }
            for c in chosen
        ]
        states[state_id] = {
            "params": dict(params_items),
            "signatures": signatures,
        }
        if len(signatures) < 2:
            issues.append(
                f"{window}/{state_id}: меньше двух template-сигнатур "
                f"(мета знает только факт наблюдения элементов) — "
                f"досигнатуры вручную (2.3)"
            )

    model = {
        "window": {
            "id": window,
            "title": WINDOW_TITLES[window],
            "size": [aw, ah],
        },
        "elements": {name: elements[name] for name in sorted(elements)},
        "states": {sid: states[sid] for sid in sorted(states)},
    }
    return model, imported, issues


# --- Импорт кропов в assets/ ---


def import_assets(imported: dict, scr_dir: Path, assets_dir: Path, report: DiffReport) -> None:
    """Скопировать кропы модели в assets/<window>/<crop_id>.png.

    Не перезаписывает существующие файлы: совпадение байт-в-байт —
    пропуск, расхождение — конфликт в отчёте.
    """
    for window in sorted(imported):
        for card in imported[window]:
            rel = f"{window}/{card['crop_id']}.png"
            src = scr_dir / card["file"]
            dest = assets_dir / rel
            data = src.read_bytes()
            if dest.exists():
                if dest.read_bytes() == data:
                    report.assets_identical.append(rel)
                else:
                    report.assets_conflicts.append(rel)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            report.assets_added.append(rel)
    report.assets_added.sort()
    report.assets_identical.sort()
    report.assets_conflicts.sort()


# --- Слияние с существующей моделью и запись ---


def merge_window_model(window: str, machine: dict, existing: dict, report: DiffReport) -> dict:
    """Слить машинную модель с существующей: ручное побеждает, машина
    дополняет. Секции transitions/ocr_regions/table_geometry машина не
    генерирует и берёт из существующей модели как есть."""
    if existing is None:
        report.elements_added[window] = sorted(machine["elements"])
        report.states_added[window] = sorted(machine["states"])
        return {
            "window": machine["window"],
            "elements": machine["elements"],
            "states": machine["states"],
            "transitions": [],
            "ocr_regions": {},
        }

    merged = dict(existing)
    if existing.get("window") != machine["window"]:
        report.issues.append(
            f"{window}: секция window ручной модели отличается от машинной "
            f"(size/title) — оставлена ручная"
        )
    existing_elements = existing.get("elements", {})
    added, kept, conflicts = [], [], []
    for element_id in sorted(machine["elements"]):
        if element_id in existing_elements:
            if existing_elements[element_id] == machine["elements"][element_id]:
                kept.append(element_id)
            else:
                conflicts.append(element_id)
        else:
            added.append(element_id)
    merged_elements = dict(existing_elements)
    merged_elements.update(
        {eid: machine["elements"][eid] for eid in sorted(added)}
    )
    merged["elements"] = merged_elements
    report.elements_added[window] = added
    report.elements_kept[window] = kept
    report.elements_conflicts[window] = conflicts

    existing_states = existing.get("states", {})
    added_states, kept_states, conflict_states = [], [], []
    for state_id in sorted(machine["states"]):
        if state_id in existing_states:
            if existing_states[state_id] == machine["states"][state_id]:
                kept_states.append(state_id)
            else:
                conflict_states.append(state_id)
        else:
            added_states.append(state_id)
    merged_states = dict(existing_states)
    merged_states.update({sid: machine["states"][sid] for sid in sorted(added_states)})
    merged["states"] = merged_states
    report.states_added[window] = added_states
    report.states_kept[window] = kept_states
    report.states_conflicts[window] = conflict_states
    return merged


class _FlowList(list):
    """Список, сериализуемый в YAML в потоковом виде [a, b, c]."""


class _FlowDict(dict):
    """Словарь, сериализуемый в YAML в потоковом виде {k: v}."""


def _flow(obj):
    """Обёртка координат/коротких спек в потоковый стиль YAML (FORMAT.md).

    Словарь со скалярными значениями — потоковый {k: v}; словарь с
    вложенными структурами — блочный, рекурсия применяется к значениям;
    список целых — потоковый [a, b, c], прочие списки — блочные.
    """
    if isinstance(obj, dict):
        if all(not isinstance(v, (dict, list)) for v in obj.values()):
            return _FlowDict(obj)
        return {key: _flow(value) for key, value in obj.items()}
    if isinstance(obj, list):
        if all(isinstance(v, int) and not isinstance(v, bool) for v in obj):
            return _FlowList(obj)
        return [_flow(v) for v in obj]
    return obj


class _Dumper(yaml.SafeDumper):
    """SafeDumper с представителями потоковых списков/словарей."""


_Dumper.add_representer(
    _FlowList,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)
_Dumper.add_representer(
    _FlowDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data, flow_style=True
    ),
)

_HEADER = (
    "# UI-модель окна {window}. Импорт конвертером (l2core.model.converter) "
    "из base_screenshot/scr/meta.json.\n"
    "# Политика слияния: ручные правки побеждают, машинное дополняет; "
    "diff-отчёт — `python -m l2core.model.converter`.\n"
    "# Сигнатуры состояний, verify/on_fail, OCR-регионы и table_geometry "
    "доразмечаются вручную (задачи 2.3-2.5).\n"
)


def _dump_model(window: str, merged: dict) -> str:
    ordered = {}
    for section in ("window", "elements", "states", "transitions", "ocr_regions", "table_geometry"):
        if section in merged:
            ordered[section] = merged[section]
    body = yaml.dump(
        _flow(ordered),
        Dumper=_Dumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )
    return _HEADER.format(window=window) + body


# --- Точка входа ---


def convert(meta_path, scr_dir, model_dir, assets_dir) -> DiffReport:
    """Прогнать конвертер: импорт кропов, сбор моделей, слияние, отчёт.

    meta_path/scr_dir/model_dir/assets_dir — пути (str или Path).
    Существующие ui_model/<window>.yaml читаются и дополняются, ручные
    правки не перезаписываются. Возвращает DiffReport.
    """
    meta_path = Path(meta_path)
    scr_dir = Path(scr_dir)
    model_dir = Path(model_dir)
    assets_dir = Path(assets_dir)
    meta = load_meta(meta_path)

    report = DiffReport()
    meta_windows = {card.get("window") for card in meta["crops"].values()}
    machine_models = {}
    imported = {}
    for window in WINDOW_IDS:
        if window not in meta_windows:
            continue
        machine, cards, issues = build_window_model(window, meta)
        machine_models[window] = machine
        imported[window] = cards
        report.issues.extend(issues)

    import_assets(imported, scr_dir, assets_dir, report)

    model_dir.mkdir(parents=True, exist_ok=True)
    for window in WINDOW_IDS:
        if window not in machine_models:
            continue
        model_file = model_dir / f"{window}.yaml"
        existing = None
        if model_file.exists():
            loaded = yaml.safe_load(model_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        merged = merge_window_model(window, machine_models[window], existing, report)
        if existing is not None and merged == existing:
            # Слияние ничего не добавило — файл не трогаем (в т.ч.
            # комментарии ручной разметки остаются на месте).
            report.issues.append(f"{window}: модель без изменений, файл не перезаписан")
            continue
        model_file.write_text(_dump_model(window, merged), encoding="utf-8")
        report.written_models.append(str(model_file))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Конвертер meta.json → UI-модель: импорт кропов в "
        "assets/, карточки элементов, diff-отчёт (ручные правки "
        "не перезаписываются)."
    )
    parser.add_argument("--meta", default="base_screenshot/scr/meta.json")
    parser.add_argument("--scr-dir", default="base_screenshot/scr")
    parser.add_argument("--model-dir", default="ui_model")
    parser.add_argument("--assets", default="assets")
    args = parser.parse_args(argv)

    report = convert(args.meta, args.scr_dir, args.model_dir, args.assets)
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
