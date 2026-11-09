"""Публичный синхронный API ядра для модулей стратегий (задача 7.1).

Фасад L2CoreAPI (блокирующие вызовы только из потока-владельца) и сборка
провайдеров ApiWiring / build_wiring / фабрика create_api — контракты в
докстрингах l2core/api/core.py и l2core/api/wiring.py.
"""
from l2core.api.core import (
    ApiError,
    CoreState,
    HealthResult,
    L2CoreAPI,
    NavigationResult,
    ShopStateResult,
)
from l2core.api.engine import (
    EngineError,
    EngineShutdown,
    EventDispatcher,
    L2CoreBot,
    ObserverSnapshot,
    StateObserver,
)
from l2core.api.events import (
    EVENT_ESCALATION,
    EVENT_KILLED,
    EVENT_MODAL,
    EVENT_RECOVERED,
    EVENT_SOLD,
    EVENT_WINDOW_CLOSED,
    Event,
    EventQueue,
)
from l2core.api.hid_owner import HidOwnershipError, OwnerGuardedHid
from l2core.api.wiring import ApiWiring, build_wiring, create_api

__all__ = [
    "ApiError",
    "ApiWiring",
    "CoreState",
    "EngineError",
    "EngineShutdown",
    "EVENT_ESCALATION",
    "EVENT_KILLED",
    "EVENT_MODAL",
    "EVENT_RECOVERED",
    "EVENT_SOLD",
    "EVENT_WINDOW_CLOSED",
    "Event",
    "EventDispatcher",
    "EventQueue",
    "HealthResult",
    "HidOwnershipError",
    "L2CoreAPI",
    "L2CoreBot",
    "NavigationResult",
    "ObserverSnapshot",
    "OwnerGuardedHid",
    "ShopStateResult",
    "StateObserver",
    "build_wiring",
    "create_api",
]
