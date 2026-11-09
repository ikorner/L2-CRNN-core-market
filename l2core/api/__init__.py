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
from l2core.api.wiring import ApiWiring, build_wiring, create_api

__all__ = [
    "ApiError",
    "ApiWiring",
    "CoreState",
    "HealthResult",
    "L2CoreAPI",
    "NavigationResult",
    "ShopStateResult",
    "build_wiring",
    "create_api",
]
