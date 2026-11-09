"""Sanity-проверка: пакет l2core и его подпакеты импортируются."""


def test_import_l2core():
    import l2core  # noqa: F401


def test_import_subpackages():
    import l2core.model  # noqa: F401
    import l2core.detector  # noqa: F401
    import l2core.primitives  # noqa: F401
    import l2core.scenarios  # noqa: F401
    import l2core.adapters  # noqa: F401
    import l2core.api  # noqa: F401
