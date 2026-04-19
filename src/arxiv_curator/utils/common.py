"""Common utilities for Databricks notebooks and scripts."""

from __future__ import annotations


def get_widget(name: str, default: str | None = None) -> str | None:
    """Get a Databricks widget value with a fallback default.

    Args:
        name: Widget name.
        default: Default value if widget is not defined.

    Returns:
        Widget value or default.
    """
    try:
        from databricks.sdk.runtime import dbutils

        return dbutils.widgets.get(name)
    except Exception:
        return default
