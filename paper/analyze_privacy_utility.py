"""Compatibility shim. The real module now lives at
``memarena/figures/analyze_privacy_utility.py`` and is the one used by the
fig:d6 generator. This file re-exports its public API so any stale import
path continues to work.
"""
from __future__ import annotations

from memarena.figures.analyze_privacy_utility import *  # noqa: F401,F403
from memarena.figures.analyze_privacy_utility import (  # noqa: F401
    BACKENDS,
    BACKEND_LABEL,
    BACKEND_COLOR,
    READERS,
    CAT_ORDER,
    EVAL_INSTANCES,
    PER_ITEM_DIRS,
    collect_distribution,
    collect_distribution_per_trial,
    load_eval_instances,
)
