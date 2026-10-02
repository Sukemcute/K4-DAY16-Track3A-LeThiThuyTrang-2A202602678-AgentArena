"""Shared resource checks, so retry cannot spend the submit reservation."""

from __future__ import annotations

import math


def finite_limit(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, float(value)) if math.isfinite(value) else None


def tool_budget_spent(ctx, reserve=1):
    limit = finite_limit(ctx.max_tool_calls)
    reserve = max(reserve, ctx.state.get("budget.reserve", 0))
    return limit is not None and ctx.tools.calls + 1 > limit - reserve


def finalizing(ctx):
    if ctx.state.get("budget.finalizing", False):
        return True
    limit = finite_limit(ctx.budget.get("max_tokens"))
    if limit is not None and ctx.state.get("budget.tokens", 0) >= limit - ctx.state.get("budget.reserve_tokens", 0):
        return True
    clock = ctx.state.get("budget.clock")
    deadline = ctx.state.get("budget.deadline")
    return callable(clock) and deadline is not None and clock() >= deadline
