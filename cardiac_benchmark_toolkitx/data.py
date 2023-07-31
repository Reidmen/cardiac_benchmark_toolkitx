from dataclasses import dataclass

from dolfinx.fem import Function


@dataclass(frozen=True)
class MARKERS:
    BASE = 3
    ENDOCARDIUM = 1
    EPICARDIUM = 2


@dataclass(frozen=True)
class DEFAULTS:
    QUOTA_BASE = 1e-2
    R_SHORT_ENDO = 2.5e-2
    R_SHORT_EPI = 3.5e-2
    R_LONG_ENDO = 9.0e-2
    R_LONG_EPI = 9.7e-2
    FIBER_ALPHA_ENDO = -60.0
    FIBER_ALPHA_EPI = +60.0


@dataclass(frozen=True)
class FiberDirections:
    fiber: Function
    sheet: Function
    sheet_normal: Function
