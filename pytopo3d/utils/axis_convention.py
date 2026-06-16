"""One-time runtime notice for the 0.3.0 force_field axis-convention fix.

Transitional notice, emitted once per session (for Fx/Fy loads only) so users upgrading
notice that results differ. Remove when cutting 0.4.0. (The earlier 0.2.0 STL x/y notice
was retired here after its transition period.)
"""

import warnings

_FORCE_WARNED = False

_FORCE_MESSAGE = (
    "PyTopo3D 0.3.0 fixed an x/y transpose in force_field: components Fx (index 0) and "
    "Fy (index 1) now act along the correct axes (they were swapped before). force_field "
    "loads with distinct Fx and Fy change; -z-only and x<->y-symmetric loads are unchanged. "
    "Pin the previous release to reproduce the old behavior. See CHANGELOG.md. "
    "(Shown once; removed in 0.4.0.)"
)


def warn_force_xy_change_once() -> None:
    """Emit the force_field x/y fix notice at most once per session (only for Fx/Fy loads)."""
    global _FORCE_WARNED
    if not _FORCE_WARNED:
        _FORCE_WARNED = True
        warnings.warn(_FORCE_MESSAGE, UserWarning, stacklevel=3)
