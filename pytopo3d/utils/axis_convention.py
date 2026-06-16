"""One-time runtime notices for axis-convention changes.

Transitional notices, emitted once per session so users upgrading notice that results
differ: (1) the 0.2.0 STL x/y axis change (remove in 0.3.0), and (2) the 0.3.0 `force_field`
Fx/Fy transpose fix (remove in 0.4.0).
"""

import warnings

_WARNED = False

_MESSAGE = (
    "PyTopo3D 0.2.0 changed the STL axis convention: an STL's x-axis now maps to the "
    "domain's x-axis (nelx), so STL import/export results differ from 0.1.x. Non-STL "
    "(pure-array) usage is unchanged. Pin pytopo3d==0.1.2 to reproduce pre-0.2.0 "
    "results. See CHANGELOG.md. (Shown once; removed in 0.3.0.)"
)


def warn_stl_axis_change_once() -> None:
    """Emit the STL axis-convention change notice at most once per session."""
    global _WARNED
    if not _WARNED:
        _WARNED = True
        # stacklevel=3: warn -> warn_stl_axis_change_once -> public STL fn -> user code,
        # so 3 points the warning at the caller of stl_to_design_space / voxel_to_stl.
        warnings.warn(_MESSAGE, UserWarning, stacklevel=3)


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
