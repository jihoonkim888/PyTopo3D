"""One-time runtime notice for the 0.2.0 STL axis-convention change.

Transitional: emitted once per session the first time an STL is imported or exported,
so users upgrading from 0.1.x notice that STL results now differ. Remove in 0.3.0.
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
