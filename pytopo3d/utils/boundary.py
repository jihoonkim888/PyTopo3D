"""
Boundary condition utilities for 3D topology optimization.

This module contains functions for handling boundary conditions such as loads and constraints.
"""

from typing import Tuple

import numpy as np


def create_boundary_arrays(
    nelx: int, nely: int, nelz: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create density arrays for loads and constraints visualization.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (loads_array, constraints_array), each with shape (nely, nelx, nelz)
        where positions have value 1.0 and the rest have value 0.0
    """
    # Create empty arrays for loads and constraints
    loads_array = np.zeros((nely, nelx, nelz))
    constraints_array = np.zeros((nely, nelx, nelz))

    # DEPRECATED - Hardcoded default BCs, use create_bc_visualization_arrays instead
    # Set loads at the right face (x=nelx-1) at the bottom (y=nely-1) on all Z levels
    # loads_array[nely-1, nelx-1, :] = 1.0

    # Set constraints at the left face (x=0) on all Y and Z levels
    # constraints_array[:, 0, :] = 1.0
    print(
        "Warning: create_boundary_arrays is deprecated. Use create_bc_visualization_arrays."
    )
    return loads_array, constraints_array


def create_bc_visualization_arrays(
    nelx: int, nely: int, nelz: int, ndof: int, F: np.ndarray, fixeddof0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates visualization arrays for loads and constraints based on actual FEA boundary conditions.

    Maps nodal forces and fixed DOFs to the element grid for visualization.
    Marks all elements adjacent to a node with applied BCs.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    ndof : int
        Total number of degrees of freedom.
    F : np.ndarray
        Global force vector (shape: ndof).
    fixeddof0 : np.ndarray
        Array of 0-based fixed degree of freedom indices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (loads_array, constraints_array), each with shape (nely, nelx, nelz).
    """
    loads_array = np.zeros((nely, nelx, nelz), dtype=float)
    constraints_array = np.zeros((nely, nelx, nelz), dtype=float)

    # --- Map Fixed DOFs to Element Constraints ---
    fixed_nid0 = np.unique(fixeddof0 // 3)
    nelyp1 = nely + 1
    nelxp1_nelyp1 = (nelx + 1) * (nely + 1)

    for nid in fixed_nid0:
        # Inverse calculation for Fortran node index -> (ix, iy, iz)
        iz = nid // nelxp1_nelyp1
        rem = nid % nelxp1_nelyp1
        ix = rem // nelyp1
        iy = rem % nelyp1

        # Mark adjacent elements (within grid bounds)
        for elz in range(max(0, iz - 1), min(nelz, iz + 1)):
            for elx in range(max(0, ix - 1), min(nelx, ix + 1)):
                for ely in range(max(0, iy - 1), min(nely, iy + 1)):
                    if 0 <= elx < nelx and 0 <= ely < nely and 0 <= elz < nelz:
                        constraints_array[ely, elx, elz] = 1.0

    # --- Map Forces to Element Loads ---
    loaded_dof0 = np.where(F != 0)[0]
    loaded_nid0 = np.unique(loaded_dof0 // 3)

    for nid in loaded_nid0:
        # Inverse calculation for Fortran node index -> (ix, iy, iz)
        iz = nid // nelxp1_nelyp1
        rem = nid % nelxp1_nelyp1
        ix = rem // nelyp1
        iy = rem % nelyp1

        # Mark adjacent elements (within grid bounds)
        for elz in range(max(0, iz - 1), min(nelz, iz + 1)):
            for elx in range(max(0, ix - 1), min(nelx, ix + 1)):
                for ely in range(max(0, iy - 1), min(nely, iy + 1)):
                    if 0 <= elx < nelx and 0 <= ely < nely and 0 <= elz < nelz:
                        loads_array[ely, elx, elz] = 1.0

    return loads_array, constraints_array
