"""
Assembly utilities for 3D topology optimization.

This module contains helper functions for assembling the force vector,
boundary conditions, and element DOF matrices.
"""

from typing import Optional, Set, Tuple

import numpy as np

from pytopo3d.utils.axis_convention import warn_force_xy_change_once


def build_force_vector(
    nelx: int,
    nely: int,
    nelz: int,
    ndof: int,
    force_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build the global force vector F.

    If force_field is None, applies default forces to the right face (x=nelx)
    in the negative z-direction.

    If force_field is provided, it distributes the forces specified for each
    element equally among its 8 corner nodes.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    ndof : int
        Total number of degrees of freedom (3 * number_of_nodes).
    force_field : Optional[np.ndarray], shape (nely, nelx, nelz, 3), optional
        A 4D array where `force_field[y, x, z, :]` is the [Fx, Fy, Fz] force
        vector associated with the element at grid position (y, x, z).
        Defaults to None (use default load case).

    Returns
    -------
    np.ndarray
        Global force vector F (shape: ndof) with applied nodal loads.
    """
    F = np.zeros(ndof)

    if force_field is None:
        # Default load: a unit downward (-z) force on every node of the far x face at
        # z=0, i.e. the line x=nelx, z=0 spanning all y (il=nelx, jl=all y, kl=0).
        il, jl, kl = np.meshgrid(
            [nelx],
            np.arange(nely + 1),
            [0],
            indexing="ij",
        )
        # Calculate 0-based global node indices using Fortran order
        # nid = iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)
        loadnid_0based = (
            jl.flatten()
            + il.flatten() * (nely + 1)
            + kl.flatten() * (nelx + 1) * (nely + 1)
        )
        # Calculate 0-based DOF indices for z-direction (3*nid + 2)
        loaddof_0based = 3 * loadnid_0based + 2
        # Apply unit force in negative z-direction
        F[loaddof_0based] = -1.0

    else:
        # Validate force_field shape
        expected_shape = (nely, nelx, nelz, 3)
        if force_field.shape != expected_shape:
            raise ValueError(
                f"force_field has shape {force_field.shape}, "
                f"but expected {expected_shape}"
            )

        # The x/y force-direction transpose was fixed in 0.3.0; warn once if this load
        # carries Fx or Fy components (whose behaviour changed). Fz-only loads are unaffected.
        if np.any(force_field[..., 0]) or np.any(force_field[..., 1]):
            warn_force_xy_change_once()

        # Iterate through each element and distribute its force to its 8 nodes
        for elz in range(nelz):
            for elx in range(nelx):
                for ely in range(nely):
                    element_force = force_field[ely, elx, elz, :]

                    # Skip if force is zero for this element
                    if not np.any(element_force):
                        continue

                    # Distribute force to the 8 corner nodes
                    force_per_node = element_force / 8.0

                    # Loop over the 8 local corners (relative coordinates dx, dy, dz in {0, 1})
                    for dz in [0, 1]:
                        for dx in [0, 1]:
                            for dy in [0, 1]:
                                # Global coordinates of the node
                                ix = elx + dx
                                iy = ely + dy
                                iz = elz + dz

                                # Calculate 0-based global node index (Fortran order)
                                nid = (
                                    iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)
                                )

                                # Calculate 0-based global DOF indices
                                dof_x = 3 * nid
                                dof_y = 3 * nid + 1
                                dof_z = 3 * nid + 2

                                # Add force contribution to the global force vector F
                                # Ensure DOFs are within bounds (although they should be
                                # if nid is calculated correctly)
                                if dof_x < ndof:
                                    F[dof_x] += force_per_node[0]
                                if dof_y < ndof:
                                    F[dof_y] += force_per_node[1]
                                if dof_z < ndof:
                                    F[dof_z] += force_per_node[2]

    return F


def build_supports(
    nelx: int,
    nely: int,
    nelz: int,
    ndof: int,
    support_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build support constraints (fixed DOFs).

    If support_mask is None, applies default supports (fixed nodes)
    on the left face (x=0).

    If support_mask is provided, it identifies elements marked as supported.
    All 8 corner nodes of these marked elements will have their DOFs fixed.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    ndof : int
        Total number of degrees of freedom.
    support_mask : Optional[np.ndarray], shape (nely, nelx, nelz), optional
        Boolean mask indicating which elements are supported.
        If None, uses default left-face support placement.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - freedofs0: 0-based array of free (unconstrained) DOFs.
        - fixeddof0: 0-based array of fixed (constrained) DOFs.
    """
    fixeddof0 = np.array([], dtype=int)  # Initialize as empty array
    fixednid_0based: np.ndarray = np.array([], dtype=int)  # Ensure defined type

    if support_mask is None:
        # Default implementation - fixed DOFs on nodes of the left face (x=0)
        iif, jf, kf = np.meshgrid(
            [0], np.arange(nely + 1), np.arange(nelz + 1), indexing="ij"
        )
        # Calculate 0-based global node indices using Fortran order
        # nid = iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)
        fixednid_0based = (
            jf.flatten()
            + iif.flatten() * (nely + 1)
            + kf.flatten() * (nelx + 1) * (nely + 1)
        )

    else:
        # Apply supports based on the support_mask provided for elements
        # Validate support_mask shape
        expected_shape = (nely, nelx, nelz)
        if support_mask.shape != expected_shape:
            raise ValueError(
                f"support_mask has shape {support_mask.shape}, "
                f"but expected {expected_shape}"
            )

        # Find elements marked for support
        y_indices, x_indices, z_indices = np.where(support_mask)

        if len(y_indices) == 0:
            # Fall back to default if no supports are specified in the mask
            print(
                "Warning: Support mask is provided but is empty. "
                "Falling back to default left-face support."
            )
            iif, jf, kf = np.meshgrid(
                [0], np.arange(nely + 1), np.arange(nelz + 1), indexing="ij"
            )
            fixednid_0based = (
                jf.flatten()
                + iif.flatten() * (nely + 1)
                + kf.flatten() * (nelx + 1) * (nely + 1)
            )
        else:
            # Strategy: For each element marked in support_mask, fix all DOFs
            # of its 8 corner nodes. Collect all unique fixed node indices.
            all_fixed_node_indices: Set[int] = set()

            for ely, elx, elz in zip(y_indices, x_indices, z_indices):
                # Loop over the 8 local corners (relative coordinates dx, dy, dz in {0, 1})
                for dz in [0, 1]:
                    for dx in [0, 1]:
                        for dy in [0, 1]:
                            # Global coordinates of the node
                            ix = elx + dx
                            iy = ely + dy
                            iz = elz + dz

                            # Ensure node coordinates are within the grid boundaries
                            if 0 <= iy <= nely and 0 <= ix <= nelx and 0 <= iz <= nelz:
                                # Calculate 0-based global node index (Fortran order)
                                nid = (
                                    iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)
                                )
                                all_fixed_node_indices.add(nid)

            fixednid_0based = np.array(list(all_fixed_node_indices), dtype=int)

    # If fixednid_0based is defined and not empty (either from default or mask)
    if fixednid_0based.size > 0:
        # Fix all degrees of freedom (X, Y, Z) at these nodes
        # DOFs are 0-based: 3*node_idx, 3*node_idx+1, 3*node_idx+2
        fixeddof0 = np.concatenate(
            [
                3 * fixednid_0based,
                3 * fixednid_0based + 1,
                3 * fixednid_0based + 2,
            ]
        )
        # Ensure uniqueness and sort
        fixeddof0 = np.unique(fixeddof0)
        # Ensure DOFs are within bounds
        fixeddof0 = fixeddof0[(fixeddof0 >= 0) & (fixeddof0 < ndof)]

    # Determine free DOFs
    all_dofs0 = np.arange(ndof)  # 0-based DOFs
    # Use np.isin for potentially faster set difference calculation
    is_fixed = np.isin(all_dofs0, fixeddof0, assume_unique=True)
    freedofs0 = all_dofs0[~is_fixed]

    # Return 0-based indices
    return freedofs0, fixeddof0


def build_edof(
    nelx: int, nely: int, nelz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build element DOF mapping and global assembly indices (iK, jK).

    Uses Fortran-style (column-major) node numbering convention.
    edofMat maps element number to its 24 global DOFs (8 nodes * 3 DOFs/node).
    iK, jK provide the row and column indices for assembling the global
    stiffness matrix K in sparse COO format.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - edofMat: Array (nele, 24) mapping element index to 1-based global DOFs.
        - iK: Array (nele * 576,) row indices for COO sparse matrix assembly (1-based).
        - jK: Array (nele * 576,) column indices for COO sparse matrix assembly (1-based).
    """
    # Generate node numbers for the grid in Fortran order
    nodenrs = np.arange(1, (nelx + 1) * (nely + 1) * (nelz + 1) + 1).reshape(
        (nely + 1, nelx + 1, nelz + 1), order="F"
    )
    # Get the node number for the 'bottom-front-left' corner of each element
    # (iy=0, ix=0, iz=0 local coords relative to element)
    edofVec_node_ids = nodenrs[:-1, :-1, :-1].ravel(order="F")

    # Get the 1-based DOF index for the first DOF (x-direction) of the first node
    edofVec = (
        3 * edofVec_node_ids - 2
    )  # 3*nid - 2 -> x-dof ; 3*nid - 1 -> y-dof ; 3*nid -> z-dof

    # Define the offsets to get the 24 DOFs of an H8 element relative to the first DOF.
    # Node order (local): 0,1,2,3 (bottom face z=0), 4,5,6,7 (top face z=1) -- matches the
    # lk_H8 element's local node ordering, so the element's first local axis aligns with the
    # grid's x (nelx) axis. Local node coords (x, y, z):
    # 0:(0,0,0), 1:(1,0,0), 2:(1,1,0), 3:(0,1,0)
    # 4:(0,0,1), 5:(1,0,1), 6:(1,1,1), 7:(0,1,1)
    # DOFs follow node order: [node0_x, node0_y, node0_z, node1_x, ..., node7_z].
    # The +x neighbour is one node over in x (offset 3*(nely+1)); the +y neighbour is the
    # next node id (offset 3*1). NOTE: before the 0.3.0 fix nodes 1<->3 (and 5<->7) used the
    # opposite offsets, which transposed the element's x and y axes -- a regression from
    # commit 75174dd that made an "Fx" load act along y. See tests/test_force_direction.py.
    dof_offsets = np.array(
        [
            0,
            1,
            2,  # Node 0 (ix=0, iy=0, iz=0)
            3 * (nely + 1) + 0,
            3 * (nely + 1) + 1,
            3 * (nely + 1) + 2,  # Node 1 (ix=1, iy=0, iz=0) Offset=3*dx*(nely+1)
            3 * (nely + 1 + 1) + 0,
            3 * (nely + 1 + 1) + 1,
            3 * (nely + 1 + 1)
            + 2,  # Node 2 (ix=1, iy=1, iz=0) Offset=3*(dx*(nely+1)+dy) = 3*(nely+1+1)
            3 * 1 + 0,
            3 * 1 + 1,
            3 * 1 + 2,  # Node 3 (ix=0, iy=1, iz=0) Offset=3*dy=3
            3 * (nelx + 1) * (nely + 1) + 0,
            3 * (nelx + 1) * (nely + 1) + 1,
            3 * (nelx + 1) * (nely + 1)
            + 2,  # Node 4 (ix=0, iy=0, iz=1) Offset=3*dz*(nelx+1)*(nely+1)
            3 * (nelx + 1) * (nely + 1) + 3 * (nely + 1) + 0,
            3 * (nelx + 1) * (nely + 1) + 3 * (nely + 1) + 1,
            3 * (nelx + 1) * (nely + 1) + 3 * (nely + 1) + 2,  # Node 5 (ix=1, iy=0, iz=1)
            3 * (nelx + 1) * (nely + 1) + 3 * (nely + 1 + 1) + 0,
            3 * (nelx + 1) * (nely + 1) + 3 * (nely + 1 + 1) + 1,
            3 * (nelx + 1) * (nely + 1)
            + 3 * (nely + 1 + 1)
            + 2,  # Node 6 (ix=1, iy=1, iz=1)
            3 * (nelx + 1) * (nely + 1) + 3 * 1 + 0,
            3 * (nelx + 1) * (nely + 1) + 3 * 1 + 1,
            3 * (nelx + 1) * (nely + 1)
            + 3 * 1
            + 2,  # Node 7 (ix=0, iy=1, iz=1)
        ],
        dtype=int,
    )

    # Build edofMat (nele, 24) containing 1-based global DOF indices for each element
    edofMat = edofVec[:, np.newaxis] + dof_offsets[np.newaxis, :]

    # Prepare iK, jK indices (1-based) for COO sparse matrix format
    iK = np.kron(edofMat, np.ones((1, 24), dtype=int)).ravel()
    jK = np.kron(edofMat, np.ones((24, 1), dtype=int)).ravel()

    return edofMat, iK, jK
