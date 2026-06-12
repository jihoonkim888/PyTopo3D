"""
Optimality Criteria (OC) update scheme for 3D topology optimization.

This module contains the function for updating design variables
using the optimality criteria method.
"""

from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp

# Import filtering function
from pytopo3d.utils.filter import apply_filter, HAS_CUPY

# Check if CuPy is available for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def optimality_criteria_update(
    x: Union[np.ndarray, "cp.ndarray"],
    dc: Union[np.ndarray, "cp.ndarray"],
    dv: Union[np.ndarray, "cp.ndarray"],
    volfrac: float,
    H: Union[sp.csr_matrix, "cusp.csr_matrix"],
    Hs: Union[np.ndarray, "cp.ndarray"],
    nele: int,
    obstacle_mask: Union[np.ndarray, "cp.ndarray"],
    design_nele: int,
    use_gpu: bool = False
) -> Tuple[Union[np.ndarray, "cp.ndarray"], float]:
    """
    Performs the optimality criteria (OC) update with bisection on the Lagrange
    multiplier. The volume constraint is enforced only on the design domain
    (excluding obstacle elements).

    Parameters
    ----------
    x : ndarray or cupy.ndarray
        Current design variables.
    dc : ndarray or cupy.ndarray
        Sensitivity of compliance.
    dv : ndarray or cupy.ndarray
        Sensitivity of volume.
    volfrac : float
        Target volume fraction.
    H : scipy.sparse.csr_matrix or cupyx.scipy.sparse.csr_matrix
        Filter matrix.
    Hs : ndarray or cupy.ndarray
        Filter normalization factors.
    nele : int
        Total number of elements.
    obstacle_mask : ndarray or cupy.ndarray of bool
        Mask indicating obstacle elements.
    design_nele : int
        Number of elements in design domain (not obstacles).
    use_gpu : bool, optional
        Whether to use GPU acceleration if available. Default is False.

    Returns
    -------
    tuple
        (updated design variables, maximum change)
    """
    # Check if inputs are already on GPU
    inputs_on_gpu = HAS_CUPY and (
        any(
            isinstance(arr, cp.ndarray)
            for arr in [x, dc, dv, obstacle_mask, Hs]
        )
        or isinstance(H, cusp.csr_matrix)
    )
    
    # Determine whether to use GPU based on inputs and use_gpu flag
    use_gpu_for_calc = (use_gpu and HAS_CUPY) or inputs_on_gpu
    
    l1, l2 = 1e-9, 1e9
    move = 0.2
    
    if use_gpu_for_calc:
        # Transfer arrays to GPU if not already there
        if isinstance(x, cp.ndarray):
            x_gpu = x
        else:
            x_gpu = cp.asarray(x)
            
        if isinstance(dc, cp.ndarray):
            dc_gpu = dc
        else:
            dc_gpu = cp.asarray(dc)
            
        if isinstance(dv, cp.ndarray):
            dv_gpu = dv
        else:
            dv_gpu = cp.asarray(dv)
            
        if isinstance(obstacle_mask, cp.ndarray):
            obstacle_mask_gpu = obstacle_mask
        else:
            obstacle_mask_gpu = cp.asarray(obstacle_mask)
        
        # Create design cells mask on GPU
        design_cells_gpu = ~obstacle_mask_gpu
        
        # Convert filter to GPU if needed
        if isinstance(H, cusp.csr_matrix):
            H_gpu = H
        else:
            H_gpu = cusp.csr_matrix((cp.asarray(H.data), 
                                    cp.asarray(H.indices), 
                                    cp.asarray(H.indptr)),
                                    shape=H.shape)
            
        # Convert Hs to GPU if needed
        if isinstance(Hs, cp.ndarray):
            Hs_gpu = Hs
        else:
            Hs_gpu = cp.asarray(Hs)
            
        # Create result on GPU
        xnew_gpu = x_gpu.copy()

        # OC update loop (on GPU)
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            
            # GPU-accelerated OC update
            update_term = -dc_gpu / (dv_gpu * lmid)
            
            # Handle potential issues
            update_term[dv_gpu < 1e-9] = 0.0
            update_term[update_term < 0] = 0.0
            
            # Calculate updated x
            x_candidate_gpu = x_gpu * cp.sqrt(update_term)
            
            # Apply move limits and obstacles
            x_candidate_gpu = cp.clip(
                x_candidate_gpu,
                cp.maximum(0.0, x_gpu - move),
                cp.minimum(1.0, x_gpu + move),
            )
            x_candidate_gpu[obstacle_mask_gpu] = 0.0
            
            # Apply filter directly using our improved apply_filter function
            # This avoids unnecessary CPU-GPU transfers
            xPhysCandidate_gpu = apply_filter(
                H_gpu, 
                x_candidate_gpu, 
                Hs_gpu, 
                x_gpu.shape, 
                use_gpu=True
            )
            xPhysCandidate_gpu[obstacle_mask_gpu] = 0.0
            
            # Check volume constraint (remaining on GPU)
            vol_constraint = cp.sum(xPhysCandidate_gpu[design_cells_gpu])
            if vol_constraint > volfrac * design_nele:
                l1 = lmid
            else:
                l2 = lmid
                
            xnew_gpu = x_candidate_gpu
        
        # Calculate change on GPU
        change = float(cp.max(cp.abs(xnew_gpu - x_gpu)))
        
        # Return GPU array if input was GPU or use_gpu=True
        if isinstance(x, cp.ndarray) or use_gpu:
            return xnew_gpu, change
        else:
            # Otherwise transfer back to CPU
            return cp.asnumpy(xnew_gpu), change
    else:
        # CPU version (unchanged)
        xnew = x.copy()
        design_cells = ~obstacle_mask

        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            
            # Calculate update term
            update_term = -dc / (dv * lmid)
            update_term[dv < 1e-9] = 0.0
            update_term[update_term < 0] = 0.0
            
            # Calculate new x
            x_candidate = x * np.sqrt(update_term)
            x_candidate = np.clip(
                x_candidate,
                np.maximum(0.0, x - move),
                np.minimum(1.0, x + move),
            )
            x_candidate[obstacle_mask] = 0.0
            
            # Apply filter
            xPhysCandidate = apply_filter(H, x_candidate, Hs, x.shape, use_gpu=False)
            xPhysCandidate[obstacle_mask] = 0.0
            
            # Check volume constraint
            if xPhysCandidate[design_cells].sum() > volfrac * design_nele:
                l1 = lmid
            else:
                l2 = lmid
            xnew = x_candidate
        
        change = np.abs(xnew - x).max()
        return xnew, change
