import logging
import time
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from skimage import measure

from pytopo3d.utils.axis_convention import warn_stl_axis_change_once


def voxel_to_stl(
    input_file: Union[str, np.ndarray],
    output_file: Optional[str] = None,
    level: float = 0.5,
    padding: int = 1,
    fix_mesh: bool = True,
    smooth_mesh: bool = True,
    smooth_iterations: int = 5,
    upscale_factor: Optional[int] = None,
) -> Union[str, trimesh.Trimesh]:
    """
    Convert a voxel representation (.npy file or np.ndarray) to an STL mesh file or trimesh object.

    Parameters:
    -----------
    input_file : Union[str, np.ndarray]
        Path to the input .npy file containing voxel data or a NumPy array.
    output_file : Optional[str]
        Path where the output STL file will be saved. If None, the trimesh object is returned.
        Defaults to None.
    level : float, optional
        Contour level for the marching cubes algorithm (default: 0.5)
    padding : int, optional
        Number of zero-voxel layers to pad around the volume (default: 1)
    fix_mesh : bool, optional
        Whether to apply mesh repair operations (default: True)
    smooth_mesh : bool, optional
        Whether to apply Laplacian smoothing to the final mesh (default: True)
    smooth_iterations : int, optional
        Number of iterations for mesh smoothing (default: 5)
    upscale_factor : Optional[int], optional
        Factor to upscale voxel resolution before meshing (default: None)

    Returns:
    --------
    Union[str, trimesh.Trimesh]
        The path to the saved STL file if output_file is provided, otherwise the generated trimesh.Trimesh object.
    """
    warn_stl_axis_change_once()

    # 1. Load the voxel data from the .npy file or use the provided array
    if isinstance(input_file, str):
        voxel_data: np.ndarray = np.load(input_file)
    elif isinstance(input_file, np.ndarray):
        voxel_data = input_file
    else:
        raise TypeError("input_file must be a string path or a NumPy array.")

    # PyTopo3D stores densities as (nely, nelx, nelz). Swap the first two axes so the
    # exported mesh's x corresponds to the domain's x (nelx), mirroring the import
    # convention in import_design_space.stl_to_design_space. Without this an STL that is
    # imported and then exported comes back with its x and y transposed.
    voxel_data = voxel_data.swapaxes(0, 1)

    # 2. Pad the voxel data with zeros to ensure a closed mesh
    if padding > 0:
        padded_data: np.ndarray = np.pad(
            voxel_data, padding, mode="constant", constant_values=0
        )
    else:
        padded_data = voxel_data

    # 3. Upscale the voxel data if requested
    if upscale_factor and upscale_factor > 1:
        # Get the original shape
        original_shape: np.ndarray = np.array(padded_data.shape)
        # Calculate the new shape
        new_shape: np.ndarray = original_shape * upscale_factor
        # Create coordinates for the original data
        orig_coords: List[np.ndarray] = [np.arange(s) for s in original_shape]
        # Create coordinates for the upscaled data
        new_coords: List[np.ndarray] = [
            np.linspace(0, s - 1, ns) for s, ns in zip(original_shape, new_shape)
        ]
        # Use scipy's map_coordinates for smooth interpolation
        grid: List[np.ndarray] = np.meshgrid(*new_coords, indexing="ij")
        upscaled_data: np.ndarray = ndimage.map_coordinates(
            padded_data, grid, order=3, mode="nearest"
        )
        padded_data = upscaled_data

    # 4. Generate a triangulated mesh using marching cubes algorithm
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    vertices, faces, normals, _ = measure.marching_cubes(padded_data, level=level)

    # 5. Create a mesh object
    mesh: trimesh.Trimesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, normals=normals
    )

    # 6. Apply Laplacian smoothing to the mesh if requested
    if smooth_mesh and smooth_iterations > 0:
        # Check if the mesh has any vertices to smooth
        if len(mesh.vertices) > 0:
            # Laplacian smoothing while preserving volume
            for _ in range(smooth_iterations):
                trimesh.smoothing.filter_laplacian(mesh, iterations=1, lamb=0.5)

    # 7. Fix mesh if requested (fill holes, remove duplicate vertices, etc.)
    if fix_mesh:
        # Make sure mesh is watertight
        if not mesh.is_watertight:
            print("Attempting to fix non-watertight mesh...")
            # Fill holes
            mesh.fill_holes()
            # Remove duplicate vertices
            mesh = mesh.process(validate=True)

    # 8. If we padded the data, adjust the vertices to compensate
    if padding > 0:
        # Calculate the scale factor if we upscaled
        scale_factor: float = 1.0
        if upscale_factor and upscale_factor > 1:
            scale_factor = 1.0 / upscale_factor

        # Shift the vertices back by the padding amount, adjusted for any scaling
        mesh.vertices = (mesh.vertices * scale_factor) - (padding * scale_factor)

    # 9. Save the mesh as an STL file or return the mesh object
    if output_file:
        mesh.export(output_file)
        if mesh.is_watertight:
            print(f"Conversion complete! Watertight mesh saved as '{output_file}'")
        else:
            print(
                f"Conversion complete! Mesh saved as '{output_file}' (note: mesh may not be completely watertight)"
            )
        return output_file
    else:
        if mesh.is_watertight:
            print("Conversion complete! Watertight mesh generated.")
        else:
            print("Conversion complete! Mesh generated (note: mesh may not be completely watertight)")
        return mesh




# Define the main conversion function
def voxel_to_stl_tpms(
    input_npy_path: Union[str, np.ndarray],
    output_stl_path: Optional[str] = "gyroid_with_shell.stl",
    vox_len: float = 4.0,
    unit_len: float = 4.0,
    C_min: float = 0.0,
    C_max: float = 1.0,
    res_factor: int = 4,
    t_shell: float = 2,
    zero_thresh: float = 1e-6,
    plot_mapping: bool = True,
) -> Union[str, trimesh.Trimesh]:
    """
    Converts a 3D density mask from a numpy array or file to a STL mesh with a TPMS infill and a shell.

    Args:
        input_npy_path: Path to the input numpy file containing the 3D density array or the array itself.
        output_stl_path: Path for the output STL file. If None, the trimesh object is returned.
                         Defaults to "gyroid_with_shell.stl".
        vox_len: Voxel length in mm.
                 Increasing this value means each voxel in the input array represents a larger physical space,
                 leading to a coarser representation of the input density. Decreasing it leads to a finer representation.
                 Defaults to 4.0.
        unit_len: Unit cell length in mm. This defines the size of the repeating TPMS unit cell.
                  Increasing this value makes the TPMS features (e.g., pores, struts) larger and more sparse.
                  Decreasing it makes the TPMS features smaller and denser. Defaults to 4.0.
        C_min: κ mapping range minimum. This, along with C_max, defines the range of the iso-surface constant κ,
               which is mapped from the input density ρ and controls TPMS thickness.
               Increasing C_min (appropriately with C_max) generally leads to thicker TPMS structures.
               Decreasing C_min allows for thinner TPMS structures or finer features at lower densities.
               Defaults to 0.0.
        C_max: κ mapping range maximum.
               Increasing C_max (appropriately with C_min) allows for a wider range of TPMS thicknesses,
               potentially leading to thicker structures at high densities. Decreasing C_max limits the
               maximum thickness. Defaults to 1.0.
        res_factor: Interpolation resolution factor for the TPMS generation grid.
                    Increasing this value increases the number of evaluation points for the TPMS field,
                    leading to higher resolution, finer details, and smoother surfaces in the generated TPMS structure,
                    but increases computation time and memory. Decreasing it results in a coarser, faster-to-compute TPMS.
                    Defaults to 4.
        t_shell: Shell thickness in mm.
                 Increasing this value creates a thicker solid shell around the TPMS-infilled core.
                 Decreasing it creates a thinner solid shell. Defaults to 2.
        zero_thresh: Zero density threshold for masking. Regions in the input density with values below this
                     threshold are considered empty and excluded from materialization.
                     Increasing this value means more of the low-density input is treated as empty, potentially
                     shrinking the design. Decreasing it allows material generation in lower-density regions.
                     Defaults to 1e-6.
        plot_mapping: Whether to plot and save the κ-ρ mapping function. Defaults to True.

    Returns:
        Union[str, trimesh.Trimesh]: The path to the saved STL file if output_stl_path is provided, otherwise the generated trimesh.Trimesh object.
    """
    warn_stl_axis_change_once()

    logging.info("Starting mat2stl conversion with shell from density mask...")
    t_start = time.time()

    # ── 1. Load voxel data and set parameters ───────────────────────────────
    logging.info("1. Loading input data...")
    t1 = time.time()
    if isinstance(input_npy_path, str):
        V = np.load(input_npy_path)  # Bounding-box cropped 3D density array
    elif isinstance(input_npy_path, np.ndarray):
        V = input_npy_path
    else:
        raise TypeError("input_npy_path must be a string path or a NumPy array.")
    # Match voxel_to_stl / stl_to_design_space: densities are (nely, nelx, nelz);
    # swap the first two axes so the TPMS mesh's x maps to the domain's x (nelx).
    V = V.swapaxes(0, 1)
    logging.debug(f"   Data shape: {V.shape}   (took {time.time() - t1:.2f}s)")

    # ── 2. Define regular & evaluation grids ─────────────────────────────────
    logging.info("2. Defining grids...")
    t1 = time.time()
    Nx, Ny, Nz = V.shape
    phys = np.array([Nx, Ny, Nz]) * vox_len
    x_cent = np.linspace(vox_len / 2, phys[0] - vox_len / 2, Nx)
    y_cent = np.linspace(vox_len / 2, phys[1] - vox_len / 2, Ny)
    z_cent = np.linspace(vox_len / 2, phys[2] - vox_len / 2, Nz)

    eval_dims = np.maximum(phys / vox_len * res_factor, 2).astype(int)
    x_eval = np.linspace(0, phys[0], eval_dims[0])
    y_eval = np.linspace(0, phys[1], eval_dims[1])
    z_eval = np.linspace(0, phys[2], eval_dims[2])
    Xg, Yg, Zg = np.meshgrid(x_eval, y_eval, z_eval, indexing="ij")
    logging.debug(f"   Eval dims: {eval_dims}   (took {time.time() - t1:.2f}s)")

    # ── 3. Interpolate density to fine grid ──────────────────────────────────
    logging.info("3. Interpolating density...")
    t1 = time.time()
    interp = RegularGridInterpolator(
        (x_cent, y_cent, z_cent), V, method="linear", bounds_error=False, fill_value=0
    )
    density = interp(np.stack([Xg, Yg, Zg], axis=-1))
    logging.debug(
        f"   Density range: {density.min():.4f} to {density.max():.4f}   (took {time.time() - t1:.2f}s)"
    )

    # ── 4. Strömberg nonlinear κ-mapping ─────────────────────────────────────
    logging.info("4. Applying Strömberg nonlinear κ-mapping...")
    t1 = time.time()
    alpha = np.array([0.1019, 0, 0.3790, 0, 0.5191, 0, 0])
    rho = density
    s = (
        alpha[0] * rho
        + alpha[1] * rho**2
        + alpha[2] * rho**3
        + alpha[3] * rho / (1 + (1 - rho))
        + alpha[4] * rho / (1 + 2 * (1 - rho))
        + alpha[5] * rho / (1 + 3 * (1 - rho))
        + alpha[6] * rho / (1 + 4 * (1 - rho))
    )
    kappa = C_min + s * (C_max - C_min)
    logging.debug(
        f"   κ range: {kappa.min():.4f} to {kappa.max():.4f}   (took {time.time() - t1:.2f}s)"
    )

    # Plot the κ–ρ mapping function once if requested
    if plot_mapping:
        plt.figure(figsize=(6, 4))
        x = np.linspace(0, 1, 500)
        s_plot = (
            alpha[0] * x
            + alpha[1] * x**2
            + alpha[2] * x**3
            + alpha[3] * x / (1 + (1 - x))
            + alpha[4] * x / (1 + 2 * (1 - x))
            + alpha[5] * x / (1 + 3 * (1 - x))
            + alpha[6] * x / (1 + 4 * (1 - x))
        )
        plt.plot(x, C_min + s_plot * (C_max - C_min), "b-", lw=2)
        plt.xlabel("Density ρ")
        plt.ylabel("κ(ρ)")
        plt.title("Strömberg nonlinear κ–mapping")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("stromberg_mapping.png", dpi=300)
        plt.close()

    # ── 5. Compute Gyroid scalar field φ(x,y,z) − κ(ρ) ───────────────────────
    logging.info("5. Calculating Gyroid scalar field...")
    t1 = time.time()
    omega = 2 * np.pi / unit_len
    phi = (
        np.sin(omega * Xg) * np.cos(omega * Yg)
        + np.sin(omega * Yg) * np.cos(omega * Zg)
        + np.sin(omega * Zg) * np.cos(omega * Xg)
    )
    scalar_field = phi - kappa
    logging.debug(
        f"   Scalar field range: {scalar_field.min():.4f} to {scalar_field.max():.4f}   (took {time.time() - t1:.2f}s)"
    )

    # ── 6. Mask zero-density regions ─────────────────────────────────────────
    logging.info("6. Masking zero-density regions...")
    t1 = time.time()
    mask0 = density < zero_thresh
    valid = scalar_field[~mask0]
    lv = (np.max(np.abs(valid)) * 10 + 1) if valid.size > 0 else 1e9
    scalar_field[mask0] = lv
    logging.debug(f"   Masked {mask0.sum()} / {mask0.size} points   (took {time.time() - t1:.2f}s)")

    # ── 7. Generate shell field & mask TPMS outside core ────────────────────
    logging.info("7. Computing shell from density mask (pure shell + pure TPMS core)...")
    t1 = time.time()

    # 7.1 domain mask & signed‐distance (as before)
    domain_mask = density >= zero_thresh  # where any material exists
    dx = x_eval[1] - x_eval[0]
    dist_in = distance_transform_edt(domain_mask) * dx
    dist_out = distance_transform_edt(~domain_mask) * dx
    signed_dist = dist_in - dist_out

    # 7.2 shell field
    shell_field = signed_dist - t_shell
    shell_field[~domain_mask] = np.inf  # outside original domain, no shell

    # 7.3 carve out TPMS so it only lives **inside core** (signed_dist > t_shell)
    #     i.e. remove any TPMS inside the shell thickness band
    shell_region = signed_dist <= t_shell
    scalar_field[shell_region] = +np.inf  # force combined_field to pick shell_field there

    # 7.4 combine for final union:
    #      - inside shell band → shell_field <= 0 gives solid shell
    #      - outside shell band → combined_field = scalar_field (gyroid carve)
    combined_field = np.minimum(scalar_field, shell_field)

    logging.info(f"   Shell+core field built   (took {time.time() - t1:.2f}s)")

    # ── 8. Marching cubes & export STL ─────────────────────────────────────
    logging.info("8. Generating mesh with marching cubes...")
    t1 = time.time()
    verts, faces, normals, _ = measure.marching_cubes(
        combined_field, level=0, spacing=(dx, dx, dx)
    )
    logging.debug(
        f"   Mesh: {len(verts)} verts, {len(faces)} faces   (took {time.time() - t1:.2f}s)"
    )

    logging.info("9. Exporting to binary STL or returning mesh...")
    t1 = time.time()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    if output_stl_path:
        mesh.export(output_stl_path)
        logging.info(f"   STL saved to {output_stl_path}   (took {time.time() - t1:.2f}s)")
        logging.info(f"All done in {time.time() - t_start:.2f}s.")
        return output_stl_path
    else:
        logging.info(f"   Mesh generated   (took {time.time() - t1:.2f}s)")
        logging.info(f"All done in {time.time() - t_start:.2f}s.")
        return mesh

# # Example usage
# if __name__ == "__main__":
#     voxel_to_stl(
#         input_file="results/pytopo3d_60x30x20_obstacles_config_cylinder/optimized_design.npy",
#         output_file="results/pytopo3d_60x30x20_obstacles_config_cylinder/output_mesh.stl",
#         padding=1,
#         smooth_mesh=True,
#         smooth_iterations=5,
#     )
