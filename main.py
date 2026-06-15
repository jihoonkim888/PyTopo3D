#!/usr/bin/env python3
"""
Main entry point for the 3D topology optimization package.

This script provides a command-line interface to run the topology optimization.
"""

import sys

from pytopo3d.cli.parser import parse_args
from pytopo3d.preprocessing.geometry import load_geometry_data
from pytopo3d.runners.experiment import (
    execute_optimization,
    export_result_to_stl,
    setup_experiment,
)
from pytopo3d.utils.assembly import build_force_vector, build_supports
from pytopo3d.utils.boundary import create_bc_visualization_arrays
from pytopo3d.utils.metrics import collect_metrics
from pytopo3d.visualization.visualizer import (
    create_optimization_animation,
    visualize_final_result,
    visualize_initial_setup,
)


def main():
    """
    Main function to run the optimization from command-line arguments.
    """
    # Parse command-line arguments
    args = parse_args()

    try:
        # Setup experiment, logging and results manager
        logger, results_mgr = setup_experiment(
            verbose=args.verbose,
            quiet=args.quiet,
            log_level=args.log_level,
            log_file=args.log_file,
            experiment_name=getattr(args, "experiment_name", None),
            description=args.description,
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=args.rmin,
        )

        # Update args.experiment_name if it was generated in setup_experiment
        if not hasattr(args, "experiment_name") or not args.experiment_name:
            args.experiment_name = results_mgr.experiment_name

        # Load design space and obstacle data
        design_space_mask, obstacle_mask, combined_obstacle_mask = load_geometry_data(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            design_space_stl=getattr(args, "design_space_stl", None),
            pitch=getattr(args, "pitch", 1.0),
            invert_design_space=getattr(args, "invert_design_space", False),
            obstacle_config=getattr(args, "obstacle_config", None),
            experiment_name=args.experiment_name,
            logger=logger,
            results_mgr=results_mgr,
        )

        # An STL design space defines the working grid via its voxelized resolution
        # (mesh size / pitch). Adopt it so the DOFs, boundary conditions, visualization,
        # and solver all match the STL-shaped masks; otherwise they keep the CLI
        # nelx/nely/nelz and crash on a shape mismatch. The CLI dims are ignored in this
        # case, so warn rather than silently change the grid the user may have requested.
        if getattr(args, "design_space_stl", None):
            args.nely, args.nelx, args.nelz = design_space_mask.shape
            logger.warning(
                f"--design-space-stl set: using STL-derived resolution "
                f"nelx={args.nelx}, nely={args.nely}, nelz={args.nelz}; "
                f"any --nelx/--nely/--nelz values are ignored."
            )

        # Determine number of DOFs
        ndof = 3 * (args.nelx + 1) * (args.nely + 1) * (args.nelz + 1)

        # --- Build Boundary Conditions ---
        # TODO: Allow passing force_field and support_mask from args or config file
        force_field = None  # Use default for now
        support_mask = None  # Use default for now

        logger.info("Building force vector (using default settings)")
        F = build_force_vector(
            args.nelx, args.nely, args.nelz, ndof, force_field=force_field
        )
        logger.info("Building support constraints (using default settings)")
        freedofs0, fixeddof0 = build_supports(
            args.nelx, args.nely, args.nelz, ndof, support_mask=support_mask
        )

        # Create visualization arrays from actual BCs
        loads_array, constraints_array = create_bc_visualization_arrays(
            args.nelx, args.nely, args.nelz, ndof, F, fixeddof0
        )
        logger.info("Generated boundary condition visualization arrays")

        # Create and save initial visualization
        visualize_initial_setup(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            loads_array=loads_array,
            constraints_array=constraints_array,
            experiment_name=args.experiment_name,
            logger=logger,
            results_mgr=results_mgr,
            combined_obstacle_mask=combined_obstacle_mask,
        )

        # Run the optimization - Passing force_field and support_mask
        xPhys, history, run_time = execute_optimization(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=args.rmin,
            disp_thres=args.disp_thres,
            # Pass the variables (currently None for defaults)
            force_field=force_field,
            support_mask=support_mask,
            # Removed F, freedofs0, fixeddof0
            tolx=getattr(args, "tolx", 0.01),
            maxloop=getattr(args, "maxloop", 2000),
            create_animation=getattr(args, "create_animation", False),
            animation_frequency=getattr(args, "animation_frequency", 10),
            logger=logger,
            combined_obstacle_mask=combined_obstacle_mask,
            use_gpu=args.gpu,
        )

        # Save the result to the experiment directory
        result_path = results_mgr.save_result(xPhys, "optimized_design.npy")
        logger.debug(f"Optimization result saved to {result_path}")

        # Create visualization of the final result
        visualize_final_result(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            experiment_name=args.experiment_name,
            disp_thres=args.disp_thres,
            logger=logger,
            results_mgr=results_mgr,
            xPhys=xPhys,
            combined_obstacle_mask=combined_obstacle_mask,
            loads_array=loads_array,
            constraints_array=constraints_array,
        )

        # Create animation if history was captured
        gif_path = None
        if history:
            gif_path = create_optimization_animation(
                nelx=args.nelx,
                nely=args.nely,
                nelz=args.nelz,
                experiment_name=args.experiment_name,
                disp_thres=args.disp_thres,
                animation_frames=getattr(args, "animation_frames", 50),
                animation_fps=getattr(args, "animation_fps", 5),
                logger=logger,
                results_mgr=results_mgr,
                history=history,
                combined_obstacle_mask=combined_obstacle_mask,
                loads_array=loads_array,
                constraints_array=constraints_array,
            )

        # Export result as STL if requested
        stl_exported = export_result_to_stl(
            export_stl=getattr(args, "export_stl", False),
            stl_level=getattr(args, "stl_level", 0.5),
            smooth_stl=getattr(args, "smooth_stl", False),
            smooth_iterations=getattr(args, "smooth_iterations", 3),
            logger=logger,
            results_mgr=results_mgr,
            result_path=result_path,
        )

        # Collect and save metrics
        metrics = collect_metrics(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=args.rmin,
            disp_thres=args.disp_thres,
            tolx=getattr(args, "tolx", 0.01),
            maxloop=getattr(args, "maxloop", 2000),
            design_space_stl=getattr(args, "design_space_stl", None),
            pitch=getattr(args, "pitch", 1.0),
            obstacle_config=getattr(args, "obstacle_config", None),
            animation_fps=getattr(args, "animation_fps", 5),
            stl_level=getattr(args, "stl_level", 0.5),
            smooth_stl=getattr(args, "smooth_stl", False),
            smooth_iterations=getattr(args, "smooth_iterations", 3),
            xPhys=xPhys,
            design_space_mask=design_space_mask,
            obstacle_mask=obstacle_mask,
            combined_obstacle_mask=combined_obstacle_mask,
            run_time=run_time,
            gif_path=gif_path,
            stl_exported=stl_exported,
        )
        results_mgr.update_metrics(metrics)
        logger.debug("Metrics updated")

        logger.info(f"Optimization complete in {run_time:.2f} seconds.")
        logger.info(f"Result saved to {result_path}")
        logger.info(f"All experiment files are in {results_mgr.experiment_dir}")

    except Exception as e:
        if "logger" in locals():
            logger.error(f"Error in main function: {e}")
            import traceback

            logger.debug(f"Error details: {traceback.format_exc()}")
        else:
            print(f"Error during initialization: {e}", file=sys.stderr)
            import traceback

            print(f"Error details: {traceback.format_exc()}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
