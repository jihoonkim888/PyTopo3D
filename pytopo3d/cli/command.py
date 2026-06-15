#!/usr/bin/env python3
"""
Command-line entry point for the pytopo3d package.
This module provides the main entry point when installed as a package.
"""

import sys
from typing import List, Optional

# Import the main function from the main module
from pytopo3d.cli.parser import parse_args
from pytopo3d.preprocessing.geometry import load_geometry_data
from pytopo3d.runners.experiment import (
    execute_optimization,
    export_result_to_stl,
    setup_experiment,
)
from pytopo3d.utils.metrics import collect_metrics
from pytopo3d.visualization.visualizer import (
    create_optimization_animation,
    visualize_final_result,
    visualize_initial_setup,
)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main function to run the optimization from command-line arguments.

    Args:
        args: Command line arguments (sys.argv[1:] by default)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Parse command-line arguments
    if args is None:
        args = sys.argv[1:]

    parsed_args = parse_args(args)

    try:
        # Setup experiment, logging and results manager
        logger, results_mgr = setup_experiment(
            verbose=parsed_args.verbose,
            quiet=parsed_args.quiet,
            log_level=parsed_args.log_level,
            log_file=parsed_args.log_file,
            experiment_name=getattr(parsed_args, "experiment_name", None),
            description=parsed_args.description,
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            volfrac=parsed_args.volfrac,
            penal=parsed_args.penal,
            rmin=parsed_args.rmin,
        )

        # Update args.experiment_name if it was generated in setup_experiment
        if (
            not hasattr(parsed_args, "experiment_name")
            or not parsed_args.experiment_name
        ):
            parsed_args.experiment_name = results_mgr.experiment_name

        # Load design space and obstacle data
        design_space_mask, obstacle_mask, combined_obstacle_mask = load_geometry_data(
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            design_space_stl=getattr(parsed_args, "design_space_stl", None),
            pitch=getattr(parsed_args, "pitch", 1.0),
            invert_design_space=getattr(parsed_args, "invert_design_space", False),
            obstacle_config=getattr(parsed_args, "obstacle_config", None),
            experiment_name=parsed_args.experiment_name,
            logger=logger,
            results_mgr=results_mgr,
        )

        # Create and save initial visualization
        loads_array, constraints_array, _ = visualize_initial_setup(
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            experiment_name=parsed_args.experiment_name,
            logger=logger,
            results_mgr=results_mgr,
            combined_obstacle_mask=combined_obstacle_mask,
        )

        # Run the optimization
        xPhys, history, run_time = execute_optimization(
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            volfrac=parsed_args.volfrac,
            penal=parsed_args.penal,
            rmin=parsed_args.rmin,
            disp_thres=parsed_args.disp_thres,
            tolx=getattr(parsed_args, "tolx", 0.01),
            maxloop=getattr(parsed_args, "maxloop", 2000),
            create_animation=getattr(parsed_args, "create_animation", False),
            animation_frequency=getattr(parsed_args, "animation_frequency", 10),
            logger=logger,
            combined_obstacle_mask=combined_obstacle_mask,
            use_gpu=parsed_args.gpu,
        )

        # Save the result to the experiment directory
        result_path = results_mgr.save_result(xPhys, "optimized_design.npy")
        logger.debug(f"Optimization result saved to {result_path}")

        # Create visualization of the final result
        visualize_final_result(
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            experiment_name=parsed_args.experiment_name,
            disp_thres=parsed_args.disp_thres,
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
                nelx=parsed_args.nelx,
                nely=parsed_args.nely,
                nelz=parsed_args.nelz,
                experiment_name=parsed_args.experiment_name,
                disp_thres=parsed_args.disp_thres,
                animation_frames=getattr(parsed_args, "animation_frames", 50),
                animation_fps=getattr(parsed_args, "animation_fps", 5),
                logger=logger,
                results_mgr=results_mgr,
                history=history,
                combined_obstacle_mask=combined_obstacle_mask,
                loads_array=loads_array,
                constraints_array=constraints_array,
            )

        # Export result as STL if requested
        stl_exported = export_result_to_stl(
            export_stl=getattr(parsed_args, "export_stl", False),
            stl_level=getattr(parsed_args, "stl_level", 0.5),
            smooth_stl=getattr(parsed_args, "smooth_stl", False),
            smooth_iterations=getattr(parsed_args, "smooth_iterations", 3),
            logger=logger,
            results_mgr=results_mgr,
            result_path=result_path,
        )

        # Collect and save metrics
        metrics = collect_metrics(
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            volfrac=parsed_args.volfrac,
            penal=parsed_args.penal,
            rmin=parsed_args.rmin,
            disp_thres=parsed_args.disp_thres,
            tolx=getattr(parsed_args, "tolx", 0.01),
            maxloop=getattr(parsed_args, "maxloop", 2000),
            design_space_stl=getattr(parsed_args, "design_space_stl", None),
            pitch=getattr(parsed_args, "pitch", 1.0),
            obstacle_config=getattr(parsed_args, "obstacle_config", None),
            animation_fps=getattr(parsed_args, "animation_fps", 5),
            stl_level=getattr(parsed_args, "stl_level", 0.5),
            smooth_stl=getattr(parsed_args, "smooth_stl", False),
            smooth_iterations=getattr(parsed_args, "smooth_iterations", 3),
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
