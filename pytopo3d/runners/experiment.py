"""
Experiment setup and management for topology optimization.

This module contains functions for setting up and managing topology optimization experiments.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from pytopo3d.cli.parser import generate_experiment_name
from pytopo3d.core.optimizer import top3d
from pytopo3d.utils.export import voxel_to_stl
from pytopo3d.utils.logger import setup_logger
from pytopo3d.utils.results_manager import ResultsManager


def setup_experiment(
    verbose: bool = False,
    quiet: bool = False,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    experiment_name: Optional[str] = None,
    description: Optional[str] = None,
    nelx: int = 40,
    nely: int = 20,
    nelz: int = 10,
    volfrac: float = 0.3, 
    penal: float = 3.0,
    rmin: float = 1.5
) -> Tuple[logging.Logger, ResultsManager]:
    """
    Set up experiment name, logging, and results manager.

    Args:
        verbose: Whether to enable verbose logging
        quiet: Whether to enable quiet mode
        log_level: Logging level
        log_file: Path to log file
        experiment_name: Name of the experiment (if None, will be generated)
        description: Description of the experiment
        nelx: Number of elements in x direction (for name generation)
        nely: Number of elements in y direction (for name generation)
        nelz: Number of elements in z direction (for name generation)
        volfrac: Volume fraction (for name generation)
        penal: Penalization factor (for name generation)
        rmin: Filter radius (for name generation)

    Returns:
        Tuple containing configured logger and results manager
    """
    # Configure logging from parameters
    if verbose:
        log_level_value = logging.DEBUG
    elif quiet:
        log_level_value = logging.WARNING
    else:
        log_level_value = getattr(logging, log_level)

    # Setup logger
    logger = setup_logger(level=log_level_value, log_file=log_file)
    logger.debug("Logging configured successfully")

    # Generate experiment name if not provided
    if experiment_name is None:
        import hashlib
        from datetime import datetime

        # Generate a name based on parameters and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create parameter string
        param_str = f"{nelx}x{nely}x{nelz}_vf{volfrac}_p{penal}_r{rmin}"
        
        # Generate a short hash of the parameters
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:6]
        
        experiment_name = f"topo3d_{timestamp}_{param_hash}"
    
    logger.info(f"Experiment name: {experiment_name}")

    # Create a results manager for this experiment
    results_mgr = ResultsManager(
        experiment_name=experiment_name, description=description
    )
    logger.debug(
        f"Results manager created with experiment directory: {results_mgr.experiment_dir}"
    )

    return logger, results_mgr


def execute_optimization(
    nelx: int,
    nely: int,
    nelz: int,
    volfrac: float,
    penal: float,
    rmin: float,
    disp_thres: float,
    tolx: float = 0.01,
    maxloop: int = 2000,
    create_animation: bool = False,
    animation_frequency: int = 10,
    logger: logging.Logger = None,
    combined_obstacle_mask: Optional[np.ndarray] = None,
    benchmark: bool = False, 
    save_benchmark: bool = False,
    benchmark_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[Dict], float, Optional[Dict]]:
    """
    Run the topology optimization process.

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        nelz: Number of elements in z direction
        volfrac: Volume fraction constraint
        penal: Penalization factor
        rmin: Filter radius
        disp_thres: Threshold for displaying elements
        tolx: Convergence tolerance
        maxloop: Maximum number of iterations
        create_animation: Whether to save optimization history for animation
        animation_frequency: Frequency of saving frames for animation
        logger: Configured logger
        combined_obstacle_mask: Combined obstacle and design space mask
        benchmark: Whether to perform detailed performance benchmarking
        save_benchmark: Whether to save benchmark results to a file
        benchmark_dir: Directory to save benchmark results (if None, uses default)

    Returns:
        Tuple containing optimization result, history (if saved), runtime in seconds,
        and benchmark results (if benchmarking was enabled)
    """
    # Initialize benchmarking if requested
    benchmark_tracker = None
    if benchmark:
        try:
            from pytopo3d.utils.benchmarking import BenchmarkTracker
            benchmark_tracker = BenchmarkTracker(track_memory=True, track_detailed_timing=True)
            if logger:
                logger.info("Benchmarking enabled - tracking detailed performance metrics")
        except ImportError:
            if logger:
                logger.warning("Benchmarking module not found - continuing without benchmarking")
    
    # Run the optimization with timing
    if logger:
        logger.info(
            f"Starting optimization with {nelx}x{nely}x{nelz} elements..."
        )
    start_time = time.time()

    logger.debug(
        f"Optimization parameters: tolx={tolx}, maxloop={maxloop}, "
        f"save_history={create_animation}, history_frequency={animation_frequency}"
    )

    # Run the optimization with history if requested
    optimization_result = top3d(
        nelx,
        nely,
        nelz,
        volfrac,
        penal,
        rmin,
        disp_thres,
        obstacle_mask=combined_obstacle_mask,
        tolx=tolx,
        maxloop=maxloop,
        save_history=create_animation,
        history_frequency=animation_frequency,
        benchmark_tracker=benchmark_tracker,
    )

    # Check if we got history back
    history = None
    if create_animation:
        xPhys, history = optimization_result
        if logger:
            logger.info(
                f"Optimization history captured with {len(history['density_history'])} frames"
            )
    else:
        xPhys = optimization_result

    end_time = time.time()
    run_time = end_time - start_time
    if logger:
        logger.debug(f"Optimization finished in {run_time:.2f} seconds")
    
    # Process benchmark results if benchmarking was enabled
    benchmark_results = None
    if benchmark_tracker:
        benchmark_tracker.finalize()
        benchmark_results = benchmark_tracker.get_summary()
        
        # Log benchmark summary
        if logger:
            logger.info(f"Benchmark results: total time={benchmark_results['total_time_seconds']:.2f}s")
            if 'phases' in benchmark_results:
                logger.info("Phase timing breakdown:")
                for phase, data in benchmark_results['phases'].items():
                    logger.info(f"  {phase}: {data['total_seconds']:.2f}s ({data['percentage']:.1f}%)")
            
            if 'peak_memory_mb' in benchmark_results:
                logger.info(f"Peak memory usage: {benchmark_results['peak_memory_mb']:.1f} MB")
        
        # Save benchmark results to file if requested
        if save_benchmark and benchmark_tracker:
            if benchmark_dir is None:
                benchmark_dir = "results/benchmarks"
            
            os.makedirs(benchmark_dir, exist_ok=True)
            problem_size = nelx * nely * nelz
            benchmark_file = os.path.join(
                benchmark_dir, 
                f"benchmark_size_{problem_size}_nelx{nelx}_nely{nely}_nelz{nelz}.json"
            )
            benchmark_tracker.save_to_file(benchmark_file)
            
            if logger:
                logger.info(f"Benchmark results saved to {benchmark_file}")

    return xPhys, history, run_time, benchmark_results


def export_result_to_stl(
    export_stl: bool = False,
    stl_level: float = 0.5,
    smooth_stl: bool = False,
    smooth_iterations: int = 3,
    logger: logging.Logger = None,
    results_mgr: ResultsManager = None,
    result_path: str = None,
) -> bool:
    """
    Export the optimization result as an STL file if requested.

    Args:
        export_stl: Whether to export as STL
        stl_level: Threshold level for STL export
        smooth_stl: Whether to smooth the STL mesh
        smooth_iterations: Number of smoothing iterations
        logger: Configured logger
        results_mgr: Results manager instance
        result_path: Path to the saved optimization result

    Returns:
        True if STL export was successful, False otherwise
    """
    if not export_stl:
        return False

    try:
        # Create the STL filename
        stl_filename = os.path.join(results_mgr.experiment_dir, "optimized_design.stl")

        # Export the result as an STL file
        if logger:
            logger.info("Exporting optimization result as STL file...")
        voxel_to_stl(
            input_file=result_path,
            output_file=stl_filename,
            level=stl_level,
            smooth_mesh=smooth_stl,
            smooth_iterations=smooth_iterations,
        )
        if logger:
            logger.info(f"STL file exported to {stl_filename}")
        return True

    except Exception as e:
        if logger:
            logger.error(f"Error exporting STL file: {e}")
        return False
