"""
Command-line argument parsing for the 3D topology optimization package.
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the topology optimization.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="3D Topology Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Basic parameters
    basic_group = parser.add_argument_group("Basic parameters")
    basic_group.add_argument(
        "--nelx", type=int, default=60, help="Number of elements in x direction"
    )
    basic_group.add_argument(
        "--nely", type=int, default=30, help="Number of elements in y direction"
    )
    basic_group.add_argument(
        "--nelz", type=int, default=20, help="Number of elements in z direction"
    )
    basic_group.add_argument(
        "--volfrac", type=float, default=0.3, help="Volume fraction constraint"
    )
    basic_group.add_argument(
        "--penal", type=float, default=3.0, help="Penalty parameter"
    )
    basic_group.add_argument("--rmin", type=float, default=3.0, help="Filter radius")
    basic_group.add_argument(
        "--disp_thres",
        type=float,
        default=0.5,
        help="Threshold for displaying elements in visualization",
    )
    basic_group.add_argument(
        "--tolx",
        type=float,
        default=0.01,
        help="Convergence tolerance on design change",
    )
    basic_group.add_argument(
        "--maxloop",
        type=int,
        default=2000,
        help="Maximum number of iterations",
    )

    # Output parameters
    output_group = parser.add_argument_group("Output parameters")
    output_group.add_argument(
        "--output",
        type=str,
        default="optimized_design.npy",
        help="Output filename for the optimized design",
    )
    output_group.add_argument(
        "--export-stl",
        action="store_true",
        help="Export the final optimization result as an STL file",
    )
    output_group.add_argument(
        "--stl-level",
        type=float,
        default=0.5,
        help="Contour level for STL export (default: 0.5)",
    )
    output_group.add_argument(
        "--smooth-stl",
        action="store_true",
        default=True,
        help="Apply smoothing to the exported STL (default: True)",
    )
    output_group.add_argument(
        "--smooth-iterations",
        type=int,
        default=5,
        help="Number of smoothing iterations for STL export (default: 5)",
    )
    output_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom name for the experiment (optional)",
    )
    output_group.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description of the experiment (optional)",
    )

    # Obstacle related arguments
    obstacle_group = parser.add_argument_group("Obstacle parameters")
    obstacle_group.add_argument(
        "--obstacle-config", type=str, help="Path to a JSON file defining obstacles"
    )

    # Logging parameters
    log_group = parser.add_argument_group("Logging parameters")
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    log_group.add_argument("--log-file", type=str, default=None, help="Log file path")
    log_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (DEBUG level)",
    )
    log_group.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output (WARNING level)"
    )

    return parser.parse_args()


def generate_experiment_name(args: argparse.Namespace) -> str:
    """
    Generate an experiment name from command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    str
        Generated experiment name.
    """
    if args.experiment_name:
        return args.experiment_name

    dims = f"{args.nelx}x{args.nely}x{args.nelz}"

    # Include obstacle info in experiment name
    obstacle_type = "no_obstacle"
    if args.obstacle_config:
        obstacle_type = os.path.basename(args.obstacle_config).replace(".json", "")

    return f"{dims}_{obstacle_type}"


def create_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create a configuration dictionary from command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    return config
