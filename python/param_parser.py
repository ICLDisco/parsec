#!/usr/bin/env python3
"""
Common parameter parser for Py_PaRSEC examples and tests.

This module provides a unified command-line argument parsing system
for both stencil and DTD examples/tests.
"""

import argparse
import sys
import os


class ParsecParams:
    """Container for parsed PaRSEC parameters"""
    
    def __init__(self):
        # Common parameters
        self.verbose = 1
        self.quiet = False
        self.debug = False
        
        # Matrix parameters
        self.M = 8
        self.N = 8
        self.K = 8
        
        # Block/tile parameters
        self.mb = 4
        self.nb = 4
        self.kb = 4
        
        # Stencil-specific parameters
        self.iterations = 10
        self.radius = 1
        
        # DTD-specific parameters
        self.device = "CPU"
        self.nruns = 5
        self.P = -1
        self.Q = -1
        self.Alarm = 0.0
        
        # Performance parameters
        self.cores = -1
        self.rank = 0
        self.world_size = 1


def create_common_parser(description="Py_PaRSEC Example"):
    """Create a common argument parser with shared options"""
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stencil example
  python stencil_1d.py 100 100 10 10 5 1 --verbose 1
  
  # DTD GEMM example
  python dtd_simple_gemm.py --M 1024 --mb 128 --device CPU --verbose 0
        """
    )
    
    # Verbose control
    parser.add_argument('--verbose', nargs='?', const=2, type=int, choices=[0, 1, 2, 10],
                       help='Verbose level: 0=minimal, 1=normal (default), 2=detailed, 10=very detailed (task messages)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (same as --verbose=0)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser


def create_stencil_parser(description="Py_PaRSEC Stencil Example"):
    """Create argument parser for stencil examples/tests"""
    
    parser = create_common_parser(description)
    
    # Positional arguments (for backward compatibility)
    parser.add_argument('M', type=int, nargs='?', default=8,
                       help='Matrix height (default: 8)')
    parser.add_argument('N', type=int, nargs='?', default=8,
                       help='Matrix width (default: 8)')
    parser.add_argument('MB', type=int, nargs='?', default=4,
                       help='Row tile size (default: 4)')
    parser.add_argument('NB', type=int, nargs='?', default=4,
                       help='Column tile size (default: 4)')
    parser.add_argument('iter', type=int, nargs='?', default=10,
                       help='Number of iterations (default: 10)')
    parser.add_argument('R', type=int, nargs='?', default=1,
                       help='Stencil radius (default: 1)')
    
    # Named arguments (alternative to positional)
    parser.add_argument('--M', type=int, dest='M_named',
                       help='Matrix height (alternative to positional)')
    parser.add_argument('--N', type=int, dest='N_named',
                       help='Matrix width (alternative to positional)')
    parser.add_argument('--MB', type=int, dest='MB_named',
                       help='Row tile size (alternative to positional)')
    parser.add_argument('--NB', type=int, dest='NB_named',
                       help='Column tile size (alternative to positional)')
    parser.add_argument('--iterations', type=int, dest='iter_named',
                       help='Number of iterations (alternative to positional)')
    parser.add_argument('--radius', type=int, dest='R_named',
                       help='Stencil radius (alternative to positional)')
    
    return parser


def create_dtd_parser(description="Py_PaRSEC DTD GEMM Example"):
    """Create argument parser for DTD examples/tests"""
    
    parser = create_common_parser(description)
    
    # Matrix dimensions
    parser.add_argument('--M', type=int, default=1024,
                       help='Matrix height (default: 1024)')
    parser.add_argument('--N', type=int, default=None,
                       help='Matrix width (default: same as --M)')
    parser.add_argument('--K', type=int, default=None,
                       help='Inner dimension (default: same as --M)')
    
    # Block sizes
    parser.add_argument('--mb', type=int, default=128,
                       help='Block height (default: 128)')
    parser.add_argument('--nb', type=int, default=None,
                       help='Block width (default: same as --mb)')
    parser.add_argument('--kb', type=int, default=None,
                       help='Block depth (default: same as --mb)')
    
    # Process grid
    parser.add_argument('--P', type=int, default=-1,
                       help='Process grid height (default: auto)')
    parser.add_argument('--Q', type=int, default=-1,
                       help='Process grid width (default: auto)')
    
    # Device and performance
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU',
                       help='Device to use (default: GPU)')
    parser.add_argument('--cores', type=int, default=-1,
                       help='Number of cores to use (default: -1, use all available)')
    parser.add_argument('--nruns', type=int, default=5,
                       help='Number of runs (default: 5)')
    parser.add_argument('--Alarm', type=float, default=0.0,
                       help='Minimum performance threshold (default: 0.0)')
    
    return parser


def parse_stencil_args(args=None):
    """Parse stencil-specific arguments and return ParsecParams object"""
    
    parser = create_stencil_parser()
    parsed_args = parser.parse_args(args)
    
    params = ParsecParams()
    
    # Handle verbose settings
    if parsed_args.quiet:
        params.verbose = 0
        params.quiet = True
    elif parsed_args.verbose is not None:
        params.verbose = parsed_args.verbose
    else:
        params.verbose = 1
    
    params.debug = parsed_args.debug
    
    # Use named arguments if provided, otherwise use positional
    params.M = parsed_args.M_named if parsed_args.M_named is not None else parsed_args.M
    params.N = parsed_args.N_named if parsed_args.N_named is not None else parsed_args.N
    params.mb = parsed_args.MB_named if parsed_args.MB_named is not None else parsed_args.MB
    params.nb = parsed_args.NB_named if parsed_args.NB_named is not None else parsed_args.NB
    params.iterations = parsed_args.iter_named if parsed_args.iter_named is not None else parsed_args.iter
    params.radius = parsed_args.R_named if parsed_args.R_named is not None else parsed_args.R
    
    return params


def parse_dtd_args(args=None):
    """Parse DTD-specific arguments and return ParsecParams object"""
    
    parser = create_dtd_parser()
    parsed_args = parser.parse_args(args)
    
    params = ParsecParams()
    
    # Handle verbose settings
    if parsed_args.quiet:
        params.verbose = 0
        params.quiet = True
    elif parsed_args.verbose is not None:
        params.verbose = parsed_args.verbose
    else:
        params.verbose = 1
    
    params.debug = parsed_args.debug
    
    # Matrix dimensions
    params.M = parsed_args.M
    params.N = parsed_args.N if parsed_args.N is not None else parsed_args.M
    params.K = parsed_args.K if parsed_args.K is not None else parsed_args.M
    
    # Block sizes
    params.mb = parsed_args.mb
    params.nb = parsed_args.nb if parsed_args.nb is not None else parsed_args.mb
    params.kb = parsed_args.kb if parsed_args.kb is not None else parsed_args.mb
    
    # Process grid
    params.P = parsed_args.P
    params.Q = parsed_args.Q
    
    # Device and performance
    params.device = parsed_args.device
    params.cores = parsed_args.cores
    params.nruns = parsed_args.nruns
    params.Alarm = parsed_args.Alarm
    
    return params


def print_params_summary(params, example_type="stencil"):
    """Print a summary of parsed parameters"""
    
    if params.verbose == 0:
        return  # Skip output for minimal verbose level
    
    print(f"Py_PaRSEC {example_type.upper()} Parameters")
    print("=" * 50)
    
    if example_type == "stencil":
        print(f"Matrix dimensions: {params.M}x{params.N}")
        print(f"Tile sizes: {params.mb}x{params.nb}")
        print(f"Iterations: {params.iterations}")
        print(f"Radius: {params.radius}")
        print(f"Cores: {params.cores}")
    elif example_type == "dtd":
        print(f"Matrix dimensions: {params.M}x{params.N}x{params.K}")
        print(f"Block sizes: {params.mb}x{params.nb}x{params.kb}")
        print(f"Process grid: {params.P}x{params.Q}")
        print(f"Device: {params.device}")
        print(f"Cores: {params.cores}")
        print(f"Runs: {params.nruns}")
    
    print(f"Verbose level: {params.verbose}")
    if params.debug:
        print("Debug mode: enabled")
    print()


def setup_verbose_system(params):
    """Setup verbose system based on parsed parameters"""
    
    # Set environment variable for verbose system
    if params.verbose == 0:
        os.environ['PARSEC_VERBOSE'] = '0'
        # Initialize verbose system to suppress all output
        try:
            from verbose_config import init_verbose_system
            init_verbose_system()
        except ImportError:
            pass
    else:
        os.environ['PARSEC_VERBOSE'] = str(params.verbose)


# Example usage functions
def main_stencil():
    """Example usage for stencil parameter parsing"""
    params = parse_stencil_args()
    print_params_summary(params, "stencil")
    setup_verbose_system(params)
    
    print(f"Running stencil with M={params.M}, N={params.N}, "
          f"MB={params.mb}, NB={params.nb}, iter={params.iterations}, R={params.radius}")


def main_dtd():
    """Example usage for DTD parameter parsing"""
    params = parse_dtd_args()
    print_params_summary(params, "dtd")
    setup_verbose_system(params)
    
    print(f"Running DTD GEMM with M={params.M}, N={params.N}, K={params.K}, "
          f"mb={params.mb}, nb={params.nb}, kb={params.kb}, device={params.device}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stencil":
        # Remove 'stencil' from argv and parse stencil args
        sys.argv = sys.argv[1:]
        main_stencil()
    elif len(sys.argv) > 1 and sys.argv[1] == "dtd":
        # Remove 'dtd' from argv and parse dtd args
        sys.argv = sys.argv[1:]
        main_dtd()
    else:
        print("Usage: python param_parser.py [stencil|dtd]")
        print("Examples:")
        print("  python param_parser.py stencil 100 100 10 10 5 1 --verbose 1")
        print("  python param_parser.py dtd --M 1024 --mb 128 --device CPU --verbose 0")
