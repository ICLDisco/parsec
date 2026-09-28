#!/usr/bin/env python3
"""
Verbose configuration for Py_PaRSEC tests and examples
"""

import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Default verbose level
DEFAULT_VERBOSE = 1

# Global variable to store original stdout/stderr
_original_stdout = sys.stdout
_original_stderr = sys.stderr

def get_verbose_level():
    """Get verbose level from environment variable or command line args"""
    # Check environment variable first
    verbose = os.environ.get('PARSEC_VERBOSE', DEFAULT_VERBOSE)
    
    # Check command line arguments
    if '--verbose' in sys.argv:
        verbose = 2
    elif '--quiet' in sys.argv or '--verbose=0' in sys.argv:
        verbose = 0
    
    # Check for --verbose=N format
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--verbose='):
            try:
                verbose = int(arg.split('=')[1])
            except ValueError:
                verbose = DEFAULT_VERBOSE
        elif arg == '--verbose' and i + 1 < len(sys.argv):
            # Handle --verbose N format (space instead of equals)
            try:
                verbose = int(sys.argv[i + 1])
            except ValueError:
                verbose = DEFAULT_VERBOSE
    
    return int(verbose)

def print_verbose(level, message, min_level=1):
    """Print message only if current verbose level >= min_level"""
    current_level = get_verbose_level()
    if current_level >= min_level:
        # For minimal output, suppress PaRSEC internal messages
        if current_level == 0 and min_level == 0:
            # Temporarily restore stdout for our minimal output
            sys.stdout = _original_stdout
            print(message)
            # Redirect stdout back to suppress PaRSEC messages
            sys.stdout = StringIO()
        else:
            print(message)

def print_minimal(message):
    """Print minimal output (verbose level 0)"""
    current_level = get_verbose_level()
    if current_level == 0:
        # For minimal output, only print if it's a performance message
        if "Performance:" in message or "Matrix:" in message:
            # Temporarily restore stdout to print our message
            # then redirect back to suppress PaRSEC messages
            original_stdout = sys.stdout
            sys.stdout = _original_stdout
            print(message)
            sys.stdout = original_stdout
        # Otherwise, suppress all output
    else:
        print(message)

def print_normal(message):
    """Print normal output (verbose level 1)"""
    print_verbose(1, message, 1)

def print_detailed(message):
    """Print detailed output (verbose level 2)"""
    print_verbose(2, message, 2)

def print_very_detailed(message):
    """Print very detailed output (verbose level 10+)"""
    print_verbose(10, message, 10)

# Global verbose level
VERBOSE_LEVEL = get_verbose_level()

def is_verbose(level=1):
    """Check if current verbose level >= level"""
    return VERBOSE_LEVEL >= level

class FilteredOutput:
    """Custom output filter to suppress specific PaRSEC messages"""
    def __init__(self, original_stream, verbose_level):
        self.original_stream = original_stream
        self.verbose_level = verbose_level
        self.buffer = ""
    
    def write(self, text):
        # Buffer the text to handle multi-line messages
        self.buffer += text
        
        # Process complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            # Filter out PaRSEC internal messages for verbose levels 1-9
            if self.verbose_level < 10:
                if any(msg in line for msg in [
                    "Inserted task with class GEMM",
                    "Data flush all completed",
                    "Taskpool wait completed",
                    "PaRSEC context wait completed",
                    "Released task class:",
                    "Created task class:",
                    "Added chore for device type",
                    "Created DTD taskpool",
                    "Taskpool added to context",
                    "PaRSEC context started"
                ]):
                    continue
            self.original_stream.write(line + '\n')
        
        # Flush any remaining buffer
        if self.buffer:
            self.original_stream.write(self.buffer)
            self.buffer = ""
    
    def flush(self):
        if self.buffer:
            self.original_stream.write(self.buffer)
            self.buffer = ""
        self.original_stream.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stream, name)

def init_verbose_system():
    """Initialize the verbose system based on current verbose level"""
    current_level = get_verbose_level()
    if current_level == 0:
        # For minimal output, set PaRSEC's own verbose level to suppress its output
        os.environ['PARSEC_VERBOSE'] = '0'
        
        # Also redirect stdout/stderr to suppress any remaining output
        sys.stdout = StringIO()
        sys.stderr = StringIO()
    elif current_level >= 10:
        # For very detailed output (level 10+), allow PaRSEC to show task insertion messages
        os.environ['PARSEC_VERBOSE'] = '10'
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr
    else:
        # For normal and detailed output (levels 1-9), use aggressive filtering
        os.environ['PARSEC_VERBOSE'] = '0'
        sys.stdout = FilteredOutput(_original_stdout, current_level)
        sys.stderr = FilteredOutput(_original_stderr, current_level)

# Initialize the verbose system only if explicitly requested
# This prevents automatic initialization that might suppress output
# init_verbose_system()
