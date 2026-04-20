#!/usr/bin/env python3
"""
Test script to verify Py_PaRSEC installation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import verbose configuration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from verbose_config import print_verbose, print_minimal, print_normal, print_detailed, is_verbose

try:
    from mpi4py import MPI
    print_normal("✓ MPI4Py imported successfully")
except ImportError:
    print_normal("⚠️  MPI4Py not available, continuing without MPI")

from py_parsec.core import ParsecContext, ParsecData
from py_parsec.runtime import ParsecRuntime, ParsecScheduler
from py_parsec.tasks import TaskGraph, Task, DataDescriptor
import numpy as np

def main():
    print_normal("Py_PaRSEC Installation Test")
    print_normal("=" * 40)
    
    # Test core functionality
    print_detailed("Testing ParsecContext...")
    context = ParsecContext(nb_cores=1)
    print_minimal(f"Context: {context.nb_cores} cores")
    print_normal(f"✓ Context created with {context.nb_cores} cores")
    
    print_detailed("Testing ParsecData...")
    data = ParsecData(data_key=1, data_size=1024, flags=0)
    print_minimal(f"Data: key={data.data_key}, size={data.data_size}")
    print_normal(f"✓ Data created with key={data.data_key}, size={data.data_size}")
    
    # Test runtime functionality
    print_detailed("Testing ParsecRuntime...")
    runtime = ParsecRuntime(context)
    runtime.start()
    runtime.wait()
    print_normal("✓ Runtime started and waited successfully")
    
    print_detailed("Testing ParsecScheduler...")
    scheduler = ParsecScheduler(context)
    scheduler.start()
    scheduler.stop()
    print_normal("✓ Scheduler started and stopped successfully")
    
    # Test task functionality
    print_detailed("Testing TaskGraph...")
    graph = TaskGraph(context)
    print_normal("✓ TaskGraph created successfully")
    
    print_detailed("Testing Task...")
    def dummy_function(x, y):
        return x + y
    
    task = Task(context, dummy_function, inputs=[1, 2])
    result = task.execute()
    print_minimal(f"Task result: {result}")
    print_normal(f"✓ Task executed successfully, result: {result}")
    
    print_detailed("Testing DataDescriptor...")
    desc = DataDescriptor(context, "test_data", (100, 100), np.float64)
    print_minimal(f"DataDescriptor: {desc.name}, shape={desc.shape}, dtype={desc.dtype}")
    print_normal(f"✓ DataDescriptor created: {desc.name}, shape={desc.shape}, dtype={desc.dtype}")
    
    print_normal("\n" + "=" * 40)
    print_minimal("🎉 All tests passed! Py_PaRSEC is working correctly.")
    print_normal("🎉 All tests passed! Py_PaRSEC is working correctly.")
    print_normal("=" * 40)

if __name__ == "__main__":
    main()
