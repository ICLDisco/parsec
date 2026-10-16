# Core PaRSEC functionality

import numpy as np

cdef class ParsecContext:
    """PaRSEC context wrapper"""
    
    cdef int _nb_cores
    cdef bint _started
    
    def __init__(self, int nb_cores=1):
        self._nb_cores = nb_cores
        self._started = False
        print(f"Created PaRSEC context with {nb_cores} cores")
    
    def start(self):
        """Start the PaRSEC context"""
        if not self._started:
            print("Starting PaRSEC context...")
            self._started = True
    
    def wait(self):
        """Wait for the PaRSEC context to complete"""
        if self._started:
            print("Waiting for PaRSEC context to complete...")
    
    def test(self):
        """Test if the PaRSEC context is complete"""
        if self._started:
            return 1  # Always complete for now
        return 0
    
    @property
    def nb_cores(self):
        """Get number of cores"""
        return self._nb_cores
    
    @property
    def started(self):
        """Check if context is started"""
        return self._started

cdef class ParsecData:
    """PaRSEC data wrapper"""
    
    cdef unsigned long long _data_key
    cdef size_t _data_size
    cdef unsigned char _flags
    
    def __init__(self, unsigned long long data_key, size_t data_size, unsigned char flags=0):
        self._data_key = data_key
        self._data_size = data_size
        self._flags = flags
        print(f"Created PaRSEC data with key={data_key}, size={data_size}")
    
    def create_copy(self, ptr, int dtt=0):
        """Create a data copy with the given pointer and datatype"""
        print(f"Creating data copy for key={self._data_key}")
    
    def get_ptr(self, unsigned int device=0):
        """Get pointer to data on specified device"""
        print(f"Getting pointer for device={device}")
        return None
    
    @property
    def data_key(self):
        """Get data key"""
        return self._data_key
    
    @property
    def data_size(self):
        """Get data size"""
        return self._data_size
    
    @property
    def flags(self):
        """Get data flags"""
        return self._flags
