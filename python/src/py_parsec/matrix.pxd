# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

from py_parsec.cparsec cimport (parsec_data_collection_t, parsec_tiled_matrix_t,
                                parsec_matrix_block_cyclic_t)
from py_parsec.runtime cimport Context


cdef object host_array(void *ptr, int mb, int nb, int typenum, object owner)


cdef class TiledMatrix:
    cdef parsec_tiled_matrix_t *_desc
    cdef Context _context
    cdef bytes _name
    cdef readonly object dtype
    cdef int _typenum
    cdef bint _dtd_registered
    cdef object __weakref__

    cdef parsec_tiled_matrix_t *ptr(self) except NULL
    cdef parsec_data_collection_t *dc(self) except NULL
    cdef int _check_tile(self, int m, int n) except -1
    cdef dtd_register(self)
    cdef dtd_unregister(self)
    cdef destroy(self)


cdef class BlockCyclic(TiledMatrix):
    cdef parsec_matrix_block_cyclic_t *_bc
