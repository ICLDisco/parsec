# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

from py_parsec.cparsec cimport parsec_context_t, parsec_taskpool_t

ctypedef void (*taskpool_destructor_t)(parsec_taskpool_t *tp) noexcept nogil


cdef class Context:
    cdef parsec_context_t *_ctx
    cdef char **_c_argv
    cdef list _argv
    cdef dict _dtd_arenas
    cdef object _taskpools
    cdef object _collections
    cdef readonly int rank
    cdef readonly int nb_nodes
    cdef readonly int nb_cores
    cdef object __weakref__

    cdef parsec_context_t *ptr(self) except NULL
    cdef bint is_active(self)
    cdef int dtd_tile_arena(self, int mtype, int mb, int nb, int ld) except -1
    cdef register_collection(self, object collection)


cdef class Taskpool:
    cdef parsec_taskpool_t *_tp
    cdef taskpool_destructor_t _destruct
    cdef Context _context
    cdef list _keepalive
    cdef bint _needs_wait
    cdef object __weakref__

    cdef parsec_taskpool_t *ptr(self) except NULL
    cdef flush_collections(self)
    cdef check_errors(self)
    cdef release(self)
    cdef destroy(self)
    cdef forget(self)
