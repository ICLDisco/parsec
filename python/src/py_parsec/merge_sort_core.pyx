# cython: language_level=3
"""
Direct PaRSEC core API bindings for merge_sort.
Mirrors tests/apps/merge_sort/main.c (in the PaRSEC repo root).
"""
from cpython.pycapsule cimport PyCapsule_New

cdef extern from "parsec/runtime.h":
    ctypedef struct parsec_context_t:
        pass
    ctypedef struct parsec_taskpool_t:
        pass

    parsec_context_t* parsec_init(int nb_cores, int* pargc, char*** pargv) nogil
    int parsec_fini(parsec_context_t** ctx) nogil
    int parsec_context_add_taskpool(parsec_context_t* context, parsec_taskpool_t* tp) nogil
    int parsec_context_start(parsec_context_t* context) nogil
    int parsec_context_wait(parsec_context_t* context) nogil
    int parsec_taskpool_wait(parsec_taskpool_t* tp) nogil
    void parsec_taskpool_free(parsec_taskpool_t* tp) nogil

cdef extern from "parsec/data_dist/matrix/matrix.h":
    ctypedef struct parsec_tiled_matrix_t:
        pass

cdef extern from "parsec/data.h":
    ctypedef struct parsec_data_collection_t:
        pass
    void parsec_data_collection_set_key(parsec_data_collection_t* dc, const char* key) nogil

cdef extern from "sort_data.h":
    parsec_tiled_matrix_t* create_and_distribute_data(int rank, int world, int nb, int nt, int typesize) nogil
    void free_data(parsec_tiled_matrix_t* d) nogil

cdef extern from "merge_sort_wrapper.h":
    parsec_taskpool_t* merge_sort_new(parsec_tiled_matrix_t* A, int size, int nt) nogil


cdef class ParsecMergeSortContext:
    """Direct wrapper for parsec_context_t, for merge_sort (core API)."""
    cdef parsec_context_t* _ctx

    def __cinit__(self, int nb_cores=-1):
        cdef int argc = 0
        cdef char** argv = NULL
        with nogil:
            self._ctx = parsec_init(nb_cores, &argc, &argv)
        if self._ctx == NULL:
            raise RuntimeError("parsec_init failed")

    def fini(self):
        cdef parsec_context_t* tmp = self._ctx
        if tmp != NULL:
            with nogil:
                parsec_fini(&tmp)
            self._ctx = NULL

    def __dealloc__(self):
        self.fini()

    def add_taskpool(self, taskpool):
        cdef parsec_taskpool_t* tp = (<ParsecMergeSortTaskpool>taskpool)._tp
        cdef int ret
        with nogil:
            ret = parsec_context_add_taskpool(self._ctx, tp)
        if ret != 0:
            raise RuntimeError(f"parsec_context_add_taskpool failed with code {ret}")

    def start(self):
        cdef int ret
        with nogil:
            ret = parsec_context_start(self._ctx)
        if ret != 0:
            raise RuntimeError(f"parsec_context_start failed with code {ret}")

    def wait(self):
        cdef int ret
        with nogil:
            ret = parsec_context_wait(self._ctx)
        if ret != 0:
            raise RuntimeError(f"parsec_context_wait failed with code {ret}")


cdef class ParsecMergeSortMatrix:
    """Wrapper for the merge_sort data descriptor created by sort_data.c."""
    cdef parsec_tiled_matrix_t* _mat
    cdef bytes _key

    def __cinit__(self, int rank, int world, int nb, int nt, int typesize=4, str key="A"):
        self._mat = NULL
        self._key = key.encode("utf-8")
        with nogil:
            self._mat = create_and_distribute_data(rank, world, nb, nt, typesize)
        if self._mat == NULL:
            raise MemoryError("create_and_distribute_data failed")

        cdef const char* key_ptr = self._key
        cdef parsec_data_collection_t* dc = <parsec_data_collection_t*>self._mat
        with nogil:
            parsec_data_collection_set_key(dc, key_ptr)

    def __dealloc__(self):
        if self._mat != NULL:
            with nogil:
                free_data(self._mat)
            self._mat = NULL

    def as_capsule(self):
        """Return a PyCapsule for passing to C API if needed."""
        return PyCapsule_New(<void*>self._mat, b"parsec_tiled_matrix_t", NULL)


cdef class ParsecMergeSortTaskpool:
    """Wrapper for the merge_sort taskpool created by merge_sort_new."""
    cdef parsec_taskpool_t* _tp

    def __cinit__(self, ParsecMergeSortMatrix A, int nb, int nt):
        self._tp = NULL
        with nogil:
            self._tp = merge_sort_new(A._mat, nb, nt)
        if self._tp == NULL:
            raise RuntimeError("merge_sort_new failed")

    def free(self):
        if self._tp != NULL:
            with nogil:
                parsec_taskpool_free(self._tp)
            self._tp = NULL

    def wait(self):
        cdef int ret
        if self._tp == NULL:
            raise RuntimeError("Taskpool is not initialized")
        with nogil:
            ret = parsec_taskpool_wait(self._tp)
        if ret != 0:
            raise RuntimeError(f"parsec_taskpool_wait failed with code {ret}")

    def __dealloc__(self):
        self.free()

