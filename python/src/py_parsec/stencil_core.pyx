# cython: language_level=3
"""
Direct PaRSEC core API bindings for stencil - NO DTD!
Exactly mirrors the official testing_stencil_1D.c workflow.
"""
from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer

cdef extern from "parsec.h":
    ctypedef struct parsec_context_t:
        pass
    
    parsec_context_t* parsec_init(int nb_cores, int* pargc, char*** pargv) nogil
    int parsec_fini(parsec_context_t** ctx) nogil


cdef extern from "parsec/data_dist/matrix/matrix.h":
    ctypedef struct parsec_tiled_matrix_t:
        int mb, nb, m, n, mt, nt, mtype
    
    ctypedef struct parsec_execution_stream_t:
        pass
    
    ctypedef enum parsec_matrix_uplo_t:
        pass
    
    ctypedef int (*parsec_tiled_matrix_unary_op_t)(parsec_execution_stream_t *es,
                                                   const parsec_tiled_matrix_t *descA,
                                                   void *_A,
                                                   parsec_matrix_uplo_t uplo,
                                                   int m, int n,
                                                   void *args) nogil
    
    int parsec_apply(parsec_context_t* parsec,
                     parsec_matrix_uplo_t uplo,
                     parsec_tiled_matrix_t* A,
                     parsec_tiled_matrix_unary_op_t operation,
                     void* op_args) nogil


cdef extern from "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h":
    ctypedef struct parsec_matrix_block_cyclic_t:
        parsec_tiled_matrix_t super
        char* mat
    
    void parsec_matrix_block_cyclic_init(parsec_matrix_block_cyclic_t* dc,
                                         int mtype, int storage, int rank,
                                         int mb, int nb,
                                         int lm, int ln,
                                         int i, int j,
                                         int m, int n,
                                         int P, int Q,
                                         int kp, int kq,
                                         int ip, int jq) nogil
    
    void* parsec_data_allocate(size_t size) nogil
    void parsec_data_free(void* ptr) nogil
    void parsec_tiled_matrix_destroy_data(parsec_tiled_matrix_t* dc) nogil
    size_t parsec_datadist_getsizeoftype(int mtype) nogil


cdef extern from "parsec/data.h":
    ctypedef struct parsec_data_collection_t:
        pass
    
    void parsec_data_collection_set_key(parsec_data_collection_t* dc, const char* key) nogil
    void parsec_data_collection_destroy(parsec_data_collection_t* dc) nogil


# Stencil-specific functions
cdef extern from *:
    """
    #include "stencil_internal.h"
    #include <mpi.h>
    #include <sys/time.h>
    
    /* Timing helper like HiCMA */
    static inline double py_get_cur_time() {
        #ifdef PARSEC_HAVE_MPI
            return MPI_Wtime();
        #else
            struct timeval tv;
            double t;
            gettimeofday(&tv, NULL);
            t = tv.tv_sec + tv.tv_usec / 1e6;
            return t;
        #endif
    }
    
    /* SYNC_TIME macros from HiCMA */
    double sync_time_elapsed = 0.0;
    
    #ifdef PARSEC_HAVE_MPI
    #define PY_SYNC_TIME_START() do { \
            MPI_Barrier(MPI_COMM_WORLD); \
            sync_time_elapsed = py_get_cur_time(); \
        } while(0)
    #define PY_SYNC_TIME_STOP() do { \
            MPI_Barrier(MPI_COMM_WORLD); \
            sync_time_elapsed = py_get_cur_time() - sync_time_elapsed; \
        } while(0)
    #else
    #define PY_SYNC_TIME_START() do { \
            sync_time_elapsed = py_get_cur_time(); \
        } while(0)
    #define PY_SYNC_TIME_STOP() do { \
            sync_time_elapsed = py_get_cur_time() - sync_time_elapsed; \
        } while(0)
    #endif
    
    /* Define weight_1D globally */
    double* weight_1D = NULL;
    
    /* Local implementation of init operator */
    static int py_stencil_1D_init_ops(parsec_execution_stream_t *es,
                                const parsec_tiled_matrix_t *descA,
                                void *_A, parsec_matrix_uplo_t uplo,
                                int m, int n, void *args)
    {
        double *A = (double *)_A;
        int R = ((int *)args)[0];
        int i, j;

        for(j = R; j < descA->nb - R; j++)
            for(i = 0; i < descA->mb; i++)
                A[j*descA->mb+i] = (double)1.0 * i + (double)1.0 * j;

        for(j = 0; j < R; j++)
            for(i = 0; i < descA->mb; i++)
                A[j*descA->mb+i] = (double)0.0;

        for(j = descA->nb - R; j < descA->nb; j++)
            for(i = 0; i < descA->mb; i++)
                A[j*descA->mb+i] = (double)0.0;
        (void)es; (void)uplo; (void)m; (void)n;
        return 0;
    }
    
    static void py_init_weight_1D(int R) {
        int jj;
        if(weight_1D != NULL) {
            free(weight_1D);
        }
        weight_1D = (double*)malloc(sizeof(double) * (2*R + 1));
        for(jj = 1; jj <= R; jj++) {
            weight_1D[R + jj] = 1.0 / (2.0 * jj * R);
            weight_1D[R - jj] = -(1.0 / (2.0 * jj * R));
        }
        weight_1D[R] = 1.0;
    }
    
    /* Constants */
    static inline int py_PARSEC_MATRIX_FULL() { return PARSEC_MATRIX_FULL; }
    static inline int py_PARSEC_MATRIX_DOUBLE() { return PARSEC_MATRIX_DOUBLE; }
    static inline int py_PARSEC_MATRIX_TILE() { return PARSEC_MATRIX_TILE; }
    
    /* Helper to access sync_time_elapsed from Python */
    static inline double* py_get_sync_time_elapsed_ptr() {
        return &sync_time_elapsed;
    }
    """
    int parsec_stencil_1D(parsec_context_t* parsec,
                          parsec_tiled_matrix_t* A,
                          int iterations, int radius) nogil
    
    int py_stencil_1D_init_ops(parsec_execution_stream_t *es,
                               const parsec_tiled_matrix_t *descA,
                               void *_A, parsec_matrix_uplo_t uplo,
                               int m, int n, void *args) nogil
    
    void py_init_weight_1D(int R) nogil
    
    double py_get_cur_time() nogil
    double* py_get_sync_time_elapsed_ptr() nogil
    
    int py_PARSEC_MATRIX_FULL()
    int py_PARSEC_MATRIX_DOUBLE()
    int py_PARSEC_MATRIX_TILE()


# Export constants
PARSEC_MATRIX_FULL = py_PARSEC_MATRIX_FULL()
PARSEC_MATRIX_DOUBLE = py_PARSEC_MATRIX_DOUBLE()
PARSEC_MATRIX_TILE = py_PARSEC_MATRIX_TILE()


cdef class ParsecCoreContext:
    """Direct wrapper for parsec_context_t - NO DTD"""
    cdef parsec_context_t* _ctx
    
    def __cinit__(self, int nb_cores=0):
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
    
    def sync_time_start(self):
        """Start synchronized timing with MPI barrier (like SYNC_TIME_START)"""
        try:
            from mpi4py import MPI
            if MPI.Is_initialized():
                MPI.COMM_WORLD.Barrier()
        except ImportError:
            pass
        
        cdef double* sync_ptr
        with nogil:
            sync_ptr = py_get_sync_time_elapsed_ptr()
            sync_ptr[0] = py_get_cur_time()
    
    def sync_time_stop(self):
        """Stop synchronized timing with MPI barrier and return elapsed time (like SYNC_TIME_STOP)"""
        try:
            from mpi4py import MPI
            if MPI.Is_initialized():
                MPI.COMM_WORLD.Barrier()
        except ImportError:
            pass
        
        cdef double* sync_ptr
        cdef double elapsed_time
        with nogil:
            sync_ptr = py_get_sync_time_elapsed_ptr()
            sync_ptr[0] = py_get_cur_time() - sync_ptr[0]
            elapsed_time = sync_ptr[0]
        
        return elapsed_time
    
    def apply(self, matrix_capsule, int uplo, int radius):
        """Call parsec_apply to initialize matrix tiles"""
        cdef parsec_matrix_block_cyclic_t* dc = <parsec_matrix_block_cyclic_t*>PyCapsule_GetPointer(
            matrix_capsule, b"parsec_matrix_block_cyclic_t")
        cdef parsec_tiled_matrix_t* mat = &dc.super
        cdef int R = radius
        cdef int ret
        
        with nogil:
            ret = parsec_apply(self._ctx, <parsec_matrix_uplo_t>uplo, mat,
                             <parsec_tiled_matrix_unary_op_t>py_stencil_1D_init_ops, <void*>&R)
        if ret != 0:
            raise RuntimeError(f"parsec_apply failed with code {ret}")
    
    def stencil_1D(self, matrix_capsule, int iterations, int radius):
        """Run parsec_stencil_1D kernel"""
        cdef parsec_matrix_block_cyclic_t* dc = <parsec_matrix_block_cyclic_t*>PyCapsule_GetPointer(
            matrix_capsule, b"parsec_matrix_block_cyclic_t")
        cdef parsec_tiled_matrix_t* mat = &dc.super
        cdef int ret
        
        # Initialize weight_1D before calling kernel
        py_init_weight_1D(radius)
        
        with nogil:
            ret = parsec_stencil_1D(self._ctx, mat, iterations, radius)
        
        if ret != 0:
            raise RuntimeError(f"parsec_stencil_1D failed with code {ret}")


cdef class ParsecMatrix:
    """Direct wrapper for parsec_matrix_block_cyclic_t - NO DTD"""
    cdef parsec_matrix_block_cyclic_t* _dc
    
    def __cinit__(self):
        self._dc = NULL
    
    def init(self, str key, int myrank, int mb, int nb, int lm, int ln, int P, int Q,
             int kp=1, int kq=1, int mtype=0, int storage=0):
        """Initialize matrix like official C code (simplified API like dtd example)"""
        if mtype == 0:
            mtype = PARSEC_MATRIX_DOUBLE
        if storage == 0:
            storage = PARSEC_MATRIX_TILE
        
        if self._dc != NULL:
            raise RuntimeError("Matrix already initialized")
        
        self._dc = <parsec_matrix_block_cyclic_t*>malloc(sizeof(parsec_matrix_block_cyclic_t))
        if self._dc == NULL:
            raise MemoryError("Failed to allocate matrix descriptor")
        
        cdef bytes key_bytes = key.encode('utf-8')
        cdef const char* key_ptr = key_bytes
        
        # Official: parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
        #           rank, MB, NB+2*R, M, N+2*R*NNB, 0, 0, M, N+2*R*NNB, P, nodes/P, KP, KQ, 0, 0);
        with nogil:
            parsec_matrix_block_cyclic_init(self._dc, mtype, storage, myrank,
                                           mb, nb, lm, ln, 0, 0, lm, ln,
                                           P, Q, kp, kq, 0, 0)
        
        # Set key
        cdef parsec_data_collection_t* A = <parsec_data_collection_t*>&self._dc.super
        with nogil:
            parsec_data_collection_set_key(A, key_ptr)
        
        # Allocate contiguous buffer (like official: nb_local_tiles * bsiz * typesize)
        cdef size_t nb_local_tiles, bsiz, typesize, total_size
        cdef int mat_type = self._dc.super.mtype
        nb_local_tiles = <size_t>self._dc.super.nt * <size_t>self._dc.super.mt
        bsiz = <size_t>self._dc.super.mb * <size_t>self._dc.super.nb
        with nogil:
            typesize = parsec_datadist_getsizeoftype(mat_type)
        total_size = nb_local_tiles * bsiz * typesize
        
        with nogil:
            self._dc.mat = <char*>parsec_data_allocate(total_size)
        
        if self._dc.mat == NULL:
            free(self._dc)
            self._dc = NULL
            raise MemoryError("Failed to allocate matrix data")
    
    def __dealloc__(self):
        # Simplified cleanup: just free memory buffers, not PaRSEC structures
        # (parsec context may already be destroyed at this point)
        if self._dc != NULL:
            if self._dc.mat != NULL:
                with nogil:
                    parsec_data_free(self._dc.mat)
                self._dc.mat = NULL
            free(self._dc)
            self._dc = NULL
    
    def as_capsule(self):
        """Return PyCapsule for passing to C API"""
        return PyCapsule_New(<void*>self._dc, b"parsec_matrix_block_cyclic_t", NULL)
    
    @property
    def mt(self): return self._dc.super.mt
    @property
    def nt(self): return self._dc.super.nt
    @property
    def mb(self): return self._dc.super.mb
    @property
    def nb(self): return self._dc.super.nb
    @property
    def m(self): return self._dc.super.m
    @property
    def n(self): return self._dc.super.n
