# cython: language_level=3
"""
Real wrapper around PaRSEC's parsec_matrix_block_cyclic_t (2D block-cyclic tiled matrix).
"""
from cpython.pycapsule cimport PyCapsule_New

cdef extern from "parsec.h":
    ctypedef struct parsec_data_collection_s:
        pass
    ctypedef parsec_data_collection_s* parsec_data_collection_t
    void* parsec_data_allocate(size_t size)
    void  parsec_data_free(void* ptr)
    void  parsec_data_collection_set_key(parsec_data_collection_t* dc, const char* key)
    void  parsec_data_collection_destroy(parsec_data_collection_t* dc)

cdef extern from *:
    r"""
    #include "parsec.h"
    #include "parsec/interfaces/dtd/insert_function_internal.h"
    #include "parsec/data_dist/matrix/matrix.h"
    #include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
    #include <stdlib.h>
    #include <mpi.h>

    /* Access the global DTD tile mempool so we can check initialization state
     * and avoid calling DTD init functions when the mempool isn't ready. */
    extern parsec_mempool_t *parsec_dtd_tile_mempool;

    static inline int py_has_parsec_dtd_tile_mempool(void) {
        return (NULL != parsec_dtd_tile_mempool);
    }

    enum {
        CY_PARSEC_MATRIX_DOUBLE = PARSEC_MATRIX_DOUBLE,
        CY_PARSEC_MATRIX_TILE   = PARSEC_MATRIX_TILE
    };

    static inline size_t cy_sizeof_type(int mtype)
    {
        return (size_t)parsec_datadist_getsizeoftype(mtype);
    }

    static inline parsec_matrix_block_cyclic_t* cy_mat_create(int mtype, int storage,
                                                              int myrank,
                                                              int mb, int nb,
                                                              int lm, int ln,
                                                              int i, int j,
                                                              int m, int n,
                                                              int P, int Q,
                                                              int kp, int kq,
                                                              int ip, int jq,
                                                              const char* keyname)
    {
        parsec_matrix_block_cyclic_t* dc = (parsec_matrix_block_cyclic_t*)calloc(1, sizeof(parsec_matrix_block_cyclic_t));
        if(NULL == dc) return NULL;

        /* Ensure MPI is initialized before calling MPI_Type_size from PaRSEC
         * internals. If MPI isn't initialized, bail out early to avoid the
         * MPI runtime aborting the process. */
        int _mpi_init_flag = 0;
        MPI_Initialized(&_mpi_init_flag);
        if( !_mpi_init_flag ) {
            free(dc);
            return NULL;
        }

        parsec_matrix_block_cyclic_init(dc,
                                        mtype, storage, myrank,
                                        mb, nb,
                                        lm, ln, i, j, m, n,
                                        P, Q, kp, kq, ip, jq);

        parsec_data_collection_t* A = &dc->super.super;
        parsec_data_collection_set_key(A, keyname);

        size_t bsiz = (size_t)dc->super.bsiz;
        size_t nlt  = (size_t)dc->super.nb_local_tiles;
        size_t eltsz = cy_sizeof_type(dc->super.mtype);
        dc->mat = (char*)parsec_data_allocate(nlt * bsiz * eltsz);

        /* Defensive: ensure global DTD mempool initialized */
        if( NULL == parsec_dtd_tile_mempool ) {
            if(dc->mat) parsec_data_free(dc->mat);
            free(dc);
            return NULL;
        }

        parsec_dtd_data_collection_init(A);
        return dc;
    }

    static inline void cy_mat_destroy(parsec_matrix_block_cyclic_t* dc)
    {
        if(NULL == dc) return;
        parsec_data_collection_t* A = &dc->super.super;

        parsec_dtd_data_collection_fini(A);

        if(dc->mat) parsec_data_free(dc->mat);
        parsec_tiled_matrix_destroy_data(&dc->super);
        parsec_data_collection_destroy(&dc->super.super);
        free(dc);
    }

    static inline int cy_mat_mt(parsec_matrix_block_cyclic_t* dc) { return dc->super.mt; }
    static inline int cy_mat_nt(parsec_matrix_block_cyclic_t* dc) { return dc->super.nt; }
    static inline int cy_mat_mb(parsec_matrix_block_cyclic_t* dc) { return dc->super.mb; }
    static inline int cy_mat_nb(parsec_matrix_block_cyclic_t* dc) { return dc->super.nb; }
    static inline int cy_mat_m(parsec_matrix_block_cyclic_t* dc)  { return dc->super.m;  }
    static inline int cy_mat_n(parsec_matrix_block_cyclic_t* dc)  { return dc->super.n;  }
    """
    enum:
        CY_PARSEC_MATRIX_DOUBLE
        CY_PARSEC_MATRIX_TILE

    ctypedef struct parsec_matrix_block_cyclic_t
    parsec_matrix_block_cyclic_t* cy_mat_create(int mtype, int storage,
                                                int myrank,
                                                int mb, int nb,
                                                int lm, int ln,
                                                int i, int j,
                                                int m, int n,
                                                int P, int Q,
                                                int kp, int kq,
                                                int ip, int jq,
                                                const char* keyname)
    int py_has_parsec_dtd_tile_mempool()
    void cy_mat_destroy(parsec_matrix_block_cyclic_t* dc)
    int cy_mat_mt(parsec_matrix_block_cyclic_t* dc)
    int cy_mat_nt(parsec_matrix_block_cyclic_t* dc)
    int cy_mat_mb(parsec_matrix_block_cyclic_t* dc)
    int cy_mat_nb(parsec_matrix_block_cyclic_t* dc)
    int cy_mat_m(parsec_matrix_block_cyclic_t* dc)
    int cy_mat_n(parsec_matrix_block_cyclic_t* dc)

PARSEC_MATRIX_DOUBLE = CY_PARSEC_MATRIX_DOUBLE
PARSEC_MATRIX_TILE   = CY_PARSEC_MATRIX_TILE


cdef class ParsecMatrixBlockCyclic:
    """
    Owner wrapper for parsec_matrix_block_cyclic_t*.
    Allocates local storage and runs parsec_dtd_data_collection_init().
    """
    cdef parsec_matrix_block_cyclic_t* _dc
    cdef bytes _key

    def __cinit__(self,
                  key: str,
                  int myrank,
                  int mb, int nb,
                  int lm, int ln,
                  int i, int j,
                  int m, int n,
                  int P, int Q,
                  int kp=1, int kq=1,
                  int ip=0, int jq=0,
                  int mtype=PARSEC_MATRIX_DOUBLE,
                  int storage=PARSEC_MATRIX_TILE):
        self._dc = NULL
        self._key = key.encode("utf-8")

        # Check MPI initialization early to avoid an MPI abort
        try:
            import mpi4py.MPI as _MPI
            if not _MPI.Is_initialized():
                raise RuntimeError("MPI is not initialized; call MPI.Init() before creating Parsec matrices")
        except Exception as _e:
            # If mpi4py import failed, fall through and let cy_mat_create handle it
            pass

        # Check that PaRSEC's DTD mempool was initialized by creating a DTD taskpool
        if not py_has_parsec_dtd_tile_mempool():
            raise RuntimeError("PaRSEC DTD tile mempool is not initialized. Create and start a DTD taskpool before creating matrices.")

        self._dc = cy_mat_create(mtype, storage, myrank,
                                 mb, nb,
                                 lm, ln, i, j, m, n,
                                 P, Q, kp, kq, ip, jq,
                                 self._key)
        if self._dc == NULL:
            raise MemoryError("Failed to allocate parsec_matrix_block_cyclic_t")

    def __dealloc__(self):
        if self._dc != NULL:
            cy_mat_destroy(self._dc)
            self._dc = NULL

    def as_capsule(self):
        return PyCapsule_New(<void*>self._dc, b"parsec_matrix_block_cyclic_t", NULL)

    @property
    def mt(self): return cy_mat_mt(self._dc)
    @property
    def nt(self): return cy_mat_nt(self._dc)
    @property
    def mb(self): return cy_mat_mb(self._dc)
    @property
    def nb(self): return cy_mat_nb(self._dc)
    @property
    def m(self):  return cy_mat_m(self._dc)
    @property
    def n(self):  return cy_mat_n(self._dc)

    def __repr__(self):
        return (f"<ParsecMatrixBlockCyclic key={self._key!r} "
                f"m={self.m} n={self.n} mb={self.mb} nb={self.nb} "
                f"mt={self.mt} nt={self.nt}>")
