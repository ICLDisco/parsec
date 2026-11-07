# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# C declarations of the PaRSEC API used by the bindings.

from libc.stdint cimport uint32_t, uint64_t


cdef extern from "parsec.h" nogil:
    ctypedef struct parsec_context_t:
        pass
    ctypedef struct parsec_taskpool_t:
        pass
    ctypedef struct parsec_task_t:
        pass
    ctypedef struct parsec_task_class_t:
        pass
    ctypedef struct parsec_execution_stream_t:
        pass

    ctypedef enum parsec_hook_return_t:
        PARSEC_HOOK_RETURN_DONE
        PARSEC_HOOK_RETURN_AGAIN
        PARSEC_HOOK_RETURN_NEXT
        PARSEC_HOOK_RETURN_DISABLE
        PARSEC_HOOK_RETURN_ASYNC
        PARSEC_HOOK_RETURN_ERROR

    ctypedef enum parsec_context_query_cmd_t:
        PARSEC_CONTEXT_QUERY_NODES
        PARSEC_CONTEXT_QUERY_RANK
        PARSEC_CONTEXT_QUERY_DEVICES
        PARSEC_CONTEXT_QUERY_DEVICES_FULL_PEER_ACCESS
        PARSEC_CONTEXT_QUERY_CORES
        PARSEC_CONTEXT_QUERY_ACTIVE_TASKPOOLS

    enum:
        PARSEC_SUCCESS

    parsec_context_t *parsec_init(int nb_cores, int *pargc, char ***pargv)
    int parsec_fini(parsec_context_t **pcontext)
    int parsec_version(int *major, int *minor, int *release)
    int parsec_context_start(parsec_context_t *context)
    int parsec_context_test(parsec_context_t *context)
    int parsec_context_wait(parsec_context_t *context)
    int parsec_context_add_taskpool(parsec_context_t *context, parsec_taskpool_t *tp)
    int parsec_context_query(parsec_context_t *context, parsec_context_query_cmd_t cmd, ...)
    int parsec_taskpool_wait(parsec_taskpool_t *tp)
    int parsec_taskpool_test(parsec_taskpool_t *tp)
    void parsec_taskpool_free(parsec_taskpool_t *tp)


cdef extern from "parsec/mca/device/device.h" nogil:
    enum:
        PARSEC_DEV_CPU
        PARSEC_DEV_CUDA
        PARSEC_DEV_HIP
        PARSEC_DEV_LEVEL_ZERO
        PARSEC_DEV_ALL


cdef extern from "parsec/data_distribution.h" nogil:
    ctypedef uint64_t parsec_data_key_t
    ctypedef struct parsec_data_collection_t:
        uint32_t myrank
        uint32_t nodes
    void parsec_data_collection_set_key(parsec_data_collection_t *dc, const char *name)
    void parsec_data_collection_destroy(parsec_data_collection_t *dc)


cdef extern from "parsec/data_dist/matrix/matrix.h" nogil:
    ctypedef enum parsec_matrix_type_t:
        PARSEC_MATRIX_BYTE
        PARSEC_MATRIX_INTEGER
        PARSEC_MATRIX_FLOAT
        PARSEC_MATRIX_DOUBLE
        PARSEC_MATRIX_COMPLEX_FLOAT
        PARSEC_MATRIX_COMPLEX_DOUBLE

    ctypedef enum parsec_matrix_storage_t:
        PARSEC_MATRIX_LAPACK
        PARSEC_MATRIX_TILE

    ctypedef struct parsec_tiled_matrix_t:
        parsec_data_collection_t super
        parsec_matrix_type_t mtype
        parsec_matrix_storage_t storage
        int mb
        int nb
        size_t bsiz
        int lm
        int ln
        int lmt
        int lnt
        int i
        int j
        int m
        int n
        int mt
        int nt
        int nb_local_tiles

    void parsec_tiled_matrix_destroy(parsec_tiled_matrix_t *tdesc)

    parsec_taskpool_t *parsec_redistribute_New(parsec_tiled_matrix_t *source,
                                               parsec_tiled_matrix_t *target,
                                               int size_row, int size_col,
                                               int disi_source, int disj_source,
                                               int disi_target, int disj_target)
    int parsec_redistribute_dtd(parsec_context_t *parsec,
                                parsec_tiled_matrix_t *source,
                                parsec_tiled_matrix_t *target,
                                int size_row, int size_col,
                                int disi_source, int disj_source,
                                int disi_target, int disj_target)


cdef extern from "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h" nogil:
    ctypedef struct parsec_matrix_block_cyclic_t:
        parsec_tiled_matrix_t super
        void *mat

    void parsec_matrix_block_cyclic_init(parsec_matrix_block_cyclic_t *desc,
                                         parsec_matrix_type_t mtype,
                                         parsec_matrix_storage_t storage,
                                         int myrank,
                                         int mb, int nb,
                                         int lm, int ln,
                                         int i, int j,
                                         int m, int n,
                                         int p, int q,
                                         int kp, int kq,
                                         int ip, int jq)


cdef extern from "parsec/data.h" nogil:
    void *parsec_data_allocate(size_t size)
    void parsec_data_free(void *ptr)


cdef extern from "parsec/interfaces/dtd/insert_function.h" nogil:
    ctypedef struct parsec_dtd_tile_t:
        pass

    enum:
        PARSEC_INPUT
        PARSEC_OUTPUT
        PARSEC_INOUT
        PARSEC_VALUE
        PARSEC_GET_OP_TYPE
        PARSEC_AFFINITY
        PARSEC_DONT_TRACK
        PARSEC_PUSHOUT
        PARSEC_PULLIN
        PARSEC_GET_REGION_INFO
        PASSED_BY_REF
        PARSEC_DTD_MAX_PARAMS

    parsec_taskpool_t *parsec_dtd_taskpool_new()
    parsec_dtd_tile_t *parsec_dtd_tile_of(parsec_data_collection_t *dc, parsec_data_key_t key)
    void parsec_dtd_data_collection_init(parsec_data_collection_t *dc)
    void parsec_dtd_data_collection_fini(parsec_data_collection_t *dc)
    int parsec_dtd_data_flush(parsec_taskpool_t *tp, parsec_dtd_tile_t *tile)
    int parsec_dtd_data_flush_all(parsec_taskpool_t *tp, parsec_data_collection_t *dc)
    int parsec_dtd_task_class_add_chore(parsec_taskpool_t *tp, parsec_task_class_t *tc,
                                        int device_type, void *function)
    void parsec_dtd_task_class_release(parsec_taskpool_t *tp, parsec_task_class_t *tc)
    int parsec_dtd_free_arena_datatype(parsec_context_t *ctx, int id)


cdef extern from "pyparsec.h" nogil:
    void pyparsec_dtd_insert(parsec_taskpool_t *tp, parsec_task_class_t *tc,
                             int priority, int device_type,
                             int nb_args, const int *args_flags, void * const *args_ptrs)
    void pyparsec_dtd_unpack(parsec_task_t *task, int nb_args, void * const *args_out)
    parsec_task_class_t *pyparsec_dtd_create_task_class(const char *name, int nb_params,
                                                        const int *sizes, const int *ops)
    const parsec_task_class_t *pyparsec_task_class_of(const parsec_task_t *task)
    int pyparsec_context_is_active(const parsec_context_t *context)
    int pyparsec_dtd_max_flows()
    parsec_data_key_t pyparsec_data_key(parsec_data_collection_t *dc, int m, int n)
    int pyparsec_rank_of(parsec_data_collection_t *dc, int m, int n)
    void *pyparsec_local_tile(parsec_data_collection_t *dc, int m, int n)
    size_t pyparsec_sizeof_matrix_type(int mtype)
    int pyparsec_dtd_tile_arena_new(parsec_context_t *ctx, int mtype, int mb, int nb, int ld)
