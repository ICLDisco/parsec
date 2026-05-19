/*
 * Copyright (c) 2017-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "redistribute_test.h"
#include "common.h"

#ifdef PARSEC_HAVE_MPI
#include <mpi.h>
#endif

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include "parsec/mca/device/cuda/device_cuda.h"
#endif  /* defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) */

typedef struct testing_redistribute_matrix_s {
    parsec_tiled_matrix_t *desc;
    void **mat;
    int distribution;
    int memory_location;
    parsec_matrix_uplo_t uplo;
} testing_redistribute_matrix_t;

static int testing_redistribute_imax(int a, int b)
{
    return (a >= b) ? a : b;
}

static int testing_redistribute_sbc_nodes_match(int nodes, int r)
{
    return (nodes == (r * (r - 1)) / 2) ||
           (((r % 2) == 0) && (nodes == (r * r) / 2));
}

static int testing_redistribute_sbc_infer_r(int rows, int cols, int nodes)
{
    int preferred = testing_redistribute_imax(rows, cols);

    if( testing_redistribute_sbc_nodes_match(nodes, preferred) ) {
        return preferred;
    }

    for(int r = 2; r < 65536; r++) {
        int extended_nodes = (r * (r - 1)) / 2;
        int basic_nodes = (r * r) / 2;

        if( testing_redistribute_sbc_nodes_match(nodes, r) ) {
            return r;
        }
        if( (extended_nodes > nodes) && (basic_nodes > nodes) ) {
            break;
        }
    }

    return preferred;
}

static int testing_redistribute_region_fits_uplo(int mb, int nb,
                                                 int size_row, int size_col,
                                                 int disi, int disj,
                                                 parsec_matrix_uplo_t uplo)
{
    int m_start = disi / mb;
    int m_end = (disi + size_row - 1) / mb;
    int n_start = disj / nb;
    int n_end = (disj + size_col - 1) / nb;

    if( PARSEC_MATRIX_LOWER == uplo ) {
        return m_start >= n_end;
    }
    if( PARSEC_MATRIX_UPPER == uplo ) {
        return n_start >= m_end;
    }

    return 0;
}

static parsec_matrix_uplo_t
testing_redistribute_select_sbc_uplo(int mb, int nb,
                                     int size_row, int size_col,
                                     int disi, int disj)
{
    if( testing_redistribute_region_fits_uplo(mb, nb, size_row, size_col,
                                              disi, disj, PARSEC_MATRIX_LOWER) ) {
        return PARSEC_MATRIX_LOWER;
    }
    if( testing_redistribute_region_fits_uplo(mb, nb, size_row, size_col,
                                              disi, disj, PARSEC_MATRIX_UPPER) ) {
        return PARSEC_MATRIX_UPPER;
    }

    return PARSEC_MATRIX_LOWER;
}

static int
testing_redistribute_validate_sbc_region(testing_redistribute_matrix_t *matrix,
                                         const char *name,
                                         int size_row, int size_col,
                                         int disi, int disj)
{
    if( REDISTRIBUTE_DIST_SBC != matrix->distribution ) {
        return PARSEC_SUCCESS;
    }

    if( testing_redistribute_region_fits_uplo(matrix->desc->mb, matrix->desc->nb,
                                              size_row, size_col, disi, disj,
                                              matrix->uplo) ) {
        return PARSEC_SUCCESS;
    }

    if( 0 == matrix->desc->super.myrank ) {
        fprintf(stderr,
                "ERROR: %s SBC descriptor stores the %s triangle, but the requested "
                "rectangle crosses the tile diagonal. Move the displacement or shrink "
                "the submatrix so every referenced tile is stored.\n",
                name, (PARSEC_MATRIX_LOWER == matrix->uplo) ? "lower" : "upper");
    }

    return PARSEC_ERR_BAD_PARAM;
}

static parsec_matrix_uplo_t
testing_redistribute_apply_uplo(const testing_redistribute_matrix_t *matrix)
{
    return (REDISTRIBUTE_DIST_SBC == matrix->distribution) ?
           matrix->uplo : PARSEC_MATRIX_FULL;
}

static int
testing_redistribute_uses_cuda_memory(int memory_location)
{
    return (REDISTRIBUTE_MEMORY_MANAGED == memory_location) ||
           (REDISTRIBUTE_MEMORY_DEVICE == memory_location);
}

static int
testing_redistribute_check_cuda_devices(parsec_context_t *parsec,
                                        int source_memory,
                                        int target_memory,
                                        int rank)
{
    if( !testing_redistribute_uses_cuda_memory(source_memory) &&
        !testing_redistribute_uses_cuda_memory(target_memory) ) {
        return PARSEC_SUCCESS;
    }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    int nb_gpus = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_DEVICES, PARSEC_DEV_CUDA);
    int local_has_device = (nb_gpus > 0);
    int all_have_device = local_has_device;

#ifdef PARSEC_HAVE_MPI
    MPI_Allreduce(&local_has_device, &all_have_device, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

    if( nb_gpus < 0 || !all_have_device ) {
        if( 0 == rank ) {
            parsec_warning("This test can only run if every participating rank has at least one CUDA device");
            printf("TEST SKIPPED\n");
        }
        return PARSEC_ERR_DEVICE;
    }

    return PARSEC_SUCCESS;
#else
    if( 0 == rank ) {
        parsec_warning("This test requires CUDA device support");
        printf("TEST SKIPPED\n");
    }
    (void)parsec;
    return PARSEC_ERR_DEVICE;
#endif  /* defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) */
}

static size_t
testing_redistribute_storage_size(const parsec_tiled_matrix_t *desc)
{
    return (size_t)desc->nb_local_tiles *
           (size_t)desc->bsiz *
           (size_t)parsec_datadist_getsizeoftype(desc->mtype);
}

static int
testing_redistribute_allocate_storage(testing_redistribute_matrix_t *matrix)
{
    size_t bytes = testing_redistribute_storage_size(matrix->desc);

    *matrix->mat = NULL;
    if( 0 == bytes ) {
        return PARSEC_SUCCESS;
    }

    if( REDISTRIBUTE_MEMORY_MANAGED == matrix->memory_location ) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        cudaError_t status = cudaMallocManaged(matrix->mat, bytes, cudaMemAttachGlobal);
        if( cudaSuccess != status ) {
            fprintf(stderr, "ERROR: cudaMallocManaged(%zu) failed: %s\n",
                    bytes, cudaGetErrorString(status));
            if( cudaErrorNoDevice == status ) {
                return PARSEC_ERR_DEVICE;
            }
            return PARSEC_ERROR;
        }
#else
        fprintf(stderr, "ERROR: managed memory requires CUDA support\n");
        return PARSEC_ERR_DEVICE;
#endif
    } else {
        *matrix->mat = parsec_data_allocate(bytes);
        if( NULL == *matrix->mat ) {
            fprintf(stderr, "ERROR: could not allocate %zu bytes for %s memory\n",
                    bytes, redistribute_memory_location_name(matrix->memory_location));
            return PARSEC_ERROR;
        }
    }

    return PARSEC_SUCCESS;
}

static void
testing_redistribute_free_storage(testing_redistribute_matrix_t *matrix)
{
    if( (NULL == matrix->mat) || (NULL == *matrix->mat) ) {
        return;
    }

    if( REDISTRIBUTE_MEMORY_MANAGED == matrix->memory_location ) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        (void)cudaFree(*matrix->mat);
#endif
    } else {
        parsec_data_free(*matrix->mat);
    }
    *matrix->mat = NULL;
}

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
static void
testing_redistribute_cuda_release(parsec_data_copy_t *copy, int device)
{
    (void)device;
    if( NULL != copy->device_private ) {
        (void)cudaFree(copy->device_private);
        copy->device_private = NULL;
    }
}

static parsec_device_cuda_module_t *
testing_redistribute_get_cuda_device(void)
{
    for(int dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_cuda_module_t *cuda_device =
            (parsec_device_cuda_module_t *)parsec_mca_device_get(dev);

        if( (NULL != cuda_device) &&
            (cuda_device->super.super.type & PARSEC_DEV_CUDA) ) {
            return cuda_device;
        }
    }

    return NULL;
}
#endif  /* defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) */

static int
testing_redistribute_make_device_resident(testing_redistribute_matrix_t *matrix)
{
    if( REDISTRIBUTE_MEMORY_DEVICE != matrix->memory_location ) {
        return PARSEC_SUCCESS;
    }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    parsec_tiled_matrix_t *desc = matrix->desc;
    parsec_data_collection_t *dc = &desc->super;
    parsec_device_cuda_module_t *cuda_device = testing_redistribute_get_cuda_device();
    uint8_t device_index;
    cudaError_t status;

    if( NULL == cuda_device ) {
        fprintf(stderr, "ERROR: device memory was requested but no CUDA device is available\n");
        return PARSEC_ERR_DEVICE;
    }

    device_index = cuda_device->super.super.device_index;
    status = cudaSetDevice(cuda_device->cuda_index);
    if( cudaSuccess != status ) {
        fprintf(stderr, "ERROR: cudaSetDevice(%d) failed: %s\n",
                cuda_device->cuda_index, cudaGetErrorString(status));
        return PARSEC_ERROR;
    }

    for(int n = 0; n < desc->nt; n++) {
        for(int m = 0; m < desc->mt; m++) {
            parsec_data_t *data;
            parsec_data_copy_t *cpu_copy;
            parsec_data_copy_t *gpu_copy;
            uint32_t owner = dc->rank_of(dc, m, n);

            if( owner != dc->myrank ) {
                continue;
            }

            data = dc->data_of(dc, m, n);
            cpu_copy = parsec_data_get_copy(data, 0);
            gpu_copy = parsec_data_get_copy(data, device_index);

            if( NULL == gpu_copy ) {
                gpu_copy = PARSEC_OBJ_NEW(parsec_data_copy_t);
                gpu_copy->dtt = dc->default_dtt;
                gpu_copy->release_cb = testing_redistribute_cuda_release;
                status = cudaMalloc(&gpu_copy->device_private, data->nb_elts);
                if( cudaSuccess != status ) {
                    PARSEC_OBJ_RELEASE(gpu_copy);
                    fprintf(stderr, "ERROR: cudaMalloc(%zu) failed: %s\n",
                            data->nb_elts, cudaGetErrorString(status));
                    return PARSEC_ERROR;
                }
                parsec_data_copy_attach(data, gpu_copy, device_index);
            }

            status = cudaMemcpy(gpu_copy->device_private,
                                cpu_copy->device_private,
                                data->nb_elts,
                                cudaMemcpyHostToDevice);
            if( cudaSuccess != status ) {
                fprintf(stderr, "ERROR: cudaMemcpy(host to device, %zu) failed: %s\n",
                        data->nb_elts, cudaGetErrorString(status));
                return PARSEC_ERROR;
            }
            parsec_data_transfer_ownership_to_copy(data, device_index, PARSEC_FLOW_ACCESS_RW);
        }
    }

    status = cudaDeviceSynchronize();
    if( cudaSuccess != status ) {
        fprintf(stderr, "ERROR: cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(status));
        return PARSEC_ERROR;
    }

    return PARSEC_SUCCESS;
#else
    fprintf(stderr, "ERROR: device memory requires CUDA support\n");
    return PARSEC_ERR_DEVICE;
#endif
}

static int
testing_redistribute_init_matrix(testing_redistribute_matrix_t *matrix,
                                 parsec_matrix_block_cyclic_t *bc,
                                 parsec_matrix_sbc_t *sbc,
                                 const char *name,
                                 int distribution,
                                 int memory_location,
                                 int rank, int nodes,
                                 int grid_rows, int grid_cols,
                                 int supertile_rows, int supertile_cols,
                                 int mb, int nb, int lm, int ln,
                                 int size_row, int size_col,
                                 int disi, int disj)
{
    matrix->distribution = distribution;
    matrix->memory_location = memory_location;
    matrix->uplo = PARSEC_MATRIX_FULL;

    if( REDISTRIBUTE_DIST_2DBC == distribution ) {
        parsec_matrix_block_cyclic_init(bc, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                        rank, mb, nb, lm, ln, 0, 0,
                                        lm, ln, grid_rows, nodes / grid_rows,
                                        supertile_rows, supertile_cols, 0, 0);
        matrix->desc = (parsec_tiled_matrix_t *)bc;
        matrix->mat = &bc->mat;
    } else if( REDISTRIBUTE_DIST_SBC == distribution ) {
        int r = testing_redistribute_sbc_infer_r(grid_rows, grid_cols, nodes);
        int rc;

        matrix->uplo = testing_redistribute_select_sbc_uplo(mb, nb, size_row, size_col,
                                                            disi, disj);
        rc = parsec_matrix_sbc_init(sbc, PARSEC_MATRIX_DOUBLE,
                                    rank, mb, nb, lm, ln, 0, 0, lm, ln,
                                    nodes, r, matrix->uplo);
        if( PARSEC_SUCCESS != rc ) {
            if( 0 == rank ) {
                fprintf(stderr,
                        "ERROR: could not initialize %s SBC descriptor with r=%d "
                        "on %d ranks. SBC requires nodes = r*(r-1)/2 or, for even r, r*r/2.\n",
                        name, r, nodes);
            }
            return rc;
        }
        matrix->desc = (parsec_tiled_matrix_t *)sbc;
        matrix->mat = &sbc->mat;
    } else {
        fprintf(stderr, "ERROR: unsupported distribution '%s'\n",
                redistribute_distribution_name(distribution));
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    if( PARSEC_SUCCESS != testing_redistribute_allocate_storage(matrix) ) {
        return PARSEC_ERROR;
    }
    parsec_data_collection_set_key((parsec_data_collection_t *)matrix->desc, name);

    return testing_redistribute_validate_sbc_region(matrix, name, size_row, size_col,
                                                   disi, disj);
}

/**
 * @brief Test example of redistribute
 *
 * @detail
 * parsec_redistribute: PTG, redistribute from ANY distribution
 * to ANY distribution, with ANY displacement
 *
 * parsec_redistribute_dtd: DTD, redistribute from ANY distribution
 * to ANY distribution, with ANY displacement
 *
 * parsec_redistribute_check: check the result correctness of
 * two submatrix, if correct, print "Redistribute Result is CORRECT";
 * otherwise print the first detected location and values where values
 * are different.
 *
 * parsec_redistribute_init: init dcY to 0 or numbers based on index
 *
 * @example testing_redistribute.c
 */
int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    double dparam[IPARAM_SIZEOF];
    int MMB, NNB, MMBR, NNBR;
    double time_ptg = 0.0, time_dtd = 0.0;

    /* Source */
    iparam[IPARAM_P] = 1;
    iparam[IPARAM_Q] = 1;
    iparam[IPARAM_SOURCE_DIST] = REDISTRIBUTE_DIST_2DBC;
    iparam[IPARAM_SOURCE_MEMORY] = REDISTRIBUTE_MEMORY_HOST;
    iparam[IPARAM_M] = 4;
    iparam[IPARAM_N] = 4;
    iparam[IPARAM_MB] = 4;
    iparam[IPARAM_NB] = 4;
    iparam[IPARAM_DISI] = 0;
    iparam[IPARAM_DISJ] = 0;

    /* Target/redistribute */
    iparam[IPARAM_P_R] = 1;
    iparam[IPARAM_Q_R] = 1;
    iparam[IPARAM_TARGET_DIST] = REDISTRIBUTE_DIST_2DBC;
    iparam[IPARAM_TARGET_MEMORY] = REDISTRIBUTE_MEMORY_HOST;
    iparam[IPARAM_M_R] = 4;
    iparam[IPARAM_N_R] = 4;
    iparam[IPARAM_MB_R] = 4;
    iparam[IPARAM_NB_R] = 4;
    iparam[IPARAM_DISI_R] = 0;
    iparam[IPARAM_DISJ_R] = 0;

    /* Matrix common */
    iparam[IPARAM_RADIUS] = 0;
    iparam[IPARAM_M_SUB] = 4;
    iparam[IPARAM_N_SUB] = 4;

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam, dparam);

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];

    /* Source */
    int P       = iparam[IPARAM_P];
    int Q       = iparam[IPARAM_Q];
    int source_distribution = iparam[IPARAM_SOURCE_DIST];
    int source_memory = iparam[IPARAM_SOURCE_MEMORY];
    int M       = iparam[IPARAM_M];
    int N       = iparam[IPARAM_N];
    int MB      = iparam[IPARAM_MB];
    int NB      = iparam[IPARAM_NB];
    int SMB     = iparam[IPARAM_SMB];
    int SNB     = iparam[IPARAM_SNB];
    int disi_Y  = iparam[IPARAM_DISI];
    int disj_Y  = iparam[IPARAM_DISJ];

    /* Target/redistribute */
    int PR       = iparam[IPARAM_P_R];
    int QR       = iparam[IPARAM_Q_R];
    int target_distribution = iparam[IPARAM_TARGET_DIST];
    int target_memory = iparam[IPARAM_TARGET_MEMORY];
    int MR       = iparam[IPARAM_M_R];
    int NR       = iparam[IPARAM_N_R];
    int MBR      = iparam[IPARAM_MB_R];
    int NBR      = iparam[IPARAM_NB_R];
    int SMBR     = iparam[IPARAM_SMB_R];
    int SNBR     = iparam[IPARAM_SNB_R];
    int disi_T  = iparam[IPARAM_DISI_R];
    int disj_T  = iparam[IPARAM_DISJ_R];

    /* Matrix common */
    int R = iparam[IPARAM_RADIUS];
    int size_row = iparam[IPARAM_M_SUB];
    int size_col = iparam[IPARAM_N_SUB];

    /* Others */
    int check = iparam[IPARAM_CHECK];
    int time = iparam[IPARAM_GETTIME];
    double network_bandwidth  = dparam[DPARAM_NETWORK_BANDWIDTH];
    double memcpy_bandwidth  = dparam[DPARAM_MEMCPY_BANDWIDTH];
    int num_runs = iparam[IPARAM_NUM_RUNS];
    int thread_type = iparam[IPARAM_THREAD_MULTIPLE];
    int no_optimization_version = iparam[IPARAM_NO_OPTIMIZATION_VERSION];

    /* Used for ghost region */
    MMB = (int)(ceil((double)M/MB));
    NNB = (int)(ceil((double)N/NB));
    MMBR = (int)(ceil((double)MR/MBR));
    NNBR = (int)(ceil((double)NR/NBR));

    double *results = NULL;
    int rc = PARSEC_SUCCESS;

    /* Initializing matrix structure */
    parsec_matrix_block_cyclic_t dcY_2dbc, dcT_2dbc;
    parsec_matrix_sbc_t dcY_sbc, dcT_sbc;
    testing_redistribute_matrix_t dcY = {0};
    testing_redistribute_matrix_t dcT = {0};

    rc = testing_redistribute_check_cuda_devices(parsec, source_memory, target_memory, rank);
    if( PARSEC_SUCCESS != rc ) {
        goto cleanup_all;
    }

    if( no_optimization_version &&
        ((REDISTRIBUTE_DIST_SBC == source_distribution) ||
         (REDISTRIBUTE_DIST_SBC == target_distribution)) ) {
        if( 0 == rank ) {
            fprintf(stderr,
                    "ERROR: the no-optimization PTG path is not supported with SBC descriptors\n");
        }
        rc = PARSEC_ERR_NOT_SUPPORTED;
        goto cleanup_all;
    }

    rc = testing_redistribute_init_matrix(&dcY, &dcY_2dbc, &dcY_sbc, "dcY",
                                          source_distribution, source_memory,
                                          rank, nodes, P, Q, SMB, SNB,
                                          MB + 2 * R, NB + 2 * R,
                                          M + 2 * R * MMB, N + 2 * R * NNB,
                                          size_row, size_col, disi_Y, disj_Y);
    if( PARSEC_SUCCESS != rc ) {
        goto cleanup_all;
    }

    rc = testing_redistribute_init_matrix(&dcT, &dcT_2dbc, &dcT_sbc, "dcT",
                                          target_distribution, target_memory,
                                          rank, nodes, PR, QR, SMBR, SNBR,
                                          MBR + 2 * R, NBR + 2 * R,
                                          MR + 2 * R * MMBR, NR + 2 * R * NNBR,
                                          size_row, size_col, disi_T, disj_T);
    if( PARSEC_SUCCESS != rc ) {
        goto cleanup_all;
    }

    for(int i = 0; i < num_runs; i++) {
#if RUN_PTG
         /*
         * Init dcY not including ghost region; if initvalue is 0,
         * init to 0; otherwise init to numbers based on index
         */
        int op_args = 1;
        parsec_apply( parsec, testing_redistribute_apply_uplo(&dcY),
                      dcY.desc,
                      (parsec_tiled_matrix_unary_op_t)redistribute_init_ops, &op_args);
        rc = testing_redistribute_make_device_resident(&dcY);
        if( PARSEC_SUCCESS != rc ) {
            goto cleanup_all;
        }

        /* Timer start */
        SYNC_TIME_START();

        /* Main part, call parsec_redistribute; double is default, which could be
         * changed in parsec/data_dist/matrix/redistribute/redistribute_internal.h
         */
        if( no_optimization_version )
            parsec_redistribute_no_optimization(parsec, dcY.desc,
                                                dcT.desc,
                                                size_row, size_col, disi_Y, disj_Y,
                                                disi_T, disj_T);
        else
            parsec_redistribute(parsec, dcY.desc,
                                dcT.desc,
                                size_row, size_col, disi_Y, disj_Y,
                                disi_T, disj_T);
        rc = testing_redistribute_make_device_resident(&dcT);
        if( PARSEC_SUCCESS != rc ) {
            goto cleanup_all;
        }

        /* Timer end */
        if( time ) {
#if PRINT_MORE
            SYNC_TIME_PRINT(rank, ("\"testing_redistribute_PTG\""
                            "\tRedistributed Size: m= %d n= %d"
                            "\tSource: P= %d Q= %d M= %d N= %d MB= %d NB= %d I= %d J=%d SMB= %d SNB= %d"
                            "\tTarget: PR= %d QR= %d MR= %d NR= %d MBR= %d NBR= %d i= %d j=%d SMBR= %d SNBR= %d"
                            "\tCores: %d\n\n",
                            size_row, size_col, P, Q, M, N, MB, NB, disi_Y, disj_Y, SMB, SNB,
                            PR, QR, MR, NR, MBR, NBR, disi_T, disj_T, SMBR, SNBR, cores));
#else
            SYNC_TIME_STOP();
#endif
            time_ptg = sync_time_elapsed;
        }

        /* Check result */
        if( check ){
            if( 0 == rank )
                printf("Checking result for PTG:");

#if COPY_TO_1NODE
            parsec_redistribute_check(parsec, dcY.desc,
                                      dcT.desc,
                                      size_row, size_col, disi_Y, disj_Y,
                                      disi_T, disj_T);
#else
            /* Init dcY to 0 */
            int op_args = 0;
            parsec_apply( parsec, testing_redistribute_apply_uplo(&dcY),
                          dcY.desc,
                          (parsec_tiled_matrix_unary_op_t)redistribute_init_ops, &op_args);
            rc = testing_redistribute_make_device_resident(&dcY);
            if( PARSEC_SUCCESS != rc ) {
                goto cleanup_all;
            }

            /* Redistribute back from dcT to dcY */
            parsec_redistribute(parsec, dcT.desc,
                                dcY.desc,
                                size_row, size_col, disi_T, disj_T,
                                disi_Y, disj_Y);
            rc = testing_redistribute_make_device_resident(&dcY);
            if( PARSEC_SUCCESS != rc ) {
                goto cleanup_all;
            }

            parsec_redistribute_check2(parsec, dcY.desc,
                                       size_row, size_col, disi_Y, disj_Y);
#endif /* COPY_TO_1NODE */
        }
#endif /* RUN_PTG */

#if RUN_DTD
        /*
         * Init dcT to 0.0 for DTD
         */
        int op_args_dtd = 0;
        parsec_apply( parsec, testing_redistribute_apply_uplo(&dcT),
                      dcT.desc,
                      (parsec_tiled_matrix_unary_op_t)redistribute_init_ops, &op_args_dtd);
        rc = testing_redistribute_make_device_resident(&dcY);
        if( PARSEC_SUCCESS != rc ) {
            goto cleanup_all;
        }
        rc = testing_redistribute_make_device_resident(&dcT);
        if( PARSEC_SUCCESS != rc ) {
            goto cleanup_all;
        }

        /* Timer start */
        SYNC_TIME_START();

        /* Main part, call parsec_redistribute_dtd; double is default, which could be
         * changed in parsec/data_dist/matrix/redistribute/redistribute_internal.h
         */
        parsec_redistribute_dtd(parsec, dcY.desc,
                                dcT.desc,
                                size_row, size_col, disi_Y, disj_Y,
                                disi_T, disj_T);
        rc = testing_redistribute_make_device_resident(&dcT);
        if( PARSEC_SUCCESS != rc ) {
            goto cleanup_all;
        }

        /* Timer end */
        if( time ) {
#if PRINT_MORE
            SYNC_TIME_PRINT(rank, ("\"testing_redistribute_DTD\""
                            "\tRedistributed Size: m= %d n= %d"
                            "\tSource: P= %d Q= %d M= %d N= %d MB= %d NB= %d I= %d J=%d SMB= %d SNB= %d"
                            "\tTarget: PR= %d QR= %d MR= %d NR= %d MBR= %d NBR= %d i= %d j=%d SMBR= %d SNBR= %d"
                            "\tCores: %d\n\n",
                            size_row, size_col, P, Q, M, N, MB, NB, disi_Y, disj_Y, SMB, SNB,
                            PR, QR, MR, NR, MBR, NBR, disi_T, disj_T, SMBR, SNBR, cores));
#else
            SYNC_TIME_STOP();
#endif
            time_dtd = sync_time_elapsed;
        }

        /* Check result */
        if( check ){
            if( 0 == rank )
                printf("Checking result for DTD:");

#if COPY_TO_1NODE
            parsec_redistribute_check(parsec, dcY.desc,
                                      dcT.desc,
                                      size_row, size_col, disi_Y, disj_Y,
                                      disi_T, disj_T);
#else
            /* Init dcY to 0 */
            int op_args = 0;
            parsec_apply( parsec, testing_redistribute_apply_uplo(&dcY),
                          dcY.desc,
                          (parsec_tiled_matrix_unary_op_t)redistribute_init_ops, &op_args);
            rc = testing_redistribute_make_device_resident(&dcY);
            if( PARSEC_SUCCESS != rc ) {
                goto cleanup_all;
            }

            /* Redistribute back from dcT to dcY */
            parsec_redistribute_dtd(parsec, dcT.desc,
                                    dcY.desc,
                                    size_row, size_col, disi_T, disj_T,
                                    disi_Y, disj_Y);
            rc = testing_redistribute_make_device_resident(&dcY);
            if( PARSEC_SUCCESS != rc ) {
                goto cleanup_all;
            }

            parsec_redistribute_check2(parsec, dcY.desc,
                                       size_row, size_col, disi_Y, disj_Y);
#endif /* COPY_TO_1NODE */
        }
#endif /* RUN_DTD */

        if( time ) {
            /* Timer start */
            SYNC_TIME_START();

            /* Call parsec_redistribute_bound to get time bound */
            results = parsec_redistribute_bound(parsec, dcY.desc,
                                                dcT.desc,
                                                size_row, size_col, disi_Y, disj_Y,
                                                disi_T, disj_T);

            /* Timer end */
#if PRINT_MORE
            SYNC_TIME_PRINT(rank, ("\"testing_redistribute_bound\""
                            "\tRedistributed Size: m= %d n= %d"
                            "\tSource: P= %d Q= %d M= %d N= %d MB= %d NB= %d I= %d J=%d SMB= %d SNB= %d"
                            "\tTarget: PR= %d QR= %d MR= %d NR= %d MBR= %d NBR= %d i= %d j=%d SMBR= %d SNBR= %d"
                            "\tCores: %d\n\n",
                            size_row, size_col, P, Q, M, N, MB, NB, disi_Y, disj_Y, SMB, SNB,
                            PR, QR, MR, NR, MBR, NBR, disi_T, disj_T, SMBR, SNBR, cores));
#else
            SYNC_TIME_STOP();
#endif
        }

        /* Print info to draw figures */
        if( 0 == rank && time ) {
            double ratio_remote = results[7] / results[2];
            double input_bandwidth_mix = network_bandwidth && memcpy_bandwidth ?
                    network_bandwidth * memcpy_bandwidth / ((ratio_remote + 1) * network_bandwidth + memcpy_bandwidth) / 1.0e9 : 0.0;
            double input_bandwidth_worst = network_bandwidth && memcpy_bandwidth ?
                    network_bandwidth * memcpy_bandwidth / ((ratio_remote + 2) * network_bandwidth + memcpy_bandwidth) / 1.0e9 : 0.0;
#if PRINT_MORE
            printf("'Time_PTG', 'Time_DTD', 'm', 'n', 'P', 'Q', 'M', 'N', 'MB', 'NB', 'I', 'J', 'SMB', 'SNB', "
                    "'PR', 'QR', 'MR', 'NR', 'MBR', 'NBR', 'i', 'j', 'SMBR', 'SNBR', 'cores', 'nodes', "
                    "'ratio_remote', 'thread_multiple', 'no_optimization_version', "
                    "'Total_message_remote_bits', 'Total_message_local_bits', "
                    "'Max_send_or_receive_message_each_rank_bits', 'Max_local_message_each_rank_bits', "
                    "'Number_of_message_remote', 'Number_of_message_local', 'Max_remote_message_each_rank_bits', "
                    "'Max_local_related_remote', 'Input_network_bandwidth_Gbits', 'Input_memcpy_bandwidth_Gbits', "
                    "'Input_bandwidth_mix_Gbits', 'Input_bandwidth_worst_Gbits', "
                    "'Output_network_bandwidth_ptg_Gbits', 'Output_network_bandwidth_ptg_bidir_Gbits', "
                    "'Output_memcpy_bandwidth_ptg_Gbits', 'Output_network_bandwidth_dtd_Gbits', "
                    "'Output_network_bandwidth_dtd_bidir_Gbits', 'Output_memcpy_bandwidth_dtd_Gbits' "
                    "\n\n");
#endif
            printf("OUTPUT %lf %lf %d %d %d %d %d %d %d %d %d %d %d %d "
                   "%d %d %d %d %d %d %d %d %d %d %d %d %.2lf %d %d "
                   "%.10e %.10e %.10e %.10e %.2lf %.2lf %.10e %.10e "
                   "%.2lf %.2lf %.2lf %.2lf %.2lf %.2lf %.2lf %.2lf %.2lf %.2lf\n",
                   time_ptg, time_dtd, size_row, size_col, P, Q, M, N, MB, NB, disi_Y, disj_Y, SMB, SNB,
                   PR, QR, MR, NR, MBR, NBR, disi_T, disj_T, SMBR, SNBR, cores, nodes, ratio_remote,
                   thread_type, no_optimization_version,
                   results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7],
                   network_bandwidth / 1.0e9, memcpy_bandwidth / 1.0e9,
                   input_bandwidth_mix, input_bandwidth_worst,
                   (time_ptg ? results[2] / 1.0e9 / time_ptg : 0.0),
                   (time_ptg ? results[6] / 1.0e9 / time_ptg : 0.0),
                   (time_ptg ? (results[2] + results[3]) / 1.0e9 / time_ptg : 0.0),
                   (time_dtd ? results[2] / 1.0e9 / time_dtd: 0.0),
                   (time_dtd ? results[6] / 1.0e9 / time_ptg : 0.0),
                   (time_dtd ? (results[2] + results[3]) / 1.0e9 / time_dtd : 0.0));
        }
        free(results);
        results = NULL;

    }

cleanup_all:
    /* Free memory */
    if( NULL != dcT.desc ) {
        parsec_tiled_matrix_destroy(dcT.desc);
    }
    testing_redistribute_free_storage(&dcT);

    if( NULL != dcY.desc ) {
        parsec_tiled_matrix_destroy(dcY.desc);
    }
    testing_redistribute_free_storage(&dcY);

    cleanup_parsec(parsec, iparam, dparam);
    if( PARSEC_ERR_DEVICE == rc ) {
        return -PARSEC_ERR_DEVICE;
    }
    return (PARSEC_SUCCESS == rc) ? 0 : 1;
}
