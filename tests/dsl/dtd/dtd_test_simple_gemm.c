#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"

// The file is not compiled if CUDA is not present or CUBLAS is not found
#include "parsec/mca/device/cuda/device_cuda.h"
#include "cublas_v2.h"

#if defined(HAVE_BLAS)
// If our CMake finds a BLAS library, it defines HAVE_BLAS
// BLAS does not guarantee there is a cblas.h, we define our own prototype
typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
#define CBLAS_INDEX int

extern void cblas_dgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N,
                        const CBLAS_INDEX K, const double alpha, const double  *A,
                        const CBLAS_INDEX lda, const double  *B, const CBLAS_INDEX ldb,
                        const double beta, double  *C, const CBLAS_INDEX ldc);
#endif

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

#include <unistd.h>
#include <getopt.h>

static int TILE_FULL = -1;
static parsec_info_id_t CuHI = -1;
static parsec_info_id_t Cu1 = -1;
static int verbose = 0;
static int device = PARSEC_DEV_CUDA;
static int P = -1;
static int Q = -1;

#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20
#define NBELEM 1

static unsigned long long int Rnd64_jump(unsigned long long int n, unsigned long long int seed)
{
    unsigned long long int a_k, c_k, ran;
    int i;

    a_k = Rnd64_A;
    c_k = Rnd64_C;

    ran = seed;
    for( i = 0; n; n >>= 1, ++i ) {
        if( n & 1 )
            ran = a_k * ran + c_k;
        c_k *= (a_k + 1);
        a_k *= a_k;
    }

    return ran;
}

int initialize_tile(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    double *data;
    int i, j, mb, nb, m, n, M, ld;
    unsigned int seed;
    unsigned long long jump, ran;

    parsec_dtd_unpack_args(this_task, &data, &m, &n, &mb, &nb, &M, &ld, &seed);

    jump = (unsigned long long int)m + (unsigned long long int)n * (unsigned long long int)M;

    for( j = 0; j < nb; j++ ) {
        ran = Rnd64_jump(NBELEM * jump, seed);
        for( i = 0; i < mb; i++ ) {
            *data = 0.5f - ran * RndF_Mul;
            ran = Rnd64_A * ran + Rnd64_C;
            data++;
        }
        data += ld - i;
        jump += M;
    }
    return PARSEC_HOOK_RETURN_DONE;
}

int initialize_matrix(parsec_context_t *parsec_context, int rank, parsec_matrix_block_cyclic_t *mat, unsigned int seed,
                      const char *name, int *gpu_device_index, int nb_gpus)
{
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_data_key_t key;
    int perr;

    parsec_task_class_t *init_tc;

    perr = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    perr = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");

    init_tc = parsec_dtd_create_task_class(tp, "init",
                                           PASSED_BY_REF, PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                           sizeof(int), PARSEC_VALUE,          /* m    */
                                           sizeof(int), PARSEC_VALUE,          /* n    */
                                           sizeof(int), PARSEC_VALUE,          /* mb   */
                                           sizeof(int), PARSEC_VALUE,          /* nb   */
                                           sizeof(int), PARSEC_VALUE,          /* M    */
                                           sizeof(int), PARSEC_VALUE,          /* ld   */
                                           sizeof(unsigned int), PARSEC_VALUE, /* seed */
                                           PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, init_tc, PARSEC_DEV_CPU, initialize_tile);

    int g = 0;
    for( int i = 0; i < mat->super.mt; i++ ) {
        for( int j = 0; j < mat->super.nt; j++ ) {
            key = mat->super.super.data_key(&mat->super.super, i, j);
            parsec_dtd_insert_task_with_task_class(tp, init_tc, 1, PARSEC_DEV_CPU,
                                                   PARSEC_PUSHOUT, PARSEC_DTD_TILE_OF_KEY(&mat->super.super, key),
                                                   PARSEC_DTD_EMPTY_FLAG, &i,
                                                   PARSEC_DTD_EMPTY_FLAG, &j,
                                                   PARSEC_DTD_EMPTY_FLAG, &mat->super.mb,
                                                   PARSEC_DTD_EMPTY_FLAG, &mat->super.nb,
                                                   PARSEC_DTD_EMPTY_FLAG, &mat->super.m,
                                                   PARSEC_DTD_EMPTY_FLAG, &mat->super.mb,
                                                   PARSEC_DTD_EMPTY_FLAG, &seed,
                                                   PARSEC_DTD_ARG_END);
            if(PARSEC_DEV_CUDA == device &&
               (int)mat->super.super.rank_of_key(&mat->super.super, key) == rank ) {
                if( verbose ) {
                    fprintf(stderr, "Advice %s(%d, %d) to prefer GPU device %d (parsec device %d) of rank %d\n",
                            name, i, j, g, gpu_device_index[g], (int)mat->super.super.rank_of_key(&mat->super.super, key));
                }
                parsec_advise_data_on_device(mat->super.super.data_of_key(&mat->super.super, key),
                                             gpu_device_index[g],
                                             PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE);
            }
            g = (g + 1) % nb_gpus;
        }
    }
    parsec_dtd_data_flush_all(tp, &mat->super.super);

    // Wait for task completion
    perr = parsec_dtd_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(perr, "parsec_dtd_taskpool_wait");

    perr = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_wait");

    parsec_dtd_task_class_release(tp, init_tc);

    parsec_taskpool_free(tp);

    return 0;
}

int gemm_kernel_cuda(parsec_device_gpu_module_t *gpu_device,
                     parsec_gpu_task_t *gpu_task,
                     parsec_gpu_exec_stream_t *gpu_stream)
{
    double *A, *B, *C;
    int m, n, k, mb, nb, kb;
    parsec_task_t *this_task = gpu_task->ec;
    cublasStatus_t status;
    cublasHandle_t handle;
    double *one_device = NULL;
    struct timeval start, end, diff;
    double delta;
    double *a_gpu, *b_gpu, *c_gpu;

    (void)gpu_stream;
    (void)gpu_device;

    parsec_dtd_unpack_args(this_task,
                           &A, &B, &C,
                           &m, &n, &k,
                           &mb, &nb, &kb);

    a_gpu = parsec_dtd_get_dev_ptr(this_task, 0);
    b_gpu = parsec_dtd_get_dev_ptr(this_task, 1);
    c_gpu = parsec_dtd_get_dev_ptr(this_task, 2);

    handle = parsec_info_get(&gpu_stream->infos, CuHI);
    assert(NULL != handle);
    one_device = parsec_info_get(&gpu_device->super.infos, Cu1);
    assert(NULL != one_device);
    gettimeofday(&start, NULL);

    status = cublasDgemm_v2(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            mb, nb, kb,
                            one_device, a_gpu, mb,
                            b_gpu, kb,
                            one_device, c_gpu, mb);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    delta = (double)diff.tv_sec + (double)diff.tv_usec/1e6;
    if(verbose)
        fprintf(stderr, "GEMM(%d, %d, %d) with tiles of %dx%d, %dx%d, %dx%d on node %d, GPU %s submitted in %g s\n",
                m, n, k, mb, kb, kb, nb, mb, kb,
                this_task->taskpool->context->my_rank,
                gpu_stream->name, delta);

    PARSEC_CUDA_CHECK_ERROR("cublasDgemm_v2 ", status,
                            { return PARSEC_HOOK_RETURN_ERROR; });

    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(HAVE_BLAS)
int gemm_kernel_cpu(parsec_execution_stream_t *es,
                    parsec_task_t *this_task)
{
    double *A, *B, *C;
    int m, n, k, mb, nb, kb;
    double alpha = 1.0;
    double beta = 1.0;
    double delta;
    struct timeval start, end, diff;

    (void)es;

    parsec_dtd_unpack_args(this_task,
                           &A, &B, &C,
                           &m, &n, &k,
                           &mb, &nb, &kb);

    gettimeofday(&start, NULL);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mb, nb, kb, alpha, A, mb, B, kb, beta, C, mb);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);

    delta = (double)diff.tv_sec + (double)diff.tv_usec/1e6;
    if( verbose )
        fprintf(stderr, "GEMM(%d, %d, %d) with tiles of %dx%d, %dx%d, %dx%d on node %d, on core %d: %g s\n",
                m, n, k, mb, kb, kb, nb, mb, kb,
                this_task->taskpool->context->my_rank,
                es->core_id,
                delta);

    return PARSEC_HOOK_RETURN_DONE;
}
#endif

int simple_gemm(parsec_context_t *parsec_context, parsec_matrix_block_cyclic_t *A, parsec_matrix_block_cyclic_t *B, parsec_matrix_block_cyclic_t *C)
{
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_data_key_t keyA, keyB, keyC;
    int perr;

    parsec_task_class_t *gemm_tc;

    perr = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    perr = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");

    gemm_tc = parsec_dtd_create_task_class(tp, "GEMM",
                                           PASSED_BY_REF, PARSEC_INPUT | TILE_FULL, /* A  */
                                           PASSED_BY_REF, PARSEC_INPUT | TILE_FULL, /* B  */
                                           PASSED_BY_REF, PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY, /* C  */
                                           sizeof(int), PARSEC_VALUE,               /* m  */
                                           sizeof(int), PARSEC_VALUE,               /* n  */
                                           sizeof(int), PARSEC_VALUE,               /* k  */
                                           sizeof(int), PARSEC_VALUE,               /* mb */
                                           sizeof(int), PARSEC_VALUE,               /* nb */
                                           sizeof(int), PARSEC_VALUE,               /* kb */
                                           PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, gemm_tc, PARSEC_DEV_CUDA, gemm_kernel_cuda);
#if defined(HAVE_BLAS)
    parsec_dtd_task_class_add_chore(tp, gemm_tc, PARSEC_DEV_CPU, gemm_kernel_cpu);
#endif

    for( int i = 0; i < C->super.mt; i++ ) {
        for( int j = 0; j < C->super.nt; j++ ) {
            keyC = C->super.super.data_key(&C->super.super, i, j);
            for( int k = 0; k < A->super.nt; k++ ) {
                keyA = A->super.super.data_key(&A->super.super, i, k);
                keyB = B->super.super.data_key(&B->super.super, k, j);
                parsec_dtd_insert_task_with_task_class(tp, gemm_tc, C->super.mt*C->super.nt*A->super.nt - i*C->super.nt + j, device,
                                                       PARSEC_INPUT, PARSEC_DTD_TILE_OF_KEY(&A->super.super, keyA),
                                                       PARSEC_INPUT, PARSEC_DTD_TILE_OF_KEY(&B->super.super, keyB),
                                                       k == A->super.nt - 1 ? (PARSEC_INOUT | PARSEC_PUSHOUT) : PARSEC_INOUT,
                                                       PARSEC_DTD_TILE_OF_KEY(&C->super.super, keyC),
                                                       PARSEC_DTD_EMPTY_FLAG, &i,
                                                       PARSEC_DTD_EMPTY_FLAG, &j,
                                                       PARSEC_DTD_EMPTY_FLAG, &k,
                                                       PARSEC_DTD_EMPTY_FLAG, &C->super.mb,
                                                       PARSEC_DTD_EMPTY_FLAG, &C->super.nb,
                                                       PARSEC_DTD_EMPTY_FLAG, &B->super.mb,
                                                       PARSEC_DTD_ARG_END);
            }
        }
    }
    parsec_dtd_data_flush_all(tp, &A->super.super);
    parsec_dtd_data_flush_all(tp, &B->super.super);
    parsec_dtd_data_flush_all(tp, &C->super.super);

    // Wait for task completion
    perr = parsec_dtd_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(perr, "parsec_dtd_taskpool_wait");

    perr = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_wait");

    parsec_dtd_task_class_release(tp, gemm_tc);

    parsec_taskpool_free(tp);

    return 0;
}

int get_nb_gpu_devices()
{
    int nb = 0;

    for( int dev = 0; dev < (int)parsec_nb_devices; dev++ ) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            nb++;
        }
    }

    return nb;
}

int *get_gpu_device_index()
{
    int *dev_index = NULL;

    dev_index = (int *)malloc(parsec_nb_devices * sizeof(int));
    int i = 0;
    for( int dev = 0; dev < (int)parsec_nb_devices; dev++ ) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            dev_index[i++] = device->device_index;
        }
    }

    return dev_index;
}

static void destroy_cublas_handle(void *_h, void *_n)
{
#if defined(PARSEC_HAVE_CUDA)
    cublasHandle_t cublas_handle = (cublasHandle_t)_h;
    cublasDestroy_v2(cublas_handle);
#endif
    (void)_n;
    (void)_h;
}

static void *create_cublas_handle(void *obj, void *p)
{
#if defined(PARSEC_HAVE_CUDA)
    cublasHandle_t handle;
    cublasStatus_t status;
    parsec_cuda_exec_stream_t *stream = (parsec_cuda_exec_stream_t *)obj;
    (void)p;
    /* No need to call cudaSetDevice, as this has been done by PaRSEC before calling the task body */
    status = cublasCreate(&handle);
    assert(CUBLAS_STATUS_SUCCESS == status);
    status = cublasSetStream(handle, stream->cuda_stream);
    assert(CUBLAS_STATUS_SUCCESS == status);
    (void)status;
    return (void *)handle;
#else
    (void)obj;
    (void)p;
    return NULL;
#endif
}

static void destroy_one_on_device(void *_h, void *_n)
{
#if defined(PARSEC_HAVE_CUDA)
    cudaFree(_h);
#endif
    (void)_h;
    (void)_n;
}

static void *allocate_one_on_device(void *obj, void *p)
{
     (void)obj;
     (void)p;
#if defined(PARSEC_HAVE_CUDA)
     void *one_device;
     double one_host = 1.0;
     cudaError_t cr;

     cr = cudaMallocManaged(&one_device, sizeof(double), cudaMemAttachGlobal);
     PARSEC_CUDA_CHECK_ERROR("cudaMalloc ", cr,
                            { return NULL; });

     cr = cudaMemcpy(one_device, &one_host, sizeof(double), cudaMemcpyHostToDevice);
     PARSEC_CUDA_CHECK_ERROR("cudaMemcpy ", cr,
                            { return NULL; });

     return one_device;
#else
    return NULL;
#endif
}

static parsec_matrix_block_cyclic_t *create_initialize_matrix(parsec_context_t *parsec_context, int rank, unsigned int seed, const char *name, int mb, int nb, int M, int N, int *gpu_device_index, int nbgpus)
{
    parsec_matrix_block_cyclic_t *dc;
    dc = calloc(1, sizeof(parsec_matrix_block_cyclic_t));
    parsec_matrix_block_cyclic_init(dc, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE, rank,
                              mb, nb,
                              M, N,
                              0, 0,
                              M, N,
                              P, Q,
                              1, 1,
                              0, 0);
    parsec_data_collection_t *A = &dc->super.super;
    parsec_data_collection_set_key(A, name);
    dc->mat = parsec_data_allocate((size_t)dc->super.nb_local_tiles *
                                   (size_t)dc->super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dc->super.mtype));
    parsec_dtd_data_collection_init(A);
    initialize_matrix(parsec_context, rank, dc, seed, name, gpu_device_index, nbgpus);

    return dc;
}

static void destroy_matrix(parsec_matrix_block_cyclic_t *dc)
{
    parsec_data_collection_t *A = &dc->super.super;
    parsec_dtd_data_collection_fini(A);
    if( NULL != dc->mat ) {
        parsec_data_free(dc->mat);
    }
    parsec_tiled_matrix_destroy_data(&dc->super);
    parsec_data_collection_destroy(&dc->super.super);

    free(dc);
}

int main(int argc, char **argv)
{
    int ret = 0, rc, nbgpus = 0;
    parsec_context_t *parsec_context = NULL;
    int rank, world;
    int mb = 1024, nb = 1024, kb = 1024;
    int M = 16 * mb, N = 16 * nb, K = 16 * kb;
    double min_perf=0.0;
    int runs = 5;
    int debug=-1;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    while( 1 ) {
        int option_index = 0;
        static struct option long_options[] = {
                {"M",       required_argument, 0, 'M'},
                {"N",       required_argument, 0, 'N'},
                {"K",       required_argument, 0, 'K'},
                {"mb",      required_argument, 0, 'm'},
                {"nb",      required_argument, 0, 'n'},
                {"kb",      required_argument, 0, 'k'},
                {"P",       required_argument, 0, 'P'},
                {"Q",       required_argument, 0, 'Q'},
                {"device",  required_argument, 0, 'd'},
                {"nruns",   required_argument, 0, 't'},
                {"verbose", no_argument,       0, 'v'},
                {"Debug",   required_argument, 0, 'D'},
                {"Alarm",   required_argument, 0, 'A'},
                {"help",    no_argument,       0, 'h'},
                {0, 0,                         0, 0}
        };

        int c = getopt_long(argc, argv, "M:N:K:m:n:k:P:Q:t:d:D:A:vh",
                            long_options, &option_index);
        if( c == -1 )
            break;

        switch( c ) {
            case 'M':
                M = atoi(optarg);
                break;
            case 'N':
                N = atoi(optarg);
                break;
            case 'K':
                K = atoi(optarg);
                break;
            case 'm':
                mb = atoi(optarg);
                break;
            case 'n':
                nb = atoi(optarg);
                break;
            case 'k':
                kb = atoi(optarg);
                break;
            case 'P':
                P = atoi(optarg);
                break;
            case 'Q':
                Q = atoi(optarg);
                break;
            case 't':
                runs = atoi(optarg);
                break;
            case 'v':
                verbose = !verbose;
                break;
            case 'd':
                if(strcmp(optarg, "GPU") == 0) {
                    device=PARSEC_DEV_CUDA;
                } else if(strcmp(optarg, "CPU") == 0) {
#if defined(HAVE_BLAS)
                    device=PARSEC_DEV_CPU;
#else
                    fprintf(stderr, "Error: requested to run on CPU (--device=CPU), but no BLAS library has been found at configure time\n");
                    exit(1);
#endif
                } else {
                    fprintf(stderr, "Error: device parameter should either be 'GPU' or 'CPU' (got '%s')\n", optarg);
                    exit(1);
                }
                break;
            case 'D':
                debug = atoi(optarg);
                break;
            case 'A':
                min_perf = strtod(optarg, NULL);
                break;
            case 'h':
            case '?':
                fprintf(stderr,
                        "Usage %s [flags] [-- <parsec options>]\n"
                        " Compute pdgemm on a process grid of PxQ, using all available GPUs on each\n"
                        " node (modulo parsec options), using DTD. Compute C += AxB, where A is MxK\n"
                        " tiled in mb x kb, B is KxN tiled in kb x nb, and C is MxN tiled in mb x nb\n"
                        " Executes nruns+1 executions of the GEMM operation, and display the last\n"
                        " nruns timing and performance.\n"
                        " flags:\n"
                        "   --M|-M  / --K|-K  / --N|-N:   set M, K and N (resp.)\n"
                        "   --mb|-m / --kb/-k / --nb|-n:  set mb, kb and nb (resp.)\n"
                        "   --nruns|-t:                   set the number of runs to do\n"
                        "   --device|-d:                  which device to use (CPU or GPU)\n"
                        "   --verbose|-v:                 display which GEMM runs on which GPU\n"
                        "                                 as execution is unfolding\n"
                        "   --help|-h|-?:                 display this help\n"
                        "   --debug|-D:                   blocks the process passed as parameter and\n"
                        "                                 waits for gdb to connect to it\n"
                        "   --Alarm|-A:                   sets the expected minimum performance for a\n"
                        "                                 single GPU (kills the process if it takes longer\n"
                        "                                 than the time corresponding to the expected\n"
                        "                                 performance to complete the product)\n"
                        "\n",
                        argv[0]);
                break;
        }
    }
    int pargc = argc - optind + 1;
    char **pargv = (char **)malloc((pargc + 1) * sizeof(char *));
    pargv[0] = argv[0];
    for( int i = 0; i < argc - optind; i++ )
        pargv[i + 1] = argv[optind + i];
    pargv[pargc] = NULL;

    if( -1 == P )
        P = (int)sqrt(world);
    if( -1 == Q )
        Q = world / P;
    while( P * Q != world ) {
        P--;
        Q = world / P;
    }

    if(debug == rank) {
        int loop=1;
        char hostname[64];
        gethostname(hostname, 64);
        fprintf(stderr, "ssh -t %s gdb -p %d\n", hostname, getpid());
        while(loop) { sleep(1); }
    }

    // Number of CPU cores involved
    int ncores = -1; // Use all available cores
    parsec_context = parsec_init(ncores, &pargc, &pargv);

    int *gpu_device_index = NULL;
    if( PARSEC_DEV_CUDA == device ) {
        nbgpus = get_nb_gpu_devices();
        rc = !(nbgpus >= 1);
        if( rc != 0 ) {
            fprintf(stderr, "Rank %d doesn't have CUDA accelerators\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 0);
            return -1;
        }
        gpu_device_index = get_gpu_device_index();

        // Prepare CUBLAS Handle marshaller
        CuHI = parsec_info_register(&parsec_per_stream_infos, "CUBLAS::HANDLE",
                                    destroy_cublas_handle, NULL,
                                    create_cublas_handle, NULL,
                                    NULL);
        assert(CuHI != -1);
        Cu1 = parsec_info_register(&parsec_per_device_infos, "DEVICE::ONE",
                                   destroy_one_on_device, NULL,
                                   allocate_one_on_device, NULL,
                                   NULL);
        assert(Cu1 != -1);
    }

    // Create datatypes
    parsec_arena_datatype_t *adt = parsec_dtd_create_arena_datatype(parsec_context, &TILE_FULL);
    parsec_add2arena_rect(adt, parsec_datatype_double_t, mb, nb, mb);

    // Create and initialize the data
    parsec_matrix_block_cyclic_t *dcA = create_initialize_matrix(parsec_context, rank, 1789, "A", mb, kb, M, K,
                                                           gpu_device_index, nbgpus);
    parsec_matrix_block_cyclic_t *dcB = create_initialize_matrix(parsec_context, rank, 1805, "B", kb, nb, K, N,
                                                           gpu_device_index, nbgpus);
    parsec_matrix_block_cyclic_t *dcC = create_initialize_matrix(parsec_context, rank, 1901, "C", mb, nb, M, N,
                                                           gpu_device_index, nbgpus);

    for( int r = 0; r < runs + 1; r++ ) {
        double gflop = 2.0 * M * N * K / 1e9;
        double maxtime = 0.0;
        if(min_perf > 0.0)
            maxtime = gflop/world/nbgpus/min_perf;
        struct timeval start, end, diff;
        if(maxtime > 0.0 && maxtime < 60.0) maxtime=60.0;
        if(rank == 0 && maxtime > 0.0) fprintf(stderr, "watchdog: %d seconds\n", (int)maxtime);
        if(maxtime > 0.0) alarm((int)maxtime);
        gettimeofday(&start, NULL);
        simple_gemm(parsec_context, dcA, dcB, dcC);
        gettimeofday(&end, NULL);
        timersub(&end, &start, &diff);
        double t = (double)diff.tv_sec + (double)diff.tv_usec / 1e6;
        double gflops = gflop / t;
        if( 0 == rank && r > 0 ) {
            fprintf(stderr, "DTD_GEMM PxQxg: %d %d %d M: %d N: %d K: %d mb: %d nb: %d kb: %d Time(s): %g gflops: %10g\n",
                    P, Q, nbgpus, M, N, K, mb, nb, kb, t, gflops);
        }
    }
    // deactivate the alarm if it was set
    alarm(0);

    if(PARSEC_DEV_CUDA == device) {
        // Cleanup data and parsec data structures
        parsec_info_unregister(&parsec_per_stream_infos, CuHI, NULL);
        parsec_info_unregister(&parsec_per_device_infos, Cu1, NULL);
    }

    parsec_type_free(&adt->opaque_dtt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec_context, TILE_FULL);

    destroy_matrix(dcA);
    destroy_matrix(dcB);
    destroy_matrix(dcC);

    parsec_fini(&parsec_context);

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif

    return ret;
}
