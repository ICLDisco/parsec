#include <unistd.h>
#include <getopt.h>

#include "parsec/runtime.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

#include "udf_wrapper.h"

static int32_t calls[UDF_TT_MAX] = { 0, };

static int udf_logger(int p, udf_task_type_t task_type)
{
    parsec_atomic_fetch_inc_int32(&calls[task_type]);
    return p;
}

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    parsec_udf_taskpool_t *udf_tp;
    int largc;
    char **largv;

    static struct option long_options[] = {
        {"P",     required_argument, 0, 'P'},
        {"MB",    required_argument, 0, 'm'},
        {"NB",    required_argument, 0, 'n'},
        {"M",     required_argument, 0, 'M'},
        {"N",     required_argument, 0, 'N'},
        {"cores", required_argument, 0, 'c'},
        {"help",        no_argument, 0, 'h'},
        {0,                       0, 0,   0}
    };
    int option_index = 0, c;
    int P = -1, MB = -1, NB = -1, M = -1, N = -1;
    
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
    P = 1;
#endif

    while(1) {
        option_index = 0;
        c = getopt_long(argc, argv, "P:m:n:M:N:c:h", long_options, &option_index);
        if(-1 == c)
            break;
        switch(c) {
        case 'P':
            P = atoi(optarg);
            break;
        case 'm':
            MB = atoi(optarg);
            break;
        case 'n':
            NB = atoi(optarg);
            break;
        case 'M':
            M = atoi(optarg);
            break;
        case 'N':
            N = atoi(optarg);
            break;
        case 'c':
            cores = atoi(optarg);
            break;
        case 'h':
            if( 0 == rank ) {
                fprintf(stderr,
                        "Usage: %s [-M <M>] [-N <N>] [-m <MB>] [-n <NB>] [-P <P>]\n"
                        " Display how many times a probe function is called to build a basic PTG\n"
                        "  M:  number of rows in the matrix (default N)\n"
                        "  N:  number of columns in the matrix (required)\n"
                        "  MB: number of rows in a tile (default NB)\n"
                        "  NB: number of columns in a tile (required)\n"
                        "  P:  number of rows of processes in the 2D grid (default np, must divide np)\n"
                        "  c:  number of computing threads to create per rank (default one per core)\n"
                        "\n", argv[0]);
#if defined(PARSEC_HAVE_MPI)
                MPI_Abort(MPI_COMM_WORLD, 1);
#endif
                exit(1);
            }
#if defined(PARSEC_HAVE_MPI)
            MPI_Barrier(MPI_COMM_WORLD); /**< Will let the other ranks wait for the MPI_Abort */
#endif
            break; /**< To silent warnings */
        }
    }

    largc = argc - optind;
    largv = argv + optind;
    parsec = parsec_init(cores, &largc, &largv);
    if( NULL == parsec ) {
        exit(-1);
    }


    if( -1 == MB )
        MB = NB;
    if( -1 == M )
        M = N;
    if( -1 == P )
        P = world;
    if( -1 == N || -1 == NB ) {
        if( 0 == rank ) {
            fprintf(stderr, "Incorrect usage, see --help\n");
#if defined(PARSEC_HAVE_MPI)
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        }
#if defined(PARSEC_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD); /**< Will let the other ranks wait for the MPI_Abort */
#endif
        exit(1);
    }
    
    two_dim_block_cyclic_t A;
    two_dim_block_cyclic_init(&A,
                              matrix_ComplexDouble, matrix_Tile,
                              world, rank, MB, NB, M, N, 0, 0,
                              M, N, 1, 1, P);
    A.mat = parsec_data_allocate((size_t)A.super.nb_local_tiles *
                                 (size_t)A.super.bsiz *
                                 (size_t)parsec_datadist_getsizeoftype(A.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&A, "A");

    udf_tp = parsec_udf_new(&A, udf_logger);
    parsec_context_add_taskpool(parsec, &udf_tp->super);
    parsec_context_start(parsec);
    parsec_context_wait(parsec);
    
    printf("Rank %d - %d local tiles\n", rank, A.super.nb_local_tiles);
    for(int i = 0; i < UDF_TT_MAX; i++) {
        printf("Rank %d - user function defined for '%s': iterator is called %d times (%g / tile)\n", rank, UDF_TASKTYPE_NAME[i], calls[i], (double)calls[i]/(double)A.super.nb_local_tiles);
    }
    
    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
