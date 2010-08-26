/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/************************************************************
 *distributed matrix generation
 ************************************************************/
/* affect one tile with random values  */

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>


#ifdef USE_MPI
#include <mpi.h>
#endif

#include "data_dist/data_distribution.h"

#include "matrix.h"
#include "bindthread.h"

/*
 Rnd64seed is a global variable but it doesn't spoil thread safety. All matrix
 generating threads only read Rnd64seed. It is safe to set Rnd64seed before
 and after any calls to create_tile(). The only problem can be caused if
 Rnd64seed is changed during the matrix generation time.
 */

static unsigned long long int Rnd64seed = 100;
#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

static unsigned long long int
Rnd64_jump(unsigned long long int n) {
  unsigned long long int a_k, c_k, ran;
  int i;

  a_k = Rnd64_A;
  c_k = Rnd64_C;

  ran = Rnd64seed;
  for (i = 0; n; n >>= 1, ++i) {
    if (n & 1)
      ran = a_k * ran + c_k;
    c_k *= (a_k + 1);
    a_k *= a_k;
  }

  return ran;
}

void create_tile_zero_float(tiled_matrix_desc_t * Ddesc, void * position,  int row, int col)
{
    (void)row;
    (void)col;
    memset( position, 0, Ddesc->bsiz * sizeof(float) );
}

void create_tile_cholesky_float(tiled_matrix_desc_t * Ddesc, void * position,  int row, int col)
{
    int i, j, first_row, first_col, nb = Ddesc->nb, mn_max = Ddesc->n > Ddesc->m ? Ddesc->n : Ddesc->m;
    float *x = position;
    unsigned long long int ran;

    /* These are global values of first row and column of the tile counting from 0 */
    first_row = row * nb;
    first_col = col * nb;

    for (j = 0; j < nb; ++j) {
      ran = Rnd64_jump( first_row + (first_col + j) * (unsigned long long int)Ddesc->m );

      for (i = 0; i < nb; ++i) {
        x[0] = 0.5f - ran * RndF_Mul;
        ran = Rnd64_A * ran + Rnd64_C;
        x += 1;
      }
    }
    /* This is only required for Cholesky: diagonal is bumped by max(M, N) */
    if (row == col) {
      for (i = 0; i < nb; ++i)
          ((float *) position)[i + i * nb] += mn_max;
    }
}

void create_tile_lu_float(tiled_matrix_desc_t * Ddesc, void * position,  int row, int col)
{
    int i, j, first_row, first_col, nb = Ddesc->nb;
    float *x = position;
    unsigned long long int ran;

    /* These are global values of first row and column of the tile counting from 0 */
    first_row = row * nb;
    first_col = col * nb;

    for (j = 0; j < nb; ++j) {
        ran = Rnd64_jump( first_row + (first_col + j) * (unsigned long long int)Ddesc->m );
        
        for (i = 0; i < nb; ++i) {
            x[0] = 0.5f - ran * RndF_Mul;
            ran = Rnd64_A * ran + Rnd64_C;
            x += 1;
        }
    }
}

void create_tile_zero_double(tiled_matrix_desc_t * Ddesc, void * position,  int row, int col)
{
    (void)row;
    (void)col;
    memset( position, 0, Ddesc->bsiz * sizeof(double) );
}

void create_tile_cholesky_double(tiled_matrix_desc_t * Ddesc, void * position,  int row, int col)
{
    int i, j, first_row, first_col, nb = Ddesc->nb, mn_max = Ddesc->n > Ddesc->m ? Ddesc->n : Ddesc->m;
    double *x = position;
    unsigned long long int ran;

    /* These are global values of first row and column of the tile counting from 0 */
    first_row = row * nb;
    first_col = col * nb;

    for (j = 0; j < nb; ++j) {
      ran = Rnd64_jump( first_row + (first_col + j) * (unsigned long long int)Ddesc->m );

      for (i = 0; i < nb; ++i) {
        x[0] = 0.5 - ran * RndD_Mul;
        ran = Rnd64_A * ran + Rnd64_C;
        x += 1;
      }
    }
    /* This is only required for Cholesky: diagonal is bumped by max(M, N) */
    if (row == col) {
      for (i = 0; i < nb; ++i)
          ((double*)position)[i + i * nb] += mn_max;
    }
}

void create_tile_lu_double(tiled_matrix_desc_t * Ddesc, void * position,  int row, int col)
{
    int i, j, first_row, first_col, nb = Ddesc->nb;
    double *x = position;
    unsigned long long int ran;

    /* These are global values of first row and column of the tile counting from 0 */
    first_row = row * nb;
    first_col = col * nb;

    for (j = 0; j < nb; ++j) {
        ran = Rnd64_jump( first_row + (first_col + j) * (unsigned long long int)Ddesc->m );
        
        for (i = 0; i < nb; ++i) {
            x[0] = 0.5 - ran * RndD_Mul;
            ran = Rnd64_A * ran + Rnd64_C;
            x += 1;
        }
    }
}


typedef struct tile_coordinate{
    int row;
    int col;
} tile_coordinate_t;

typedef struct info_tiles{
    int th_id;    
    tiled_matrix_desc_t * Ddesc;
    tile_coordinate_t * tiles;    
    int nb_elements;
    int starting_position;
    void (*gen_fct)( tiled_matrix_desc_t *, void *, int, int);
} info_tiles_t;




/* thread function for affecting multiple tiles with random values
 * @param : tiles : of type dist_tiles_t

 */
static void * rand_dist_tiles(void * info)
{
    int i;
    /* bind thread to cpu */
    int bind_to_proc = ((info_tiles_t *)info)->th_id;

    dague_bindthread(bind_to_proc);

    /*printf("generating matrix on process %d, thread %d: %d tiles\n",
           ((dist_tiles_t*)tiles)->Ddesc->mpi_rank,
           ((dist_tiles_t*)tiles)->th_id,
           ((dist_tiles_t*)tiles)->nb_elements);*/
    for(i = 0 ; i < ((info_tiles_t *)info)->nb_elements ; i++ )
        {
            ((info_tiles_t *)info)->gen_fct(((info_tiles_t *)info)->Ddesc,
                                              ((info_tiles_t *)info)->Ddesc->super.data_of(
                                                                                           ((struct dague_ddesc *)((info_tiles_t *)info)->Ddesc),
                                                                                           ((info_tiles_t *)info)->tiles[((info_tiles_t *)info)->starting_position + i].row,
                                                                                           ((info_tiles_t *)info)->tiles[((info_tiles_t *)info)->starting_position + i].col ),
                                            ((info_tiles_t *)info)->tiles[((info_tiles_t *)info)->starting_position + i].row,
                                            ((info_tiles_t *)info)->tiles[((info_tiles_t *)info)->starting_position + i].col);
        }
    return NULL;
}

/* affecting the complete local view of a distributed matrix with random values */
static void rand_dist_matrix(tiled_matrix_desc_t * Mdesc, int mtype)
{
    tile_coordinate_t * tiles; /* table of tiles that node will handle */
    int tiles_coord_size;      /* size of the above table */
    unsigned int i;
    int j, pos = 0;
    pthread_t *threads;
    pthread_attr_t thread_attr;
    info_tiles_t * info_gen;
    tiles_coord_size = (Mdesc->lmt * Mdesc->lnt) / Mdesc->super.nodes; /* average number of tiles per nodes */
    tiles_coord_size = (3*tiles_coord_size)/2; /* consider imbalance in distribution */
    tiles = malloc(tiles_coord_size * sizeof(tile_coordinate_t));

    /* check which tiles to generate */
    for ( j = 0 ; j < Mdesc->lnt ; j++)
        for ( i = 0 ; i < (unsigned int)Mdesc->lmt ; i++)
            {
                if(Mdesc->super.myrank ==
                   Mdesc->super.rank_of((dague_ddesc_t *)Mdesc, i, j ))
                    {
                        if (pos == tiles_coord_size)
                            {
                                tiles_coord_size = 2 * tiles_coord_size;
                                tiles = realloc(tiles,
                                                tiles_coord_size*sizeof(tile_coordinate_t));
                                if (NULL == tiles)
                                    {
                                        perror("cannot generate random matrix\n");
                                        exit(-1);
                                    }                                
                            }
                        tiles[pos].row = i;
                        tiles[pos].col = j;
                        pos++;                        
                    }
            }

    /* have 'pos' tiles to generate, knowing which ones. Now gererating them */
    j = 0;
    info_gen = malloc(sizeof(info_tiles_t) * Mdesc->super.cores);
    for ( i = 0 ; i < Mdesc->super.cores ; i++ )
        {
            info_gen[i].th_id = i;
            info_gen[i].tiles = tiles;
            info_gen[i].Ddesc = Mdesc;
            info_gen[i].nb_elements = pos / Mdesc->super.cores;
            info_gen[i].starting_position = j;
            j += info_gen[i].nb_elements;
            if (mtype == 1) /* cholesky like generation (symetric, diagonal dominant) */
                {
                    if(Mdesc->mtype == matrix_RealFloat) 
                        {
                            info_gen[i].gen_fct = create_tile_cholesky_float;
                        }
                    else if (Mdesc->mtype == matrix_RealDouble)
                        {
                            info_gen[i].gen_fct = create_tile_cholesky_double;
                        }
                    else /* unknown type */
                        {
                            printf("unknown generation type: aborting generation\n");
                            free (info_gen);
                            free(tiles);
                            return;
                        }

                }
            else if (mtype == 0)/* LU like generation */
                {
                    if(Mdesc->mtype == matrix_RealFloat) 
                        {
                            info_gen[i].gen_fct = create_tile_lu_float;
                        }
                    else if (Mdesc->mtype == matrix_RealDouble)
                        {
                            info_gen[i].gen_fct = create_tile_lu_double;
                        }
                    else /* unknown type */
                        {
                            printf("unknown generation type: aborting generation\n");
                            free (info_gen);
                            free(tiles);
                            return;
                        }
                }
            else if(mtype == 2)
                {
                    if(Mdesc->mtype == matrix_RealFloat) 
                        {
                            info_gen[i].gen_fct = create_tile_zero_float;
                        }
                    else if (Mdesc->mtype == matrix_RealDouble)
                        {
                            info_gen[i].gen_fct = create_tile_zero_double;
                        }
                    else /* unknown type */
                        {
                            printf("unknown generation type: aborting generation\n");
                            free (info_gen);
                            free(tiles);
                            return;
                        }
                }
        }
    info_gen[i - 1].nb_elements += pos % Mdesc->super.cores;

    if (Mdesc->super.cores > 1)
        {
            pthread_attr_init(&thread_attr);
            pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
#ifdef __linux
            pthread_setconcurrency(Mdesc->super.cores);
#endif            
            threads = malloc((Mdesc->super.cores - 1) * sizeof(pthread_t));
            if (NULL == threads)
                {
                    perror("No memory for generating matrix\n");
                    exit(-1);
                }
                
            for ( i = 1 ; i < Mdesc->super.cores ; i++)
                {
                    pthread_create( &(threads[i-1]),
                                    &thread_attr,
                                    (void* (*)(void*))rand_dist_tiles,
                                    (void*)&(info_gen[i]));
                }
        }

    rand_dist_tiles((void*) &(info_gen[0]));

    if (Mdesc->super.cores > 1)
        {
            for(i = 0 ; i < Mdesc->super.cores - 1 ; i++)
                pthread_join(threads[i],NULL);
            free (threads);
        }
    free(info_gen);
    free(tiles);
    return;
}

void generate_tiled_zero_mat(tiled_matrix_desc_t * Mdesc)
{
    rand_dist_matrix(Mdesc, 2);
}

void generate_tiled_random_sym_pos_mat(tiled_matrix_desc_t * Mdesc)
{
    rand_dist_matrix(Mdesc, 1);
}

void generate_tiled_random_mat(tiled_matrix_desc_t * Mdesc)
{
    rand_dist_matrix(Mdesc, 0);
}



#if defined(DAGUE_CUDA_SUPPORT)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "lifo.h"
#include "gpu_data.h"
extern dague_atomic_lifo_t gpu_devices;
extern int use_gpu;
#endif  /* defined(DAGUE_CUDA_SUPPORT) */

void* dague_allocate_matrix(size_t matrix_size)
{
    void* mat = NULL;
#if defined(DAGUE_CUDA_SUPPORT)
    if( use_gpu ) {
        CUresult status;
        gpu_device_t* gpu_device;

        gpu_device = (gpu_device_t*)dague_atomic_lifo_pop(&gpu_devices);
        if( NULL != gpu_device ) {
            status = cuCtxPushCurrent( gpu_device->ctx );
            DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuCtxPushCurrent ", status,
                                      {goto normal_alloc;} );

            status = cuMemHostAlloc( (void**)&mat, matrix_size, CU_MEMHOSTALLOC_PORTABLE);
            if( CUDA_SUCCESS != status ) {
                DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuMemHostAlloc ", status,
                                          {} );
                mat = NULL;
            }
            status = cuCtxPopCurrent(NULL);
            DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                      {} );
            dague_atomic_lifo_push(&gpu_devices, (dague_list_item_t*)gpu_device);
        }
    }
 normal_alloc:
#endif  /* defined(DAGUE_CUDA_SUPPORT) */
    /* If nothing else worked so far, allocate the memory using PLASMA */
    if( NULL == mat ) {
        mat = malloc( matrix_size );
    }

    if( NULL == mat ) {
        printf("memory allocation of %lu\n", matrix_size);
        perror("matrix allocation failed");
        return NULL;
    }
    return mat;
}

int data_write(tiled_matrix_desc_t * Ddesc, char * filename){
    FILE * tmpf;
    int i, j;
    double * buf;
    tmpf = fopen(filename, "w");
    if(NULL == tmpf)
        {
            printf("opening file: %s", filename);
            return -1;
        }
    for (i = 0 ; i < Ddesc->lmt ; i++)
        for ( j = 0 ; j< Ddesc->lnt ; j++)
            {
                if (Ddesc->super.rank_of((dague_ddesc_t *)Ddesc, i, j) == Ddesc->super.myrank)
                    {
                        buf = (double*)Ddesc->super.data_of((dague_ddesc_t *)Ddesc, i, j);
                        fwrite(buf, Ddesc->mtype, Ddesc->bsiz, tmpf );
                    }
            }
    fclose(tmpf);
    return 0;
}

int data_read(tiled_matrix_desc_t * Ddesc, char * filename){
    FILE * tmpf;
    int i, j;
    double * buf;
    tmpf = fopen(filename, "r");
    if(NULL == tmpf)
        {
            printf("opening file: %s", filename);
            return -1;
        }
    for (i = 0 ; i < Ddesc->lmt ; i++)
        for ( j = 0 ; j< Ddesc->lnt ; j++)
            {
                if (Ddesc->super.rank_of((dague_ddesc_t *)Ddesc, i, j) == Ddesc->super.myrank)
                    {
                        buf = (double*)Ddesc->super.data_of((dague_ddesc_t *)Ddesc, i, j);
                        fread(buf, Ddesc->mtype, Ddesc->bsiz, tmpf);
                    }
            }
    fclose(tmpf);
    return 0;
}


#ifdef USE_MPI
void compare_dist_data(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b)
{
    MPI_Status status;
    void * bufferA;
    void * bufferB;
    void * tmpA = malloc(a->bsiz * a->mtype);
    void * tmpB = malloc(a->bsiz * a->mtype);
    
    int i,j;
    uint32_t rankA, rankB;
    unsigned int count = 0;

    if( (a->bsiz != b->bsiz) || (a->mtype != b->mtype) )
        {
            if(a->super.myrank == 0)
                printf("Cannot compare matrices\n");
            return;
        }
    for(i = 0 ; i < a->lmt ; i++)
        for(j = 0 ; j < a->lnt ; j++)
            {
                rankA = a->super.rank_of((dague_ddesc_t *) a, i, j );
                rankB = b->super.rank_of((dague_ddesc_t *) b, i, j );
                if (a->super.myrank == 0)
                    {
                        if ( rankA == 0)
                            {
                                bufferA = a->super.data_of((dague_ddesc_t *) a, i, j );
                            }
                        else
                            {
                                MPI_Recv(tmpA, a->bsiz, MPI_DOUBLE, rankA, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
                                bufferA = tmpA;
                            }
                        if ( rankB == 0)
                            {
                                bufferB = b->super.data_of((dague_ddesc_t *) b, i, j );
                            }
                        else
                            {
                                MPI_Recv(tmpB, b->bsiz, MPI_DOUBLE, rankB, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
                                bufferB = tmpB;
                            }

                        if (memcmp(bufferA, bufferB, a->bsiz * a->mtype))
                            {
                                count++;
                                printf("tile (%d, %d) differs\n", i, j);
                            }
                        
                    }
                else /* a->super.myrank != 0 */
                    {
                        
                        if ( rankA == a->super.myrank)
                            {
                                MPI_Send(a->super.data_of((dague_ddesc_t *) a, i, j ), a->bsiz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                            }
                        if ( rankB == b->super.myrank)
                            {
                                MPI_Send(b->super.data_of((dague_ddesc_t *) b, i, j ), b->bsiz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);                                                    
                            }
                    }
            }
    if(a->super.myrank == 0)
        printf("compared the matrices: %u difference(s)\n", count);
}

#endif
