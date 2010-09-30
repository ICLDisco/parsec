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
#include <lapack.h>
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


void create_tile_cholesky_float(tiled_matrix_desc_t * Ddesc, void * position, unsigned int row, unsigned int col)
{
    unsigned int i, j, first_row, first_col;
    unsigned int nb = Ddesc->nb;
    unsigned int mn_max = Ddesc->n > Ddesc->m ? Ddesc->n : Ddesc->m;
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

void create_tile_lu_float(tiled_matrix_desc_t * Ddesc, void * position, unsigned int row, unsigned int col)
{
    unsigned int i, j, first_row, first_col;
    unsigned int nb = Ddesc->nb;
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

void create_tile_zero(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col)
{
    (void)row;
    (void)col;
    memset( position, 0, Ddesc->bsiz * Ddesc->mtype );
}

void create_tile_cholesky_double(tiled_matrix_desc_t * Ddesc, void * position, unsigned int row, unsigned int col)
{
    unsigned int i, j, first_row, first_col;
    unsigned int nb = Ddesc->nb;
    unsigned int mn_max = Ddesc->n > Ddesc->m ? Ddesc->n : Ddesc->m;
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

void create_tile_lu_double(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col)
{
    unsigned int i, j, first_row, first_col;
    unsigned int nb = Ddesc->nb;
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
    unsigned int row;
    unsigned int col;
} tile_coordinate_t;

typedef struct info_tiles{
    int th_id;    
    tiled_matrix_desc_t * Ddesc;
    tile_coordinate_t * tiles;    
    unsigned int nb_elements;
    unsigned int starting_position;
    void (*gen_fct)( tiled_matrix_desc_t *, void *, unsigned int, unsigned int);
} info_tiles_t;




/* thread function for affecting multiple tiles with random values
 * @param : tiles : of type dist_tiles_t

 */
static void * rand_dist_tiles(void * info)
{
    unsigned int i;
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
    unsigned int tiles_coord_size;      /* size of the above table */
    unsigned int i;
    unsigned int j;
    unsigned int pos = 0;
    pthread_t *threads = NULL;
    pthread_attr_t thread_attr;
    info_tiles_t * info_gen;
    tiles_coord_size = (Mdesc->lmt * Mdesc->lnt) / Mdesc->super.nodes; /* average number of tiles per nodes */
    tiles_coord_size = (3*tiles_coord_size)/2; /* consider imbalance in distribution */
    tiles = malloc(tiles_coord_size * sizeof(tile_coordinate_t));

    /* check which tiles to generate */
    for ( j = 0 ; j < Mdesc->lnt ; j++)
        for ( i = 0 ; i < Mdesc->lmt ; i++)
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
                    info_gen[i].gen_fct = create_tile_zero;
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

int data_write(tiled_matrix_desc_t * Ddesc, char * filename){
    FILE * tmpf;
    size_t i, j;
    void* buf;
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
                        buf = Ddesc->super.data_of((dague_ddesc_t *)Ddesc, i, j);
                        fwrite(buf, Ddesc->mtype, Ddesc->bsiz, tmpf );
                    }
            }
    fclose(tmpf);
    return 0;
}

int data_read(tiled_matrix_desc_t * Ddesc, char * filename){
    FILE * tmpf;
    size_t i, j;
    void * buf;
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
                        buf = Ddesc->super.data_of((dague_ddesc_t *)Ddesc, i, j);
                        fread(buf, Ddesc->mtype, Ddesc->bsiz, tmpf);
                    }
            }
    fclose(tmpf);
    return 0;
}


#ifdef USE_MPI

void compare_dist_data_double(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b)
{
    MPI_Status status;
    void * bufferA;
    void * bufferB;
    void * tmpA = malloc(a->bsiz * a->mtype);
    void * tmpB = malloc(a->bsiz * a->mtype);

    size_t i,j;
    unsigned int k;
    uint32_t rankA, rankB;
    unsigned int count = 0;
    int diff, dc;
    double eps;
    


    eps= lapack_dlamch(lapack_eps);
    // eps = 1e-13;
    printf("epsilon is %e\n", eps);    

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
                                if (rankA < a->super.nodes)
                                    {
                                        MPI_Recv(tmpA, a->bsiz, MPI_DOUBLE, rankA, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
                                        bufferA = tmpA;
                                    }
                            }
                        if ( rankB == 0)
                            {
                                bufferB = b->super.data_of((dague_ddesc_t *) b, i, j );
                            }
                        else
                            {
                                if (rankB < a->super.nodes)
                                    {
                                        MPI_Recv(tmpB, b->bsiz, MPI_DOUBLE, rankB, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
                                        bufferB = tmpB;
                                    }
                            }
                        if(rankA < a->super.nodes)
                            {
                                diff = 0;
                                dc = 0;
                                for(k = 0 ; k < a->bsiz ; k++)
                                    if ( ( (((double *)bufferA)[k] - ((double *)bufferB)[k]) > eps) || (( ((double *)bufferA)[k]-((double *)bufferB)[k]) < -eps)  )
                                        {
                                            diff = 1;
                                            dc++;
                                        }
                                
                                if (diff)
                                    {
                                        count++;
                                        printf("tile (%zu, %zu) differs in %d numbers\n", i, j, dc);
                                    }
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



void compare_dist_data_float(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b)
{
    MPI_Status status;
    void * bufferA;
    void * bufferB;
    void * tmpA = malloc(a->bsiz * a->mtype);
    void * tmpB = malloc(a->bsiz * a->mtype);

    size_t i,j;
    unsigned int k;
    uint32_t rankA, rankB;
    unsigned int count = 0;
    int diff, dc;
    float eps;
    
    

    eps= lapack_slamch(lapack_eps);
    // eps = 1e-8;
    printf("epsilon is %e\n", eps);    

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
                                if(rankA < a->super.nodes)
                                    {
                                        MPI_Recv(tmpA, a->bsiz, MPI_FLOAT, rankA, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
                                        bufferA = tmpA;
                                    }
                            }
                        if ( rankB == 0)
                            {
                                bufferB = b->super.data_of((dague_ddesc_t *) b, i, j );
                            }
                        else
                            {
                                if(rankB < a->super.nodes)
                                    {
                                        MPI_Recv(tmpB, b->bsiz, MPI_FLOAT, rankB, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
                                        bufferB = tmpB;
                                    }
                            }
                        if (rankA < a->super.nodes)
                            {
                                diff = 0;
                                dc = 0;
                                // printf("a: %e, b: %e\n", ((float *)bufferA)[0], ((float *)bufferB)[0]);
                                for(k = 0 ; k < a->bsiz ; k++)
                                    if ( ( (((float *)bufferA)[k] - ((float *)bufferB)[k]) > eps) || (( ((float *)bufferA)[k]-((float *)bufferB)[k]) < -eps)  )
                                        {
                                            diff = 1;
                                            dc++;
                                        }
                                
                                if (diff)
                                    {
                                        count++;
                                        printf("tile (%zu, %zu) differs in %d numbers\n", i, j, dc);
                                    }
                            }
                        
                    }
                else /* a->super.myrank != 0 */
                    {
                        
                        if ( rankA == a->super.myrank)
                            {
                                MPI_Send(a->super.data_of((dague_ddesc_t *) a, i, j ), a->bsiz, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
                            }
                        if ( rankB == b->super.myrank)
                            {
                                MPI_Send(b->super.data_of((dague_ddesc_t *) b, i, j ), b->bsiz, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                                                    
                            }
                    }
            }
    if(a->super.myrank == 0)
        printf("compared the matrices: %u difference(s)\n", count);
}
#endif

#ifndef USE_MPI
void compare_dist_data_double(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b)
{
    return;
}
void compare_dist_data_float(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b)
{
    return;
}

#endif
