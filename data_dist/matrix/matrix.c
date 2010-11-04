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

#include "dague_config.h"

#include "data_distribution.h"
#include "matrix.h"
#include "bindthread.h"

void create_tile_zero(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed)
{
   
    (void)row;
    (void)col;
    (void)seed;
    memset( position, 0, Ddesc->bsiz * Ddesc->mtype );
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
    unsigned long long int seed;
    void (*gen_fct)( tiled_matrix_desc_t *, void *, unsigned int, unsigned int, unsigned long long int);
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
                                            ((info_tiles_t *)info)->tiles[((info_tiles_t *)info)->starting_position + i].col,
                                            ((info_tiles_t *)info)->seed);
        }
    return NULL;
}

/* affecting the complete local view of a distributed matrix with random values */
static void rand_dist_matrix(tiled_matrix_desc_t * Mdesc, int mtype, unsigned long long int sed)
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
            info_gen[i].seed = sed;
            j += info_gen[i].nb_elements;
            if (mtype == 1) /* cholesky like generation (symetric, diagonal dominant) */
                {
                    if(Mdesc->mtype == matrix_RealFloat) 
                        {
                            info_gen[i].gen_fct = matrix_stile_cholesky;
                        }
                    else if (Mdesc->mtype == matrix_RealDouble)
                        {
                            info_gen[i].gen_fct = matrix_dtile_cholesky;
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
                            info_gen[i].gen_fct = matrix_stile;
                        }
                    else if (Mdesc->mtype == matrix_RealDouble)
                        {
                            info_gen[i].gen_fct = matrix_dtile;
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
    
    rand_dist_matrix(Mdesc, 2, 0);
}

void generate_tiled_random_sym_pos_mat(tiled_matrix_desc_t * Mdesc, unsigned long long int seed)
{
    rand_dist_matrix(Mdesc, 1, seed);
}

void generate_tiled_random_mat(tiled_matrix_desc_t * Mdesc, unsigned long long int seed)
{
    rand_dist_matrix(Mdesc, 0, seed);
}


void pddiagset(tiled_matrix_desc_t * Mdesc, double val){
    unsigned int i, j;
    unsigned int target;
    double * buffer;
    target = (Mdesc->lmt < Mdesc->lnt) ? Mdesc->lmt : Mdesc->lnt;

    for( i = 0 ; i < target ; i++ )
        {
            if(Mdesc->super.myrank == Mdesc->super.rank_of( (dague_ddesc_t *)Mdesc, i, i ))
                {
                    buffer = Mdesc->super.data_of( (dague_ddesc_t *)Mdesc, i, i );
                    for( j = 0 ; j < Mdesc->nb ; j++)
                        {
                            buffer[(j * Mdesc->mb) + j] = val;
                        }
                }
        }
    return;
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
    for (i = 0 ; i < Ddesc->mt ; i++)
        for ( j = 0 ; j< Ddesc->nt ; j++)
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
    for (i = 0 ; i < Ddesc->mt ; i++)
        for ( j = 0 ; j< Ddesc->nt ; j++)
            {
                if (Ddesc->super.rank_of((dague_ddesc_t *)Ddesc, i, j) == Ddesc->super.myrank)
                    {
                        size_t ret;
                        buf = Ddesc->super.data_of((dague_ddesc_t *)Ddesc, i, j);
                        ret = fread(buf, Ddesc->mtype, Ddesc->bsiz, tmpf);
                        if ( ret !=  Ddesc->bsiz )
                            {
                                printf("Error reading file: %s", filename);
                                return -1;
                            }
                    }
            }
    fclose(tmpf);
    return 0;
}
