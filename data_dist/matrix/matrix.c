/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/************************************************************
 *distributed matrix generation
 ************************************************************/
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#if defined(HAVE_MPI)
#include <mpi.h>
#endif

#include "dague_config.h"
#include "data_distribution.h"
#include "matrix.h"

/***************************************************************************//**
 *  Internal static descriptor initializer (PLASMA code)
 **/
void tiled_matrix_desc_init( tiled_matrix_desc_t *tdesc, 
                             enum matrix_type    dtyp, 
                             enum matrix_storage storage, 
                             int mb, int nb,
                             int lm, int ln, 
                             int i,  int j, 
                             int m,  int n)
{
    /* Matrix address */
    /* tdesc->mat = NULL;*/
    /* tdesc->A21 = (lm - lm%mb)*(ln - ln%nb); */
    /* tdesc->A12 = (     lm%mb)*(ln - ln%nb) + tdesc->A21; */
    /* tdesc->A22 = (lm - lm%mb)*(     ln%nb) + tdesc->A12; */

    /* Matrix properties */
    tdesc->mtype   = dtyp;
    tdesc->storage = storage;
    tdesc->tileld  = (storage == matrix_Tile) ? mb : lm;
    tdesc->mb      = mb;
    tdesc->nb      = nb;
    tdesc->bsiz    = mb * nb;

    /* Large matrix parameters */
    tdesc->lm = lm;
    tdesc->ln = ln;

    /* WARNING: This has to be removed when padding will be removed */
#if defined(HAVE_MPI)
    if ( storage == matrix_Lapack ) {
        if ( tdesc->lm %mb != 0 ) {
            fprintf(stderr, "In distributed with Lapack storage, lm has to be a multiple of mb\n");
            MPI_Abort(MPI_COMM_WORLD);
        }
        if ( tdesc->ln %nb != 0 ) {
            fprintf(stderr, "In distributed with Lapack storage, ln has to be a multiple of nb\n");
            MPI_Abort(MPI_COMM_WORLD);
        }
    }
#endif

    /* Large matrix derived parameters */
    /* tdesc->lm1 = (lm/mb); */
    /* tdesc->ln1 = (ln/nb); */
    tdesc->lmt = (lm%mb==0) ? (lm/mb) : (lm/mb+1);
    tdesc->lnt = (ln%nb==0) ? (ln/nb) : (ln/nb+1);

    /* Submatrix parameters */
    tdesc->i = i;
    tdesc->j = j;
    tdesc->m = m;
    tdesc->n = n;

    /* Submatrix derived parameters */
    tdesc->mt = (i+m-1)/mb - i/mb + 1;
    tdesc->nt = (j+n-1)/nb - j/nb + 1;

#if defined(DAGUE_PROF_TRACE)
    asprintf(&(tdesc->super.key_dim), "(%d, %d)", tdesc->lmt, tdesc->lnt);
#endif

    return;
}

/*
 * Writes the data into the file filename
 * Sequential function per node
 */
int tiled_matrix_data_write(tiled_matrix_desc_t *tdesc, char *filename) {
    dague_ddesc_t *ddesc = &(tdesc->super);
    FILE *tmpf;
    void *buf;
    int i, j, k;
    uint32_t myrank = tdesc->super.myrank;
    int eltsize =  dague_datadist_getsizeoftype( tdesc->mtype );

    tmpf = fopen(filename, "w");
    if(NULL == tmpf) {
        fprintf(stderr, "ERROR: The file %s cannot be open\n", filename);
        return -1;
    }

    if ( tdesc->storage == matrix_Tile ) {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( ddesc->rank_of( ddesc, i, j ) == myrank ) {
                    buf = ddesc->data_of( ddesc, i, j );
                    fwrite(buf, eltsize, tdesc->bsiz, tmpf );
                }
            }
    } else {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( ddesc->rank_of( ddesc, i, j ) == myrank ) {
                    buf = ddesc->data_of( ddesc, i, j );
                    for (k=0; k<tdesc->nb; k++) {
                        fwrite(buf, eltsize, tdesc->mb, tmpf );
                        buf += eltsize * tdesc->lm;
                    }
                }
            }
    }


    fclose(tmpf);
    return 0;
}

/*
 * Read the data from the file filename
 * Sequential function per node
 */
int tiled_matrix_data_read(tiled_matrix_desc_t *tdesc, char *filename) {
    dague_ddesc_t *ddesc = &(tdesc->super);
    FILE *tmpf;
    void *buf;
    int i, j, k, ret;
    uint32_t myrank = tdesc->super.myrank;
    int eltsize =  dague_datadist_getsizeoftype( tdesc->mtype );

    tmpf = fopen(filename, "w");
    if(NULL == tmpf) {
        fprintf(stderr, "ERROR: The file %s cannot be open\n", filename);
        return -1;
    }

    if ( tdesc->storage == matrix_Tile ) {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( ddesc->rank_of( ddesc, i, j ) == myrank ) {
                    buf = ddesc->data_of( ddesc, i, j );
                    ret = fread(buf, eltsize, tdesc->bsiz, tmpf );
                    if ( ret !=  tdesc->bsiz ) {
                        fprintf(stderr, "ERROR: The read on tile(%d, %d) read %d elements instead of %d\n", 
                                i, j, ret, tdesc->bsiz);
                        return -1;
                    }
                }
            }
    } else {        
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( ddesc->rank_of( ddesc, i, j ) == myrank ) {
                    buf = ddesc->data_of( ddesc, i, j );
                    for (k=0; k<tdesc->nb; k++) {
                        ret = fread(buf, eltsize, tdesc->mb, tmpf );
                        if ( ret !=  tdesc->mb ) {
                            fprintf(stderr, "ERROR: The read on tile(%d, %d) read %d elements instead of %d\n", 
                                    i, j, ret, tdesc->mb);
                            return -1;
                        }
                        buf += eltsize * tdesc->lm;
                    }
                }
            }
    }

    fclose(tmpf);
    return 0;
}


/*
 * Deprecated code
 */
#if 0

typedef struct tile_coordinate{
    int row;
    int col;
} tile_coordinate_t;

typedef struct info_tiles{
    int th_id;    
    tiled_matrix_desc_t * Ddesc;
    tile_coordinate_t * tiles;    
    unsigned int nb_elements;
    unsigned int starting_position;
    unsigned long long int seed;
    void (*gen_fct)( tiled_matrix_desc_t *, void *, int, int, unsigned long long int);
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
    int tiles_coord_size;      /* size of the above table */
    unsigned int c;
    int i, j;
    int pos = 0;
    pthread_t *threads = NULL;
    pthread_attr_t thread_attr;
    info_tiles_t * info_gen;
    tiles_coord_size = (Mdesc->lmt * Mdesc->lnt) / Mdesc->super.nodes; /* average number of tiles per nodes */
    tiles_coord_size = (3*tiles_coord_size)/2; /* consider imbalance in distribution */
    tiles = malloc(tiles_coord_size * sizeof(tile_coordinate_t));

    /* check which tiles to generate */
    {
        for ( j = 0 ; j < Mdesc->lnt ; j++) {
            for ( i = 0 ; i < Mdesc->lmt ; i++) {
                if(Mdesc->super.myrank == Mdesc->super.rank_of((dague_ddesc_t *)Mdesc, i, j )) {
                    if (pos == tiles_coord_size) {
                        tiles_coord_size = 2 * tiles_coord_size;
                        tiles = realloc(tiles,
                                        tiles_coord_size*sizeof(tile_coordinate_t));
                        if (NULL == tiles) {
                            perror("cannot generate random matrix\n");
                            exit(-1);
                        }                                
                    }
                    tiles[pos].row = i;
                    tiles[pos].col = j;
                    pos++;                        
                }
            }
        }
    }
    /* have 'pos' tiles to generate, knowing which ones. Now gererating them */
    j = 0;
    info_gen = malloc(sizeof(info_tiles_t) * Mdesc->super.cores);
    for ( c = 0 ; c < Mdesc->super.cores ; c++ ) {
        info_gen[c].th_id = c;
        info_gen[c].tiles = tiles;
        info_gen[c].Ddesc = Mdesc;
        info_gen[c].nb_elements = pos / Mdesc->super.cores;
        info_gen[c].starting_position = j;
        info_gen[c].seed = sed;
        j += info_gen[c].nb_elements;
        if (mtype == 1) { /* cholesky like generation (symetric, diagonal dominant) */
            if(Mdesc->mtype == matrix_RealFloat) {
                info_gen[c].gen_fct = matrix_stile_cholesky;
            } else if (Mdesc->mtype == matrix_RealDouble) {
                info_gen[c].gen_fct = matrix_dtile_cholesky;
            } else { /* unknown type */
                printf("unknown generation type: aborting generation\n");
                free (info_gen);
                free(tiles);
                return;
            }
        } else if (mtype == 0) { /* LU like generation */
            if(Mdesc->mtype == matrix_RealFloat) {
                info_gen[c].gen_fct = matrix_stile;
            } else if (Mdesc->mtype == matrix_RealDouble) {
                info_gen[c].gen_fct = matrix_dtile;
            } else { /* unknown type */
                printf("unknown generation type: aborting generation\n");
                free (info_gen);
                free(tiles);
                return;
            }
        } else if(mtype == 2) {
            info_gen[c].gen_fct = create_tile_zero;
        }
    }
    info_gen[c - 1].nb_elements += pos % Mdesc->super.cores;

    if (Mdesc->super.cores > 1) {
        pthread_attr_init(&thread_attr);
        pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
#ifdef __linux
        pthread_setconcurrency(Mdesc->super.cores);
#endif            
        threads = malloc((Mdesc->super.cores - 1) * sizeof(pthread_t));
        if (NULL == threads) {
            perror("No memory for generating matrix\n");
            exit(-1);
        }
                
        for ( c = 1 ; c < Mdesc->super.cores ; c++) {
            pthread_create( &(threads[c-1]),
                            &thread_attr,
                            (void* (*)(void*))rand_dist_tiles,
                            (void*)&(info_gen[c]));
        }
    }

    rand_dist_tiles((void*) &(info_gen[0]));

    if (Mdesc->super.cores > 1) {
        for(c = 0 ; c < Mdesc->super.cores - 1 ; c++)
            pthread_join(threads[c],NULL);
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

void generate_tiled_random_sym_mat(tiled_matrix_desc_t * Mdesc, unsigned long long int seed)
{
    rand_dist_matrix(Mdesc, 1, seed);
}

void generate_tiled_random_mat(tiled_matrix_desc_t * Mdesc, unsigned long long int seed)
{
    rand_dist_matrix(Mdesc, 0, seed);
}


void pddiagset(tiled_matrix_desc_t * Mdesc, double val)
{
    int i, j;
    int target;
    double * buffer;
    target = (Mdesc->lmt < Mdesc->lnt) ? Mdesc->lmt : Mdesc->lnt;

    for( i = 0 ; i < target ; i++ ) {
        if(Mdesc->super.myrank == Mdesc->super.rank_of( (dague_ddesc_t *)Mdesc, i, i )) {
            buffer = Mdesc->super.data_of( (dague_ddesc_t *)Mdesc, i, i );
            for( j = 0 ; j < Mdesc->nb ; j++) {
                buffer[(j * Mdesc->mb) + j] = val;
            }
        }
    }
    return;
}
#endif

