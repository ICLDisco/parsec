/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/************************************************************
 * distributed matrix generation
 ************************************************************/
/* affect one tile with random values  */
/*
 * @precisions normal z -> s d c
 */
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "data_distribution.h"
#include "matrix.h"

#define COMPLEX
#undef REAL

#ifndef max
#define max(a,b) ( (a) > (b) ? (a) : (b) )
#endif

/*
 Rnd64seed is a global variable but it doesn't spoil thread safety. All matrix
 generating threads only read Rnd64seed. It is safe to set Rnd64seed before
 and after any calls to create_tile(). The only problem can be caused if
 Rnd64seed is changed during the matrix generation time.
 */

//static unsigned long long int Rnd64seed = 100;
#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

static unsigned long long int
Rnd64_jump(unsigned long long int n, unsigned long long int seed ) {
  unsigned long long int a_k, c_k, ran;
  int i;

  a_k = Rnd64_A;
  c_k = Rnd64_C;

  ran = seed;
  for (i = 0; n; n >>= 1, ++i) {
    if (n & 1)
      ran = a_k * ran + c_k;
    c_k *= (a_k + 1);
    a_k *= a_k;
  }

  return ran;
}

void matrix_ztile_cholesky(tiled_matrix_desc_t * Ddesc, void * position, 
                           unsigned int row, unsigned int col, unsigned long long int seed)
{
    unsigned int i, j, first_row, first_col;
    unsigned int nb = Ddesc->nb;
    Dague_Complex64_t mn_max = (Dague_Complex64_t) max(Ddesc->n, Ddesc->m);
    Dague_Complex64_t *x = (Dague_Complex64_t*) position;
    unsigned long long int ran;

    /* These are global values of first row and column of the tile counting from 0 */
    first_row = row * nb;
    first_col = col * nb;

    memset( position, 0, mb*nb*sizeof(Dague_Complex64_t) );

    for (j = 0; j < nb; ++j) {
        if( (first_col + j) >= Ddesc->n ) /* padding for columns  */
            {
                break;
            }
#ifdef COMPLEX
        ran = Rnd64_jump( 2*(first_row + (first_col + j) * (unsigned long long int)Ddesc->m) , seed);
#else
        ran = Rnd64_jump( first_row + (first_col + j) * (unsigned long long int)Ddesc->m , seed);
#endif
        for (i = 0; i < nb; ++i) {
            if( (first_row + i) >= Ddesc->lm)/* padding for rows */
            {
                break;
            }
            x[0] = 0.5f - ran * RndF_Mul;
            ran = Rnd64_A * ran + Rnd64_C;
#ifdef COMPLEX
            x[0] += I*(0.5f - ran * RndF_Mul);
            ran = Rnd64_A * ran + Rnd64_C;
#endif
            x += 1;
        }
        x += (nb - i);
    }
    /* This is only required for Cholesky: diagonal is bumped by max(M, N) */
    if (row == col) {
        x = (Dague_Complex64_t*)position;
        for (i = 0; i < nb; ++i) {
            if( ((first_row + i) >= Ddesc->m) || ((first_col + i) >= Ddesc->n) ) /* padding for diagonal */
            {
                break;
            }
#ifdef COMPLEX
            x[i + i * nb] += mn_max - I*cimag(x[i + i * nb]);
#else
            x[i + i * nb] += mn_max;
#endif
        }
    }
}

void matrix_ztile(tiled_matrix_desc_t * Ddesc, void * position, 
                  unsigned int row, unsigned int col, unsigned long long int seed)
{
    unsigned int i, j, first_row, first_col;
    unsigned int mb = Ddesc->mb;
    unsigned int nb = Ddesc->nb;
    Dague_Complex64_t *x = (Dague_Complex64_t*)position;
    unsigned long long int ran;

    /* These are global values of first row and column of the tile counting from 0 */
    first_row = row * mb;
    first_col = col * nb;

    memset( position, 0, mb*nb*sizeof(Dague_Complex64_t) );

    for (j = 0; j < nb; ++j) {
        if( (first_col + j) >= Ddesc->n ) /* padding for columns  */
            {
                break;
            }
#ifdef COMPLEX
        ran = Rnd64_jump( 2*(first_row + (first_col + j) * (unsigned long long int)Ddesc->m) , seed);
#else
        ran = Rnd64_jump( first_row + (first_col + j) * (unsigned long long int)Ddesc->m , seed);
#endif
        for (i = 0; i < mb; ++i) {
            if( (first_row + i) >= Ddesc->m)/* padding for rows */
            {
                break;
            }
            x[0] = 0.5f - ran * RndF_Mul;
            ran = Rnd64_A * ran + Rnd64_C;
#ifdef COMPLEX
            x[0] += I*(0.5f - ran * RndF_Mul);
            ran = Rnd64_A * ran + Rnd64_C;
#endif
            x += 1;
        }
        x += (mb-i); /* padding */
    }
}

#ifdef HAVE_MPI

//#include <lapack.h>

static double lamch(void) 
{
    double eps = 1.0;

    do {
        eps /= 2.0;
    } while((1.0 + eps/2.0) != 1.0);
    //eps = lapack_zlamch(lapack_eps);
    printf("epsilon is %e\n", eps);    
    return eps;
}


void matrix_zcompare_dist_data(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b)
{
    MPI_Status status;
    Dague_Complex64_t * bufferA = NULL;
    Dague_Complex64_t * bufferB = NULL;
    Dague_Complex64_t * tmpA = malloc(a->bsiz * sizeof(Dague_Complex64_t));
    Dague_Complex64_t * tmpB = malloc(a->bsiz * sizeof(Dague_Complex64_t));

    size_t i,j;
    unsigned int k;
    uint32_t rankA, rankB;
    unsigned int count = 0;
    int diff, dc;
    double eps;
    
    eps = lamch();

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
                                        MPI_Recv(tmpA, a->bsiz, MPI_DOUBLE_COMPLEX, rankA, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
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
                                        MPI_Recv(tmpB, b->bsiz, MPI_DOUBLE_COMPLEX, rankB, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
                                        bufferB = tmpB;
                                    }
                            }
                        if(rankA < a->super.nodes)
                            {
                                diff = 0;
                                dc = 0;
                                for(k = 0 ; k < a->bsiz ; k++)
                                    if ( cabs(bufferA[k] - bufferB[k]) > eps )
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
                                MPI_Send(a->super.data_of((dague_ddesc_t *) a, i, j ), a->bsiz, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
                            }
                        if ( rankB == b->super.myrank)
                            {
                                MPI_Send(b->super.data_of((dague_ddesc_t *) b, i, j ), b->bsiz, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);                                                    
                            }
                    }
            }
    if(a->super.myrank == 0)
        printf("compared the matrices: %u difference(s)\n", count);
}

#endif
