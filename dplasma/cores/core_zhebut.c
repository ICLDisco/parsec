/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include <math.h>
#include <stdlib.h>
#include "dague.h"
#include <plasma.h>
#include <cblas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

/*
 * U_but_vec is a concatanation of L+1 vectors (where L is the level of the 
 * butterfly) and is allocated in zhebut_wrapper.c
 */
extern PLASMA_Complex64_t *U_but_vec;

/* Forward declaration of kernels for the butterfly transformation */
void BFT_zQTL( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal );
void BFT_zQBL( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal );
void BFT_zQTR_trans( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal );
void BFT_zQTR( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal );
void BFT_zQBR( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal );

/* Bodies of kernels for the butterfly transformation */

void BFT_zQTL( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal )
{
    int i, j;
    if( is_transpose ){
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                C[j*lda+i] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]+bl[j*lda+i]) + (tr[i*lda+j]+br[j*lda+i])) * U_but_vec[lvl*N+off+j];
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                /*
                printf("C:%p, U_but_vec:%p, tl:%p, bl:%p, tr:%p, br:%p\n", C, U_but_vec, tl, bl, tr, br);
                printf("C[%d] = U_but_vec[%d]*((tl[%d]+bl[%d]) + (tr[%d]+br[%d])) * U_but_vec[%d]\n", j*lda+i, lvl*N+off+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, lvl*N+off+j);
                printf("*tl:%lf *bl:%lf *tr:%lf *br:%lf *U_but_vec:%lf\n", *(double *)tl, *(double *)bl, *(double *)tr, *(double *)br, *(double *)U_but_vec);
                */
                C[j*lda+i] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]+bl[j*lda+i]) + (tr[j*lda+i]+br[j*lda+i])) * U_but_vec[lvl*N+off+j];
                /*
                printf("%lf = %lf*((%lf+%lf) + (%lf+%lf)) * %lf\n",(double)C[j*lda+i], (double)U_but_vec[lvl*N+off+i], (double)tl[j*lda+i], (double)bl[j*lda+i], (double)tr[j*lda+i], (double)br[j*lda+i], (double)U_but_vec[lvl*N+off+j]);
                */
            }
        }
    }
    return;
}

void BFT_zQBL( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal )
{
    int i, j;
    if( is_transpose ){
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                C[j*lda+i] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]-bl[j*lda+i]) + (tr[i*lda+j]-br[j*lda+i])) * U_but_vec[lvl*N+off+j];
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                C[j*lda+i] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]-bl[j*lda+i]) + (tr[j*lda+i]-br[j*lda+i])) * U_but_vec[lvl*N+off+j];
            }
        }
    }
    return;
}

/* This function writes into a transposed tile, so C is always transposed. */
void BFT_zQTR_trans( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal )
{
    int i,j;
    if( is_transpose ){
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                C[i*lda+j] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[i*lda+j]+br[j*lda+i])) * U_but_vec[lvl*N+off+j];
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                C[i*lda+j] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[j*lda+i]+br[j*lda+i])) * U_but_vec[lvl*N+off+j];
            }
        }
    }
    return;
}

void BFT_zQTR( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal )
{
    int i, j;
    if( is_transpose ){
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                C[j*lda+i] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[i*lda+j]+br[j*lda+i])) * U_but_vec[lvl*N+off+j];
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                C[j*lda+i] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[j*lda+i]+br[j*lda+i])) * U_but_vec[lvl*N+off+j];
            }
        }
    }
    return;
}

void BFT_zQBR( int mb, int nb, int lda, int off, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int is_transpose, int is_diagonal )
{
    int i, j;
    if( is_transpose ){
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                C[j*lda+i] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]-bl[j*lda+i]) - (tr[i*lda+j]-br[j*lda+i])) * U_but_vec[lvl*N+off+j];
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                C[j*lda+i] = U_but_vec[lvl*N+off+i] * ((tl[j*lda+i]-bl[j*lda+i]) - (tr[j*lda+i]-br[j*lda+i])) * U_but_vec[lvl*N+off+j];
            }
        }
    }
    return;
}


