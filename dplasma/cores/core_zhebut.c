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
//#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include <core_blas.h>
#include <cblas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

/*
#define DEBUG_BUTTERFLY
*/

/* Forward declaration of kernels for the butterfly transformation */
void BFT_zQTL( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans, int is_diagonal );

void BFT_zQBL( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans, int is_diagonal );
void BFT_zQTR_trans( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans );
void BFT_zQTR( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans);
void BFT_zQBR( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans, int is_diagonal );

void RBMM_zTOP( int mb, int nb, int lda, int i_seg, int lvl, int N, int trans,
          PLASMA_Complex64_t *top, PLASMA_Complex64_t *btm,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_but_vec);
void RBMM_zBTM( int mb, int nb, int lda, int i_seg, int lvl, int N, int trans,
          PLASMA_Complex64_t *top, PLASMA_Complex64_t *btm,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_but_vec);

/* Bodies of kernels for the butterfly transformation */

void BFT_zQTL( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans, int is_diagonal )
{
    int i, j;

    (void)lvl;
    (void)N;

#if defined(DEBUG_BUTTERFLY)
    printf ("BFT_zQTL(mb:%d, nb:%d, lda:%d, i_seg:%d, j_seg:%d, lvl:%d, N:%d, bl_is_tr_trans:%d, is_diagonal:%d)\n",
            mb, nb, lda, i_seg, j_seg, lvl, N, bl_is_tr_trans, is_diagonal);
#endif

    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                PLASMA_Complex64_t ri = U_before[i_seg+i];
                PLASMA_Complex64_t rj = U_after[j_seg+j];
#if defined(DEBUG_BUTTERFLY)
                printf ("HE A[%d][%d] = U_before[%d]*((tl[%d]+bl[%d]) + (tr[%d]+br[%d])) * U_after[%d]\n", i_seg+i, j_seg+j, i_seg+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, j_seg+j);
#endif
                C[j*lda+i] = ri * ((tl[j*lda+i]+bl[j*lda+i]) + (tr[i*lda+j]+br[j*lda+i])) * rj;
#if defined(DEBUG_BUTTERFLY)
                printf ("HE %lf %lf %lf %lf %lf %lf %lf\n", creal(C[j*lda+i]), creal(ri), creal(tl[j*lda+i]), creal(bl[j*lda+i]), creal(tr[i*lda+j]), creal(br[j*lda+i]), creal(rj));
#endif
            }    
        }
    }else{
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                PLASMA_Complex64_t ri = U_before[i_seg+i];
                PLASMA_Complex64_t rj = U_after[j_seg+j];
#if defined(DEBUG_BUTTERFLY)
                printf ("GE A[%d][%d] = U_before[%d]*((tl[%d]+bl[%d]) + (tr[%d]+br[%d])) * U_after[%d]\n", i_seg+i, j_seg+j, i_seg+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, j_seg+j);
#endif
                C[j*lda+i] = ri * ((tl[j*lda+i]+bl[j*lda+i]) + (tr[j*lda+i]+br[j*lda+i])) * rj;
#if defined(DEBUG_BUTTERFLY)
                printf ("GE %lf %lf %lf %lf %lf %lf %lf\n", creal(C[j*lda+i]), creal(ri), creal(tl[j*lda+i]), creal(bl[j*lda+i]), creal(tr[j*lda+i]), creal(br[j*lda+i]), creal(rj));
#endif
            }
        }
    }
    return;
}

void BFT_zQBL( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans, int is_diagonal )
{
    int i, j;
    int r_to_s = N/(1<<(lvl+1));

#if defined(DEBUG_BUTTERFLY)
    if( is_diagonal )
        printf ("BFT_zQBL(diag)\n");
    else
        printf ("BFT_zQBL(lower)\n");
#endif

    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t si = U_before[i_seg+r_to_s+i];
                PLASMA_Complex64_t rj = U_after[j_seg+j];
#if defined(DEBUG_BUTTERFLY)
                printf ("HE A[%d][%d] = U_before[%d]*((tl[%d]-bl[%d]) + (tr[%d]-br[%d])) * U_after[%d]\n", i_seg+i+r_to_s, j_seg+j, i_seg+r_to_s+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, j_seg+j);
#endif
                if( is_diagonal && j>i ){ /* If the tile of the TL quarter is on the diagonal, then we need to take the transpose element for the tiles that came from TL and BR */
                    C[j*lda+i] = si * ((tl[i*lda+j]-bl[j*lda+i]) + (tr[i*lda+j]-br[i*lda+j])) * rj;
                } else{
                    C[j*lda+i] = si * ((tl[j*lda+i]-bl[j*lda+i]) + (tr[i*lda+j]-br[j*lda+i])) * rj;
                }
#if defined(DEBUG_BUTTERFLY)
                printf ("HE %lf %lf %lf %lf %lf %lf %lf\n",creal(C[j*lda+i]), creal(si), creal(tl[j*lda+i]), creal(bl[j*lda+i]), creal(tr[i*lda+j]), creal(br[j*lda+i]), creal(rj));
#endif
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t si = U_before[i_seg+r_to_s+i];
                PLASMA_Complex64_t rj = U_after[j_seg+j];
#if defined(DEBUG_BUTTERFLY)
                printf ("GE A[%d][%d] = U_before[%d]*((tl[%d]-bl[%d]) + (tr[%d]-br[%d])) * U_after[%d]\n", i_seg+i+r_to_s, j_seg+j, i_seg+r_to_s+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, j_seg+j);
#endif
                C[j*lda+i] = si * ((tl[j*lda+i]-bl[j*lda+i]) + (tr[j*lda+i]-br[j*lda+i])) * rj;
#if defined(DEBUG_BUTTERFLY)
                printf ("GE %lf %lf %lf %lf %lf %lf %lf\n", creal(C[j*lda+i]), creal(si), creal(tl[j*lda+i]), creal(bl[j*lda+i]), creal(tr[j*lda+i]), creal(br[j*lda+i]), creal(rj));
#endif
            }
        }
    }
    return;
}

/* This function writes into a transposed tile, so C is always transposed. */
void BFT_zQTR_trans( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans )
{
    int i,j;
    int r_to_s = N/(1<<(lvl+1));
#if defined(DEBUG_BUTTERFLY)
    printf ("BFT_zQTR_trans()\n");
#endif
    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t ri = U_before[i_seg+i];
                PLASMA_Complex64_t sj = U_after[j_seg+r_to_s+j];
#if defined(DEBUG_BUTTERFLY)
                printf ("HE A[%d][%d] = U_before[%d]*((tl[%d]+bl[%d]) - (tr[%d]+br[%d])) * U_after[%d]\n", i_seg+j, j_seg+i+r_to_s, i_seg+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, j_seg+r_to_s+j);
#endif
                C[i*lda+j] = ri * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[i*lda+j]+br[j*lda+i])) * sj;
#if defined(DEBUG_BUTTERFLY)
                printf ("HE %lf %lf %lf %lf %lf %lf %lf\n", creal(C[i*lda+j]), creal(ri), creal(tl[j*lda+i]), creal(bl[j*lda+i]), creal(tr[i*lda+j]), creal(br[j*lda+i]), creal(sj));
#endif
            }
        }
    }else{
        assert(0); // This should never happen.
    }
    return;
}

void BFT_zQTR( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans )
{
    int i, j;
    int r_to_s = N/(1<<(lvl+1));
#if defined(DEBUG_BUTTERFLY)
    printf ("BFT_zQTR()\n");
#endif
    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t ri = U_before[i_seg+i];
                PLASMA_Complex64_t sj = U_after[j_seg+r_to_s+j];
#if defined(DEBUG_BUTTERFLY)
                printf ("HE A[%d][%d] = U_before[%d]*((tl[%d]+bl[%d]) - (tr[%d]+br[%d])) * U_after[%d]\n", i_seg+i, j_seg+j+r_to_s, i_seg+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, j_seg+r_to_s+j);
#endif
                C[j*lda+i] = ri * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[i*lda+j]+br[j*lda+i])) * sj;
#if defined(DEBUG_BUTTERFLY)
                printf ("HE %lf %lf %lf %lf %lf %lf %lf\n", creal(C[j*lda+i]), creal(ri), creal(tl[j*lda+i]), creal(bl[j*lda+i]), creal(tr[i*lda+j]), creal(br[j*lda+i]), creal(sj));
#endif
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t ri = U_before[i_seg+i];
                PLASMA_Complex64_t sj = U_after[j_seg+r_to_s+j];
#if defined(DEBUG_BUTTERFLY)
                printf ("GE A[%d][%d] = U_before[%d]*((tl[%d]+bl[%d]) - (tr[%d]+br[%d])) * U_after[%d]\n", i_seg+i, j_seg+j+r_to_s, i_seg+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, j_seg+r_to_s+j);
#endif
                C[j*lda+i] = ri * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[j*lda+i]+br[j*lda+i])) * sj;
#if defined(DEBUG_BUTTERFLY)
                printf ("GE %lf %lf %lf %lf %lf %lf %lf\n", creal(C[j*lda+i]), creal(ri), creal(tl[j*lda+i]), creal(bl[j*lda+i]), creal(tr[j*lda+i]), creal(br[j*lda+i]), creal(sj));
#endif
            }
        }
    }
    return;
}

void BFT_zQBR( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_before, PLASMA_Complex64_t *U_after, int bl_is_tr_trans, int is_diagonal )
{
    int i, j;
    int r_to_s = N/(1<<(lvl+1));

#if defined(DEBUG_BUTTERFLY)
    if( is_diagonal )
        printf ("BFT_zQBR(diag)\n");
    else
        printf ("BFT_zQBR(lower)\n");
#endif

    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                PLASMA_Complex64_t si = U_before[i_seg+r_to_s+i];
                PLASMA_Complex64_t sj = U_after[j_seg+r_to_s+j];
#if defined(DEBUG_BUTTERFLY)
                printf ("HE A[%d][%d] = U_before[%d]*((tl[%d]-bl[%d]) - (tr[%d]-br[%d])) * U_after[%d]\n", i_seg+i+r_to_s, j_seg+j+r_to_s, i_seg+r_to_s+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, j_seg+r_to_s+j);
#endif
                C[j*lda+i] = si * ((tl[j*lda+i]-bl[j*lda+i]) - (tr[i*lda+j]-br[j*lda+i])) * sj;
#if defined(DEBUG_BUTTERFLY)
                printf ("HE %lf %lf %lf %lf %lf %lf %lf\n", creal(C[j*lda+i]), creal(si), creal(tl[j*lda+i]), creal(bl[j*lda+i]), creal(tr[i*lda+j]), creal(br[j*lda+i]), creal(sj));
#endif
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                PLASMA_Complex64_t si = U_before[i_seg+r_to_s+i];
                PLASMA_Complex64_t sj = U_after[j_seg+r_to_s+j];
#if defined(DEBUG_BUTTERFLY)
                printf ("GE A[%d][%d] = U_before[%d]*((tl[%d]-bl[%d]) - (tr[%d]-br[%d])) * U_after[%d]\n", i_seg+i+r_to_s, j_seg+j+r_to_s, i_seg+r_to_s+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, j_seg+r_to_s+j);
#endif
                C[j*lda+i] = si * ((tl[j*lda+i]-bl[j*lda+i]) - (tr[j*lda+i]-br[j*lda+i])) * sj;
#if defined(DEBUG_BUTTERFLY)
                printf ("GE %lf %lf %lf %lf %lf %lf %lf\n", creal(C[j*lda+i]), creal(si), creal(tl[j*lda+i]), creal(bl[j*lda+i]), creal(tr[j*lda+i]), creal(br[j*lda+i]), creal(sj));
#endif
            }
        }
    }
    return;
}



void RBMM_zTOP( int mb, int nb, int lda, int i_seg, int lvl, int N, int trans,
          PLASMA_Complex64_t *top, PLASMA_Complex64_t *btm,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_but_vec)
{
    int i, j;
    int r_to_s = N/(1<<(lvl+1));
#if defined(DEBUG_BUTTERFLY)
    printf ("RBMM_zTOP()\n");
#endif
    for (j=0; j<nb; j++) {
        for (i=0; i<mb; i++) {
            PLASMA_Complex64_t r = U_but_vec[i_seg+i];
            if( PlasmaConjTrans == trans ){
#if defined(DEBUG_BUTTERFLY)
                printf ("C[%d] = U_but_vec[%d]*(top[%d]+btm[%d])\n", j*lda+i, i_seg+i, j*lda+i, j*lda+i);
#endif
                C[j*lda+i] = r * (top[j*lda+i] + btm[j*lda+i]);
#if defined(DEBUG_BUTTERFLY)
                printf ("%lf %lf %lf %lf\n",creal(C[j*lda+i]), creal(r), creal(top[j*lda+i]), creal(btm[j*lda+i]));
#endif
            }else{
                PLASMA_Complex64_t s = U_but_vec[i_seg+r_to_s+i];
#if defined(DEBUG_BUTTERFLY)
                printf ("C[%d] = U_but_vec[%d]*top[%d] + U_but_vec[%d]*btm[%d]\n", j*lda+i, i_seg+i, j*lda+i, i_seg+r_to_s+i, j*lda+i);
#endif
                C[j*lda+i] =  r*top[j*lda+i] + s*btm[j*lda+i];
#if defined(DEBUG_BUTTERFLY)
                printf ("%lf %lf %lf %lf %lf\n",creal(C[j*lda+i]), creal(r), creal(top[j*lda+i]), creal(s), creal(btm[j*lda+i]));
#endif
            }
        }
    }
    return;
}

void RBMM_zBTM( int mb, int nb, int lda, int i_seg, int lvl, int N, int trans,
          PLASMA_Complex64_t *top, PLASMA_Complex64_t *btm,
          PLASMA_Complex64_t *C, PLASMA_Complex64_t *U_but_vec)
{
    int i, j;
    int r_to_s = N/(1<<(lvl+1));
#if defined(DEBUG_BUTTERFLY)
    printf ("RBMM_zBTM()\n");
#endif
    for (j=0; j<nb; j++) {
        for (i=0; i<mb; i++) {
            PLASMA_Complex64_t s = U_but_vec[i_seg+r_to_s+i];
            if( PlasmaConjTrans == trans ){
#if defined(DEBUG_BUTTERFLY)
                printf ("C[%d] = U_but_vec[%d]*(top[%d]-btm[%d])\n", j*lda+i, i_seg+r_to_s+i, j*lda+i, j*lda+i);
#endif
                C[j*lda+i] = s * (top[j*lda+i] - btm[j*lda+i]);
#if defined(DEBUG_BUTTERFLY)
                printf ("%lf %lf %lf %lf\n",creal(C[j*lda+i]), creal(s), creal(top[j*lda+i]), creal(btm[j*lda+i]));
#endif
            }else{
                PLASMA_Complex64_t r = U_but_vec[i_seg+i];
#if defined(DEBUG_BUTTERFLY)
                printf ("C[%d] = U_but_vec[%d]*top[%d] - U_but_vec[%d]*btm[%d]\n", j*lda+i, i_seg+i, j*lda+i, i_seg+r_to_s+i, j*lda+i);
#endif
                C[j*lda+i] = r*top[j*lda+i] - s*btm[j*lda+i];
#if defined(DEBUG_BUTTERFLY)
                printf ("%lf %lf %lf %lf %lf\n", creal(C[j*lda+i]), creal(r), creal(top[j*lda+i]), creal(s), creal(btm[j*lda+i]));
#endif
            }
        }
    }
    return;
}
