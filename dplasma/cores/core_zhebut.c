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
void BFT_zQTL( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans, int is_diagonal );
void BFT_zQBL( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans);
void BFT_zQTR_trans( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans);
void BFT_zQTR( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans);
void BFT_zQBR( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans, int is_diagonal );

void RBMM_zTOP( int mb, int nb, int lda, int i_seg, int lvl, int N, int trans,
          PLASMA_Complex64_t *top, PLASMA_Complex64_t *btm,
          PLASMA_Complex64_t *C );
void RBMM_zBTM( int mb, int nb, int lda, int i_seg, int lvl, int N, int trans,
          PLASMA_Complex64_t *top, PLASMA_Complex64_t *btm,
          PLASMA_Complex64_t *C );

/* Bodies of kernels for the butterfly transformation */

void BFT_zQTL( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans, int is_diagonal )
{
    int i, j;
    printf("BFT_zQTL()\n");
    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                PLASMA_Complex64_t ri = U_but_vec[lvl*N+i_seg+i];
                PLASMA_Complex64_t rj = U_but_vec[lvl*N+j_seg+j];
                printf("HE A[%d][%d] = U_but_vec[%d]*((tl[%d]+bl[%d]) + (tr[%d]+br[%d])) * U_but_vec[%d]\n", i_seg+i, j_seg+j, lvl*N+i_seg+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, lvl*N+j_seg+j);
                fflush(stdout);
                C[j*lda+i] = ri * ((tl[j*lda+i]+bl[j*lda+i]) + (tr[i*lda+j]+br[j*lda+i])) * rj;
                printf("HE %lf %lf %lf %lf %lf %lf %lf\n", C[j*lda+i], ri, tl[j*lda+i], bl[j*lda+i], tr[i*lda+j], br[j*lda+i], rj);
                fflush(stdout);
            }    
        }
    }else{
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                PLASMA_Complex64_t ri = U_but_vec[lvl*N+i_seg+i];
                PLASMA_Complex64_t rj = U_but_vec[lvl*N+j_seg+j];
                printf("GE A[%d][%d] = U_but_vec[%d]*((tl[%d]+bl[%d]) + (tr[%d]+br[%d])) * U_but_vec[%d]\n", i_seg+i, j_seg+j, lvl*N+i_seg+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, lvl*N+j_seg+j);
                fflush(stdout);
                C[j*lda+i] = ri * ((tl[j*lda+i]+bl[j*lda+i]) + (tr[j*lda+i]+br[j*lda+i])) * rj;
                printf("GE %lf %lf %lf %lf %lf %lf %lf\n", C[j*lda+i], ri, tl[j*lda+i], bl[j*lda+i], tr[j*lda+i], br[j*lda+i], rj);
                fflush(stdout);
            }
        }
    }
    return;
}

void BFT_zQBL( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans)
{
    int i, j;
    printf("BFT_zQBL()\n");
    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t si = U_but_vec[lvl*N+i_seg+N/2+i];
                PLASMA_Complex64_t rj = U_but_vec[lvl*N+j_seg+j];
                printf("HE A[%d][%d] = U_but_vec[%d]*((tl[%d]-bl[%d]) + (tr[%d]-br[%d])) * U_but_vec[%d]\n", i_seg+i+N/2, j_seg+j, lvl*N+i_seg+N/2+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, lvl*N+j_seg+j);
                fflush(stdout);
                C[j*lda+i] = si * ((tl[j*lda+i]-bl[j*lda+i]) + (tr[i*lda+j]-br[j*lda+i])) * rj;
                printf("HE %lf %lf %lf %lf %lf %lf %lf\n",C[j*lda+i], si, tl[j*lda+i], bl[j*lda+i], tr[i*lda+j], br[j*lda+i], rj);
                fflush(stdout);
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t si = U_but_vec[lvl*N+i_seg+N/2+i];
                PLASMA_Complex64_t rj = U_but_vec[lvl*N+j_seg+j];
                printf("GE A[%d][%d] = U_but_vec[%d]*((tl[%d]-bl[%d]) + (tr[%d]-br[%d])) * U_but_vec[%d]\n", i_seg+i+N/2, j_seg+j, lvl*N+i_seg+N/2+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, lvl*N+j_seg+j);
                fflush(stdout);
                C[j*lda+i] = si * ((tl[j*lda+i]-bl[j*lda+i]) + (tr[j*lda+i]-br[j*lda+i])) * rj;
                printf("GE %lf %lf %lf %lf %lf %lf %lf\n", C[j*lda+i], si, tl[j*lda+i], bl[j*lda+i], tr[j*lda+i], br[j*lda+i], rj);
                fflush(stdout);
            }
        }
    }
    return;
}

/* This function writes into a transposed tile, so C is always transposed. */
void BFT_zQTR_trans( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans )
{
    int i,j;
    printf("BFT_zQTR_trans()\n");
    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t ri = U_but_vec[lvl*N+i_seg+i];
                PLASMA_Complex64_t sj = U_but_vec[lvl*N+j_seg+N/2+j];
                printf("HE A[%d][%d] = U_but_vec[%d]*((tl[%d]+bl[%d]) - (tr[%d]+br[%d])) * U_but_vec[%d]\n", i_seg+j, j_seg+i+N/2, lvl*N+i_seg+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, lvl*N+j_seg+N/2+j);
                fflush(stdout);
                C[i*lda+j] = ri * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[i*lda+j]+br[j*lda+i])) * sj;
                printf("HE %lf %lf %lf %lf %lf %lf %lf\n", C[i*lda+j], ri, tl[j*lda+i], bl[j*lda+i], tr[i*lda+j], br[j*lda+i], sj);
                fflush(stdout);
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t ri = U_but_vec[lvl*N+i_seg+i];
                PLASMA_Complex64_t sj = U_but_vec[lvl*N+j_seg+N/2+j];
                printf("GE A[%d][%d] = U_but_vec[%d]*((tl[%d]+bl[%d]) - (tr[%d]+br[%d])) * U_but_vec[%d]\n", i_seg+j, j_seg+i+N/2, lvl*N+i_seg+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, lvl*N+j_seg+N/2+j);
                fflush(stdout);
                C[i*lda+j] = ri * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[j*lda+i]+br[j*lda+i])) * sj;
                printf("GE %lf %lf %lf %lf %lf %lf %lf\n", C[i*lda+j], ri, tl[j*lda+i], bl[j*lda+i], tr[j*lda+i], br[j*lda+i], sj);
                fflush(stdout);
            }
        }
    }
    return;
}

void BFT_zQTR( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans )
{
    int i, j;
    printf("BFT_zQTR()\n");
    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t ri = U_but_vec[lvl*N+i_seg+i];
                PLASMA_Complex64_t sj = U_but_vec[lvl*N+j_seg+N/2+j];
                printf("HE A[%d][%d] = U_but_vec[%d]*((tl[%d]+bl[%d]) - (tr[%d]+br[%d])) * U_but_vec[%d]\n", i_seg+i, j_seg+j+N/2, lvl*N+i_seg+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, lvl*N+j_seg+N/2+j);
                fflush(stdout);
                C[j*lda+i] = ri * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[i*lda+j]+br[j*lda+i])) * sj;
                printf("HE %lf %lf %lf %lf %lf %lf %lf\n", C[j*lda+i], ri, tl[j*lda+i], bl[j*lda+i], tr[i*lda+j], br[j*lda+i], sj);
                fflush(stdout);
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                PLASMA_Complex64_t ri = U_but_vec[lvl*N+i_seg+i];
                PLASMA_Complex64_t sj = U_but_vec[lvl*N+j_seg+N/2+j];
                printf("GE A[%d][%d] = U_but_vec[%d]*((tl[%d]+bl[%d]) - (tr[%d]+br[%d])) * U_but_vec[%d]\n", i_seg+i, j_seg+j+N/2, lvl*N+i_seg+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, lvl*N+j_seg+N/2+j);
                fflush(stdout);
                C[j*lda+i] = ri * ((tl[j*lda+i]+bl[j*lda+i]) - (tr[j*lda+i]+br[j*lda+i])) * sj;
                printf("GE %lf %lf %lf %lf %lf %lf %lf\n", C[j*lda+i], ri, tl[j*lda+i], bl[j*lda+i], tr[j*lda+i], br[j*lda+i], sj);
                fflush(stdout);
            }
        }
    }
    return;
}

void BFT_zQBR( int mb, int nb, int lda, int i_seg, int j_seg, int lvl, int N,
          PLASMA_Complex64_t *tl, PLASMA_Complex64_t *bl,
          PLASMA_Complex64_t *tr, PLASMA_Complex64_t *br,
          PLASMA_Complex64_t *C, int bl_is_tr_trans, int is_diagonal )
{
    int i, j;
    printf("BFT_zQBR()\n");
    if( bl_is_tr_trans ){
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                PLASMA_Complex64_t si = U_but_vec[lvl*N+i_seg+N/2+i];
                PLASMA_Complex64_t sj = U_but_vec[lvl*N+j_seg+N/2+j];
                printf("HE A[%d][%d] = U_but_vec[%d]*((tl[%d]-bl[%d]) - (tr[%d]-br[%d])) * U_but_vec[%d]\n", i_seg+i+N/2, j_seg+j+N/2, lvl*N+i_seg+N/2+i, j*lda+i, j*lda+i, i*lda+j, j*lda+i, lvl*N+j_seg+N/2+j);
                fflush(stdout);
                C[j*lda+i] = si * ((tl[j*lda+i]-bl[j*lda+i]) - (tr[i*lda+j]-br[j*lda+i])) * sj;
                printf("HE %lf %lf %lf %lf %lf %lf %lf\n", C[j*lda+i], si, tl[j*lda+i], bl[j*lda+i], tr[i*lda+j], br[j*lda+i], sj);
                fflush(stdout);
            }
        }
    }else{
        for (j=0; j<nb; j++) {
            int start = is_diagonal ? j : 0;
            for (i=start; i<mb; i++) {
                PLASMA_Complex64_t si = U_but_vec[lvl*N+i_seg+N/2+i];
                PLASMA_Complex64_t sj = U_but_vec[lvl*N+j_seg+N/2+j];
                printf("GE A[%d][%d] = U_but_vec[%d]*((tl[%d]-bl[%d]) - (tr[%d]-br[%d])) * U_but_vec[%d]\n", i_seg+i+N/2, j_seg+j+N/2, lvl*N+i_seg+N/2+i, j*lda+i, j*lda+i, j*lda+i, j*lda+i, lvl*N+j_seg+N/2+j);
                fflush(stdout);
                C[j*lda+i] = si * ((tl[j*lda+i]-bl[j*lda+i]) - (tr[j*lda+i]-br[j*lda+i])) * sj;
                printf("GE %lf %lf %lf %lf %lf %lf %lf\n", C[j*lda+i], si, tl[j*lda+i], bl[j*lda+i], tr[j*lda+i], br[j*lda+i], sj);
                fflush(stdout);
            }
        }
    }
    return;
}



void RBMM_zTOP( int mb, int nb, int lda, int i_seg, int lvl, int N, int trans,
          PLASMA_Complex64_t *top, PLASMA_Complex64_t *btm,
          PLASMA_Complex64_t *C )
{
    int i, j;
    printf("RBMM_zTOP()\n");
    for (j=0; j<nb; j++) {
        for (i=0; i<mb; i++) {
            PLASMA_Complex64_t r = U_but_vec[lvl*N+i_seg+i];
            if( PlasmaConjTrans == trans ){
                printf("C[%d] = U_but_vec[%d]*(top[%d]+btm[%d])\n", j*lda+i, lvl*N+i_seg+i, j*lda+i, j*lda+i);
                fflush(stdout);
                C[j*lda+i] = r * (top[j*lda+i] + btm[j*lda+i]);
                printf("*C:%lf r:%lf *top:%lf *btm:%lf\n",*(double *)C, (double)r, *(double *)top, *(double *)btm);
                fflush(stdout);
            }else{
                PLASMA_Complex64_t s = U_but_vec[lvl*N+i_seg+N/2+i];
                printf("C[%d] = U_but_vec[%d]*top[%d] + U_but_vec[%d]*btm[%d]\n", j*lda+i, lvl*N+i_seg+i, j*lda+i, lvl*N+i_seg+N/2+i, j*lda+i);
                fflush(stdout);
                C[j*lda+i] =  r*top[j*lda+i] + s*btm[j*lda+i];
                printf("*C:%lf r:%lf *top:%lf s:%lf *btm:%lf\n",*(double *)C, (double)r, *(double *)top, (double)s, *(double *)btm);
                fflush(stdout);
            }
        }
    }
    return;
}

void RBMM_zBTM( int mb, int nb, int lda, int i_seg, int lvl, int N, int trans,
          PLASMA_Complex64_t *top, PLASMA_Complex64_t *btm,
          PLASMA_Complex64_t *C )
{
    int i, j;
    printf("RBMM_zBTM()\n");
    for (j=0; j<nb; j++) {
        for (i=0; i<mb; i++) {
            PLASMA_Complex64_t s = U_but_vec[lvl*N+i_seg+N/2+i];
            if( PlasmaConjTrans == trans ){
                printf("C[%d] = U_but_vec[%d]*(top[%d]-btm[%d])\n", j*lda+i, lvl*N+i_seg+N/2+i, j*lda+i, j*lda+i);
                fflush(stdout);
                C[j*lda+i] = s * (top[j*lda+i] - btm[j*lda+i]);
                printf("*C:%lf s:%lf *top:%lf *btm:%lf\n",*(double *)C, (double)s, *(double *)top, *(double *)btm);
                fflush(stdout);
            }else{
                PLASMA_Complex64_t r = U_but_vec[lvl*N+i_seg+i];
                printf("C[%d] = U_but_vec[%d]*top[%d] - U_but_vec[%d]*btm[%d]\n", j*lda+i, lvl*N+i_seg+i, j*lda+i, lvl*N+i_seg+i, j*lda+i);
                fflush(stdout);
                C[j*lda+i] = r*top[j*lda+i] - s*btm[j*lda+i];
                printf("*C:%lf r:%lf *top:%lf s:%lf *btm:%lf\n",*(double *)C, (double)r, *(double *)top, (double)s, *(double *)btm);
                fflush(stdout);
            }
        }
    }
    return;
}
