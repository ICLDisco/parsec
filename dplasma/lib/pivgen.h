/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#ifndef _PIVGEN_H_
#define _PIVGEN_H_

struct qr_piv_s;
typedef struct qr_piv_s qr_piv_t;
struct qr_piv_s {
    int (*currpiv)(const qr_piv_t *arg, const int m, const int k);
    int (*nextpiv)(const qr_piv_t *arg, const int m, const int k, const int start);
    int (*prevpiv)(const qr_piv_t *arg, const int m, const int k, const int start);
    tiled_matrix_desc_t *desc;
    int *ipiv;
};

qr_piv_t *dplasma_pivgen_init( int type_lvl, int type_hlvl, tiled_matrix_desc_t *A );
void      dplasma_pivgen_finalize( qr_piv_t *qrpiv );

int dplasma_flat_currpiv(const qr_piv_t *arg, const int m, const int k);
int dplasma_flat_nextpiv(const qr_piv_t *arg, const int p, const int k, const int start);
int dplasma_flat_prevpiv(const qr_piv_t *arg, const int p, const int k, const int start);
void dplasma_flat_init(qr_piv_t *arg, tiled_matrix_desc_t *descA);
void dplasma_fibonacci_init(qr_piv_t *arg, tiled_matrix_desc_t *A);
void dplasma_greedy_init(qr_piv_t *arg, tiled_matrix_desc_t *A);

#endif
