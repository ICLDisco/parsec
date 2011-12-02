/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#ifndef _DPLASMA_QR_PIVGEN_H_
#define _DPLASMA_QR_PIVGEN_H_

struct qr_piv_s;
typedef struct qr_piv_s qr_piv_t;

struct qr_subpiv_s;
typedef struct qr_subpiv_s qr_subpiv_t;

struct qr_piv_s {
    tiled_matrix_desc_t *desc; /* Descriptor of the matrix to factorize */
    int a;       /* Height of the TS domain */
    int p;       /* Parameter related to the cyclic-distrbution (can be different from the real p) */
    int domino;  /* Switch to enable.disable the domino tree linking high and lw level reduction trees */
    qr_subpiv_t *llvl;
    qr_subpiv_t *hlvl;
    int *perm;
};

struct qr_subpiv_s {
    /*
     * currpiv
     *    @param[in] arg pointer to the qr_piv structure
     *    @param[in] m   line you want to eliminate
     *    @param[in] k   step in the factorization
     * 
     *  @return the annihilator p used with m at step k
     */
    int (*currpiv)(const qr_subpiv_t *arg, const int m, const int k);
    /*
     * nextpiv
     *    @param[in] arg pointer to the qr_piv structure
     *    @param[in] p   line currently used as an annihilator
     *    @param[in] k   step in the factorization
     *    @param[in] m   line actually annihilated. 
     *          m = MT to find the first time p is used as an annihilator during step k
     *            
     *  @return the next line m' using the line p as annihilator during step k
     *          desc->mt if p will never be used again as an annihilator.
     */
    int (*nextpiv)(const qr_subpiv_t *arg, const int p, const int k, const int m);
    /*
     * nextpiv
     *    @param[in] arg pointer to the qr_piv structure
     *    @param[in] p   line currently used as an annihilator
     *    @param[in] k   step in the factorization
     *    @param[in] m   line actually annihilated. 
     *          m = p to find the last time p has been used as an annihilator during step k
     *            
     *  @return the previous line m' using the line p as annihilator during step k
     *          desc->mt if p has never been used before that as an annihilator.
     */
    int (*prevpiv)(const qr_subpiv_t *arg, const int p, const int k, const int m);
    int *ipiv;
    int minMN;
    int ldd;
    int a;
    int p;
    int domino;
};

qr_piv_t *dplasma_pivgen_init( tiled_matrix_desc_t *A, int type_llvl, int type_hlvl, 
                               int a, int p, int domino );
void      dplasma_pivgen_finalize( qr_piv_t *qrpiv );

/*
 * Functions used in the jdf
 *    dplasma_qr_getnbgeqrf: returns the number of geqrt kernels to apply during the step k
 *    dplasma_qr_getm:       returns the row index of the i-th geqrt to apply during the step k
 *    dplasma_qr_geti:       returns the index of the geqrt apply to the row m
 *    dplasma_qr_gettype:    returns the type of the row m at step k
 */
int dplasma_qr_getnbgeqrf( const qr_piv_t *arg, const int k, const int gmt );
int dplasma_qr_getm(       const qr_piv_t *arg, const int k, const int i   );
int dplasma_qr_geti(       const qr_piv_t *arg, const int k, const int m   );
int dplasma_qr_gettype(    const qr_piv_t *arg, const int k, const int m   );

/*
 * dplasma_qr_currpiv
 *    @param[in] arg pointer to the qr_piv structure
 *    @param[in] m   line you want to eliminate
 *    @param[in] k   step in the factorization
 * 
 *  @return the annihilator p used with m at step k
 */
int dplasma_qr_currpiv(const qr_piv_t *arg, int m, const int k);
/*
 * dplasma_qr_nextpiv
 *    @param[in] arg pointer to the qr_piv structure
 *    @param[in] p   line currently used as an annihilator
 *    @param[in] k   step in the factorization
 *    @param[in] m   line actually annihilated. 
 *          m = MT to find the first time p is used as an annihilator during step k
 *            
 *  @return the next line m' using the line p as annihilator during step k
 *          desc->mt if p will never be used again as an annihilator.
 */
int dplasma_qr_nextpiv(const qr_piv_t *arg, int p, const int k, int m);
/*
 * dplasma_qr_nextpiv
 *    @param[in] arg pointer to the qr_piv structure
 *    @param[in] p   line currently used as an annihilator
 *    @param[in] k   step in the factorization
 *    @param[in] m   line actually annihilated. 
 *          m = p to find the last time p has been used as an annihilator during step k
 *            
 *  @return the previous line m' using the line p as annihilator during step k
 *          desc->mt if p has never been used before that as an annihilator.
 */
int dplasma_qr_prevpiv(const qr_piv_t *arg, int p, const int k, int m);

/*
 * Debugging functions
 */
int  dplasma_qr_check        ( tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_dag    ( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, char *filename );
void dplasma_qr_print_type   ( tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_pivot  ( tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_nbgeqrt( tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_next_k ( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k );
void dplasma_qr_print_prev_k ( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k );
void dplasma_qr_print_geqrt_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k );

#endif /* _DPLASMA_QR_PIVGEN_H_ */
