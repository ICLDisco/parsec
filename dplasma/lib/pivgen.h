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

struct qr_subpiv_s;
typedef struct qr_subpiv_s qr_subpiv_t;

struct qr_piv_s {
    /*
     * currpiv
     *    @param[in] arg pointer to the qr_piv structure
     *    @param[in] m   line you want to eliminate
     *    @param[in] k   step in the factorization
     * 
     *  @return the annihilator p used with m at step k
     */
    int (*currpiv)(const qr_piv_t *arg, const int m, const int k);
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
    int (*nextpiv)(const qr_piv_t *arg, const int p, const int k, const int m);
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
    int (*prevpiv)(const qr_piv_t *arg, const int p, const int k, const int m);
    tiled_matrix_desc_t *desc; /* Descriptor of the matrix to factorize */
    int a;       /* Height of the TS domain */
    int p;       /* Parameter related to the cyclic-distrbution (can be different from the real p) */
    qr_subpiv_t *llvl;
    qr_subpiv_t *hlvl;
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
    int ldd;
    int a;       /* Height of the TS domain */
};

qr_piv_t *dplasma_pivgen_init( int type_llvl, int type_hlvl, tiled_matrix_desc_t *A );
void      dplasma_pivgen_finalize( qr_piv_t *qrpiv );

#endif
