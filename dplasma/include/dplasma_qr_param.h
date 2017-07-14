/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */
#ifndef _DPLASMA_QR_PARAM_H_
#define _DPLASMA_QR_PARAM_H_

BEGIN_C_DECLS

/*
 * DPLASMA_QR_KILLED_BY_TS needs to be set to 0 for all variant of QR
 * factorization to distinguish TT kernels from TS kernels in jdf
 */
typedef enum dplasma_qr_type_ {
    DPLASMA_QR_KILLED_BY_TS        = 0,
    DPLASMA_QR_KILLED_BY_LOCALTREE = 1,
    DPLASMA_QR_KILLED_BY_DOMINO    = 2,
    DPLASMA_QR_KILLED_BY_DISTTREE  = 3,
} dplasma_qr_type_e;

typedef enum dplasma_qr_tree_ {
    DPLASMA_FLAT_TREE      = 0,
    DPLASMA_GREEDY_TREE    = 1,
    DPLASMA_FIBONACCI_TREE = 2,
    DPLASMA_BINARY_TREE    = 3,
    DPLASMA_GREEDY1P_TREE  = 4,
} dplasma_qr_tree_e;

struct dplasma_qrtree_s;
typedef struct dplasma_qrtree_s dplasma_qrtree_t;

struct dplasma_qrtree_s {
    /**
     * getnbgeqrf
     *    @param[in] arg  arguments specific to the reduction tree used
     *    @param[in] k    Factorization step
     *
     * @return The number of geqrt applied to the panel k
     */
    int (*getnbgeqrf)( const dplasma_qrtree_t *arg, int k );

    /**
     * getm:
     *    @param[in] arg  arguments specific to the reduction tree used
     *    @param[in] k    Factorization step
     *    @param[in] i    Index of the geqrt applied on the panel k
     *
     * @return The row index of the i-th geqrt applied on the panel k
     */
    int (*getm)( const dplasma_qrtree_t *arg, int k, int i );

    /**
     * geti:
     *    @param[in] arg  arguments specific to the reduction tree used
     *    @param[in] k    Factorization step
     *    @param[in] m    Row index where a geqrt is applied on panel k
     *
     *  @returns the index in the list of geqrt applied to panel k
     */
    int (*geti)( const dplasma_qrtree_t *qrtree, int k, int m );
    /**
     * gettype:
     *    @param[in] arg  arguments specific to the reduction tree used
     *    @param[in] k    Factorization step
     *    @param[in] m    Row index of the one we request the type
     *
     *  @returns The type of kernel used to kill the row m at step k:
     *           - 0  if it is a TS kernel
     *           - >0 otherwise. (TT kernel)
     */
    int (*gettype)( const dplasma_qrtree_t *qrtree, int k, int m );
    /**
     * currpiv
     *    @param[in] arg  arguments specific to the reduction tree used
     *    @param[in] k    Factorization step
     *    @param[in] m    line you want to eliminate
     *
     *  @return The index of the row annihilating the row m at step k
     */
    int (*currpiv)( const dplasma_qrtree_t *qrtree, int k, int m );
    /**
     * nextpiv
     *    @param[in] arg  arguments specific to the reduction tree used
     *    @param[in] k    Factorization step
     *    @param[in] p    line currently used as an annihilator to kill the row m at step k
     *    @param[in] m    line actually annihilated by the row p at step k. (k < m <= MT)
     *          m = MT to find the first time p is used as an annihilator during step k
     *
     *  @return the next line that the row p will kill during step k
     *          desc->mt if p will never be used again as an annihilator.
     */
    int (*nextpiv)(const dplasma_qrtree_t *qrtree, int k, int p, int m);
    /**
     * prevpiv
     *    @param[in] arg  arguments specific to the reduction tree used
     *    @param[in] k    Factorization step
     *    @param[in] p    line currently used as an annihilator to kill the row m at step k
     *    @param[in] m    line actually annihilated by the row p at step k. (k < m <= MT)
     *          m = p to find the last time p has been used as an annihilator during step k
     *
     *  @return the previous line killed by the row p during step k
     *          desc->mt if p has never been used before as an annihilator.
     */
    int (*prevpiv)(const dplasma_qrtree_t *qrtree, int k, int p, int m);

    /** Descriptor infos associated to the factorization */
    int mt;
    int nt;
    /** Size of the domain where TS kernels are applied */
    int a;
    /** Size of highest level tree (distributed one) */
    int p;
    void *args;
};

int  dplasma_systolic_init( dplasma_qrtree_t *qrtree,
                            PLASMA_enum trans, parsec_tiled_matrix_dc_t *A,
                            int p, int q );
void dplasma_systolic_finalize( dplasma_qrtree_t *qrtree );

int  dplasma_svd_init( dplasma_qrtree_t *qrtree,
                       PLASMA_enum trans, parsec_tiled_matrix_dc_t *A,
                       int type_hlvl, int p, int nbcores_per_node, int ratio );

int  dplasma_hqr_init( dplasma_qrtree_t *qrtree,
                       PLASMA_enum trans, parsec_tiled_matrix_dc_t *A,
                       int type_llvl, int type_hlvl,
                       int a, int p, int domino, int tsrr );
void dplasma_hqr_finalize( dplasma_qrtree_t *qrtree );

/*
 * Debugging functions
 */
int  dplasma_qrtree_check        ( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree );
void dplasma_qrtree_print_dag    ( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, char *filename );
void dplasma_qrtree_print_type   ( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree );
void dplasma_qrtree_print_pivot  ( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree );
void dplasma_qrtree_print_nbgeqrt( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree );
void dplasma_qrtree_print_perm   ( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, int *perm );
void dplasma_qrtree_print_next_k ( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, int k );
void dplasma_qrtree_print_prev_k ( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, int k );
void dplasma_qrtree_print_geqrt_k( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, int k );

END_C_DECLS

#endif /* _DPLASMA_QR_PARAM_H_ */
