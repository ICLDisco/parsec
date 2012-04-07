/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"


int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M, (DagDouble_t)N))

    if ( M != N && check ) {
        fprintf(stderr, "Check cannot be perfomed with M != N\n");
        check = 0;
    }

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 7657);

    if(loud > 2) printf("Done\n");

    if(loud > 2) printf("+++ Computing getrf_sp ... ");

    double ret = dplasma_zlange_inf(dague,PlasmaInfNorm, (tiled_matrix_desc_t *)&ddescA);
    printf("The infini norm of A is %g\n",ret);

    if(check)
    {
        Dague_Complex64_t *A;
        A = (Dague_Complex64_t *)malloc((ddescA.super.lm)*(ddescA.super.n)*sizeof(Dague_Complex64_t));

        twoDBC_ztolapack( &ddescA, A, LDA );

        double *work  = (double *)malloc(M* sizeof(double));
        double ret_lapacke = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', M, N, A, LDA, work);
        printf("The infini Lapacke norm of A is %g\n",ret_lapacke);
        printf("The solution is %s\n",(ret == ret_lapacke)?"correct":"bad");
        free(A);
    }

    if(loud > 2) printf("Done.\n");


    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);

    cleanup_dague(dague, iparam);

    return EXIT_SUCCESS;
}
