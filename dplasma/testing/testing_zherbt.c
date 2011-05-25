/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <plasma.h>
#include "common.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

/* Including the bulge chassing */
#define FADDS_ZHERBT(__n) (((__n) * (-8.0 / 3.0 + (__n) * (1.0 + 2.0 / 3.0 * (__n)))) - 4.0)
#define FMULS_ZHERBT(__n) (((__n) * (-1.0 / 6.0 + (__n) * (5.0 / 2.0 + 2.0 / 3.0 * (__n)))) - 15.0)


enum blas_order_type {
            blas_rowmajor = 101,
            blas_colmajor = 102 };

enum blas_uplo_type  {
            blas_upper = 121,
            blas_lower = 122 };

enum blas_cmach_type {
            blas_base      = 151,
            blas_t         = 152,
            blas_rnd       = 153,
            blas_ieee      = 154,
            blas_emin      = 155,
            blas_emax      = 156,
            blas_eps       = 157,
            blas_prec      = 158,
            blas_underflow = 159,
            blas_overflow  = 160,
            blas_sfmin     = 161};

enum blas_norm_type {
            blas_one_norm       = 171,
            blas_real_one_norm  = 172,
            blas_two_norm       = 173,
            blas_frobenius_norm = 174,
            blas_inf_norm       = 175,
            blas_real_inf_norm  = 176,
            blas_max_norm       = 177,
            blas_real_max_norm  = 178 };

static void
BLAS_error(char *rname, int err, int val, int x) {
  fprintf( stderr, "%s %d %d %d\n", rname, err, val, x );
  abort();
}

static
void
BLAS_zsy_norm(enum blas_order_type order, enum blas_norm_type norm,
  enum blas_uplo_type uplo, int n, const PLASMA_Complex64_t *a, int lda, double *res) {
  int i, j; double anorm, v;
  char rname[] = "BLAS_zsy_norm";

  if (order != blas_colmajor) BLAS_error( rname, -1, order, 0 );

  if (norm == blas_inf_norm) {
    anorm = 0.0;
    if (blas_upper == uplo) {
      for (i = 0; i < n; ++i) {
        v = 0.0;
        for (j = 0; j < i; ++j) {
          v += cabs( a[j + i * lda] );
        }
        for (j = i; j < n; ++j) {
          v += cabs( a[i + j * lda] );
        }
        if (v > anorm)
          anorm = v;
      }
    } else {
      BLAS_error( rname, -3, norm, 0 );
      return;
    }
  } else {
    BLAS_error( rname, -2, norm, 0 );
    return;
  }

  if (res) *res = anorm;
}

static
void
BLAS_zge_norm(enum blas_order_type order, enum blas_norm_type norm,
  int m, int n, const PLASMA_Complex64_t *a, int lda, double *res)
{
  int i, j;
  double anorm, v;
  char rname[] = "BLAS_zge_norm";

  if (order != blas_colmajor) BLAS_error( rname, -1, order, 0 );

  if (norm == blas_frobenius_norm) {
    anorm = 0.0;
    for (j = n; j; --j) {
      for (i = m; i; --i) {
        v = a[0];
        anorm += v * v;
        a++;
      }
      a += lda - m;
    }
    anorm = sqrt( anorm );
  } else if (norm == blas_inf_norm) {
    anorm = 0.0;
    for (i = 0; i < m; ++i) {
      v = 0.0;
      for (j = 0; j < n; ++j) {
        v += cabs( a[i + j * lda] );
      }
      if (v > anorm)
        anorm = v;
    }
  } else {
    BLAS_error( rname, -2, norm, 0 );
    return;
  }

  if (res) *res = anorm;
}

static
double
BLAS_dpow_di(double x, int n) {
  double rv = 1.0;

  if (n < 0) {
    n = -n;
    x = (double)1.0 / x;
  }

  for (; n; n >>= 1, x *= x) {
    if (n & 1)
      rv *= x;
  }

  return rv;
}

static
double
BLAS_dfpinfo(enum blas_cmach_type cmach) {
  double eps = 1.0, r = 1.0, o = 1.0, b = 2.0;
  int t = 53, l = 1024, m = -1021;
  char rname[] = "BLAS_dfpinfo";

  if ((sizeof eps) == sizeof(float)) {
    t = 24;
    l = 128;
    m = -125;
  } else {
    t = 53;
    l = 1024;
    m = -1021;
  }

  /* for (i = 0; i < t; ++i) eps *= half; */
  eps = BLAS_dpow_di( b, -t );
  /* for (i = 0; i >= m; --i) r *= half; */
  r = BLAS_dpow_di( b, m-1 );

  o -= eps;
  /* for (i = 0; i < l; ++i) o *= b; */
  o = (o * BLAS_dpow_di( b, l-1 )) * b;

  switch (cmach) {
    case blas_eps: return eps;
    case blas_sfmin: return r;
    default:
      BLAS_error( rname, -1, cmach, 0 );
      break;
  }
  return 0.0;
}

static int check_orthogonality(int, int, int, PLASMA_Complex64_t*, double);
static int check_reduction(int, int, int, PLASMA_Complex64_t*, PLASMA_Complex64_t*, int, PLASMA_Complex64_t*, double);
static int check_solution(int, double*, double*, double);
static int check_solution2(int, double*, double*, double);

int main(int argc, char *argv[])
{
    dague_context_t *dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_desc *plasmaDescA;
    PLASMA_desc *plasmaDescT;

     /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 144, 144);
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS_COUNT(FADDS_ZHERBT, FMULS_ZHERBT, ((DagDouble_t)N))

    LDA = max(M, LDA);
    LDB = max( LDB, N );
    SMB = 1; SNB = 1;

    PLASMA_Init(1);

    /*
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
         sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
         nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
         N, N, P, MatrixLower))
    */

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
         two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
         nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
         N, N, SMB, SNB, P))

    PLASMA_Desc_Create(&plasmaDescA, ddescA.mat, PlasmaComplexDouble, 
         ddescA.super.mb, ddescA.super.nb, ddescA.super.bsiz, 
         ddescA.super.lm, ddescA.super.ln, ddescA.super.i, ddescA.super.j, 
         ddescA.super.m, ddescA.super.n);

    /*
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1, 
         sym_two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, 
         nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
         MT*IB, N, P, MatrixLower))
    */

    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1, 
         two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, 
         nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
         MT*IB, N, SMB, SNB, P))

    PLASMA_Desc_Create(&plasmaDescT, ddescT.mat, PlasmaComplexDouble, 
         ddescT.super.mb, ddescT.super.nb, ddescT.super.bsiz, 
         ddescT.super.lm, ddescT.super.ln, ddescT.super.i, ddescT.super.j, 
         ddescT.super.m, ddescT.super.n);

    PLASMA_enum uplo = PlasmaLower;

    generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);

    PLASMA_Complex64_t *A2 = (PLASMA_Complex64_t *)malloc(LDA*N*sizeof(PLASMA_Complex64_t));
    double *W1             = (double *)malloc(N*sizeof(double));
    double *W2             = (double *)malloc(N*sizeof(double));

    if( check ) {
        /*int i, j;*/
        PLASMA_Tile_to_Lapack(plasmaDescA, (void*)A2, N);

        LAPACKE_zheev( LAPACK_COL_MAJOR,
               lapack_const(PlasmaNoVec), lapack_const(uplo), 
               N, A2, LDA, W1);

	/*
        printf("A2 avant\n");
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++) {
                //printf("%f+%fi ", creal(A2[LDA*j+i]), cimag(A2[LDA*j+i]));
                printf("%f ", A2[LDA*j+i]);
            }
            printf("\n");
        }
	printf("Eigenvalues original\n");
        for (i = 0; i < N; i++){
            printf("%f \n", W1[i]);
        }
        printf("\n");
	*/
    }

    PASTE_CODE_ENQUEUE_KERNEL(dague, zherbt, 
         (uplo, IB, *plasmaDescA, (tiled_matrix_desc_t*)&ddescA, *plasmaDescT, (tiled_matrix_desc_t*)&ddescT));

    PASTE_CODE_PROGRESS_KERNEL(dague, zherbt);

    if( check ) {
        int i, j;
        PLASMA_Tile_to_Lapack(plasmaDescA, (void*)A2, N);
        for (j = 0; j < N; j++)
            for (i = j+NB+1; i < N; i++)
                A2[LDA*j+i]=0.0;

        LAPACKE_zheev( LAPACK_COL_MAJOR,
               lapack_const(PlasmaNoVec), lapack_const(uplo), 
               N, A2, LDA, W2);

	/*
        printf("A2 apres\n");
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++) {
                //printf("%f+%fi ", creal(A2[LDA*j+i]), cimag(A2[LDA*j+i]));
                printf("%f ", A2[LDA*j+i]);
            }
            printf("\n");
        }
	
        printf("Eigenvalues computed\n");
        for (i = 0; i < N; i++){
            printf("%f \n", W2[i]);
        }
        printf("\n");
	*/

        double eps = BLAS_dfpinfo( blas_eps );
        printf("\n");
        printf("------ TESTS FOR PLASMA ZHEEV ROUTINE -------  \n");
        printf("        Size of the Matrix %d by %d\n", N, N);
        printf("\n");
        printf(" The matrix A is randomly generated for each test.\n");
        printf("============\n");
        printf(" The relative machine precision (eps) is to be %e \n",eps);
        printf(" Computational tests pass if scaled residuals are less than 60.\n");
    
        /* Check the eigen solutions */
        int info_solution = check_solution(N, W1, W2, eps);
    
        if (info_solution == 0) {
            printf("***************************************************\n");
            printf(" ---- TESTING ZHEEV ...................... PASSED !\n");
            printf("***************************************************\n");
        }
        else {
            printf("************************************************\n");
            printf(" - TESTING ZHEEV ... FAILED !\n");
            printf("************************************************\n");
        }
    }

    free(A2); free(W1); free(W2);
    dplasma_zherbt_Destruct( DAGUE_zherbt );

    dague_data_free(ddescA.mat);
    dague_data_free(ddescT.mat);
    
    cleanup_dague(dague);
        
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescT);
        
    return EXIT_SUCCESS;
}

/*--------------------------------------------------------------
 * Check the solution
 */

static int check_solution(int N, double *E1, double *E2, double eps)
{
    int info_solution, i;
    double *Residual = (double *)malloc(N*sizeof(double));
    double maxtmp;
    double maxel = fabs(fabs(E1[0])-fabs(E2[0]));
    double maxeig = fmax(fabs(E1[0]), fabs(E2[0]));
    for (i = 1; i < N; i++){
        Residual[i] = fabs(fabs(E1[i])-fabs(E2[i]));
        maxtmp      = fmax(fabs(E1[i]), fabs(E2[i]));
        maxeig      = fmax(maxtmp, maxeig);
        //printf("Residu: %f E1: %f E2: %f\n", Residual[i], E1[i], E2[i] );
        if (maxel < Residual[i])
           maxel =  Residual[i];
    }

    //printf("maxel: %.16f maxeig: %.16f \n", maxel, maxeig );

    printf(" ======================================================\n");
    printf(" | D -  eigcomputed | / (|D| ulp)      : %15.3E \n",  maxel/(maxeig*eps) );
    printf(" ======================================================\n");


    printf("============\n");
    printf("Checking the eigenvalues of A\n");
    if (isnan(maxel / eps) || isinf(maxel / eps) || ((maxel / (maxeig*eps)) > 1000.0) ) {
        //printf("isnan: %d %f %e\n", isnan(maxel / eps), maxel, eps );
        //printf("isinf: %d %f %e\n", isinf(maxel / eps), maxel, eps );
        printf("-- The eigenvalues are suspicious ! \n");
        info_solution = 1;
    }
    else{
        printf("-- The eigenvalues are CORRECT ! \n");
        info_solution = 0;
    }

    free(Residual);
    return info_solution;
}
