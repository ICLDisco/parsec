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
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/diag_band_to_rect.h"

/* TODO: need to correct... Including the bulge chasing */
//#define FADDS_ZGEBRD(__n) (((__n) * (-8.0 / 3.0 + (__n) * (1.0 + 2.0 / 3.0 * (__n)))) - 4.0)
//#define FMULS_GEBRD(__n) (((__n) * (-1.0 / 6.0 + (__n) * (5.0 / 2.0 + 2.0 / 3.0 * (__n)))) - 15.0)



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
    anorm = dplasma_zsqrt( anorm );
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

static int check_solution(int, double*, double*, double);

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    PLASMA_desc *plasmaDescA;
    PLASMA_desc *plasmaDescT;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 144, 144);
#if defined(PARSEC_HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    //PASTE_CODE_FLOPS_COUNT(FADDS_ZHERBT, FMULS_ZHERBT, ((DagDouble_t)N))

    LDA = max(M, LDA);
    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                    nodes, rank, MB, NB, LDA, N, 0, 0,
                                    M, N, MB, NB, P))
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1,
        two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, matrix_Tile,
                                    nodes, rank, IB, NB, MT*IB, N, 0, 0,
                                    MT*IB, N, MB, NB, P))
    PASTE_CODE_ALLOCATE_MATRIX(ddescBAND, 1,
        two_dim_block_cyclic, (&ddescBAND, matrix_ComplexDouble, matrix_Tile,
        nodes, rank, MB+1, NB+2, MB+1, (NB+2)*NT, 0, 0,
        MB+1, (NB+2)*NT, 1, SNB, 1 /* 1D cyclic */ ));

    PLASMA_Desc_Create(&plasmaDescA, ddescA.mat, PlasmaComplexDouble,
         ddescA.super.mb, ddescA.super.nb, ddescA.super.bsiz,
         ddescA.super.lm, ddescA.super.ln, ddescA.super.i, ddescA.super.j,
         ddescA.super.m, ddescA.super.n);
    PLASMA_Desc_Create(&plasmaDescT, ddescT.mat, PlasmaComplexDouble,
         ddescT.super.mb, ddescT.super.nb, ddescT.super.bsiz,
         ddescT.super.lm, ddescT.super.ln, ddescT.super.i, ddescT.super.j,
         ddescT.super.m, ddescT.super.n);

    dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescA, 3872);

    PASTE_CODE_ENQUEUE_KERNEL(parsec, zgerbb,
         (IB, *plasmaDescA, (tiled_matrix_desc_t*)&ddescA, *plasmaDescT, (tiled_matrix_desc_t*)&ddescT));
    PASTE_CODE_PROGRESS_KERNEL(parsec, zgerbb);

    SYNC_TIME_START();
    parsec_diag_band_to_rect_handle_t* PARSEC_diag_band_to_rect = parsec_diag_band_to_rect_new((two_dim_block_cyclic_t*)&ddescA, &ddescBAND,
            MT, NT, MB, NB, sizeof(matrix_ComplexDouble));
    parsec_arena_t* arena = PARSEC_diag_band_to_rect->arenas[PARSEC_diag_band_to_rect_DEFAULT_ARENA];
    dplasma_add2arena_tile(arena,
        MB*NB*sizeof(parsec_complex64_t),
        PARSEC_ARENA_ALIGNMENT_SSE,
        MPI_DOUBLE_COMPLEX, MB);
    parsec_enqueue(parsec, (parsec_handle_t*)PARSEC_diag_band_to_rect);
    parsec_context_wait(parsec);
    SYNC_TIME_PRINT(rank, ( "diag_band_to_rect N= %d NB = %d : %f s\n", N, NB, sync_time_elapsed));

    PASTE_CODE_ENQUEUE_KERNEL(parsec, zgbrdb, ((tiled_matrix_desc_t*)&ddescBAND));
    PASTE_CODE_PROGRESS_KERNEL(parsec, zgbrdb)


    if(!check)
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescT);
        if(loud > 2) printf("Done\n");

        /* Create PaRSEC */
        PASTE_CODE_ENQUEUE_KERNEL(parsec, zgeqrf,
                                  ((tiled_matrix_desc_t*)&ddescA,
                                   (tiled_matrix_desc_t*)&ddescT))

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(parsec, zgeqrf)
    }

    dplasma_zgerbb_Destruct( PARSEC_zgerbb );
    parsec_diag_band_to_rect_destroy( PARSEC_diag_band_to_rect );
    dplasma_zgbrdb_Destruct( PARSEC_zgbrdb );

    parsec_data_free(ddescBAND.mat);
    parsec_data_free(ddescA.mat);
    parsec_data_free(ddescT.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescBAND);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescT);

    cleanup_parsec(parsec, iparam);

    return EXIT_SUCCESS;
}

