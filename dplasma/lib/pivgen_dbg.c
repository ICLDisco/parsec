/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 *
 * This file contains all the function to describe the dependencies
 * used in the Xgeqrf_param.jdf file.
 * The QR factorization done with this file relies on three levels:
 *     - the first one is using a flat tree with TS kernels. The
 *       height of this tree is defined by the parameter 'a'. If 'a'
 *       is set to A->mt, the factorization is identical to the one
 *       perform by PLASMA_zgeqrf.
 *       For all subdiagonal "macro-tiles", the line reduced is always the first.
 *       For all diagonal "macro-tiles", the factorization performed
 *       is identical to the one performed by PLASMA_zgeqrf.
 *
 *     - the third level is using a reduction tree of size 'p'. By
 *       default, the parameter 'p' should be equal to the number of
 *       processors used for the computation, but can be set
 *       differently. (see further example). The type of tree used at
 *       this level is defined by the hlvl parameter. It can be flat
 *       or greedy.
 *       CODE DETAILS: This tree and all the function related to it
 *       are performing a QR factorization on a band matrix with 'p'
 *       the size of the band. All the functions take global indices
 *       as input and return global indices as output.
 *
 *     - Finally, a second 'low' level of reduction tree is applied.
 *       The size of this tree is induced by the parameters 'a' and 'p'
 *       from the first and third levels and is A->mt / ( p * a ). This
 *       tree is reproduced p times for each subset of tiles
 *       S_k = {i in [0, A->mt-1] \ i%p*a = k } with k in [0, p-1].
 *       The tree used for the reduction is defined by the llvl
 *       parameter and can be: flat, greedy, fibonacci or binary.
 *       CODE DETAILS: For commodity, the size of this tree is always
 *       ceil(A->mt / (p * a) ) inducing some extra tests in the code.
 *       All the functions related to this level of tree take as input
 *       the local indices in the A->mt / (p*a) matrix and the global
 *       k. They return the local index. The reductions are so
 *       performed on a trapezoidal matrices where the step is defined
 *       by a:
 *                                    <- min( lhlvl_mt, min( mt, nt ) ) ->
 *                                     __a__   a     a
 *                                    |     |_____
 *                                    |           |_____
 *                                    |                 |_____
 *        llvl_mt = ceil(MT/ (a*p))   |                       |_____
 *                                    |                             |_____
 *                                    |___________________________________|
 *
 *
 *
 *   At each step of the factorization, the lines of tiles are divided
 *   in 4 types:
 *     - QRPARAM_TILE_TS: They are the lines annihilated by a TS
 *     kernel, these lines are never used as an annihilator.  They are
 *     the lines i, with 1 < (i/p)%a < a and i > (k+1)*p
 *     - QRPARAM_TILE_LOCALTT: They are the lines used as annhilitor
 *     in the TS kernels annihiling the QRPARAM_TILE_TS lines.  They
 *     are themselves annihilated by the TT kernel of the low level
 *     reduction tree.  The roots of the local trees are the lines i,
 *     with i/p = k.
 *     - QRPARAM_TILE_DOMINO: These are the lines that are
 *     annhilihated with a domino effect in the band defined by (i/p)
 *     <= k and i >= k
 *     - QRPARAM_TILE_DISTTT: These are the lines annihilated by the
 *     high level tree to reduce communications.
 *     These lines are defined by (i-k)/p = 0.
 */
#include <dague.h>
#include <plasma.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#include "dplasma_qr_pivgen.h"

#include <math.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */

#ifndef min
#define min(__a, __b) ( ( (__a) < (__b) ) ? (__a) : (__b) )
#endif

#ifndef max
#define max(__a, __b) ( ( (__a) > (__b) ) ? (__a) : (__b) )
#endif

/* static int dplasma_qr_getinon0( const qr_piv_t *arg,  */
/*                                 const int k, int i, int mt ); */

#define ENDCHECK( test, ret )                   \
    if ( !test )                                \
        return ret;

int dplasma_qr_check( tiled_matrix_desc_t *A, qr_piv_t *qrpiv)
{
    int minMN = min(A->mt, A->nt );
    int i, m, k, nb;
    int check;

    int a = qrpiv->a;
    int p = qrpiv->p;

    /*
     * Check Formula for NB geqrt
     */
    {
        /* dplasma_qr_print_type( A, qrpiv ); */
        /* dplasma_qr_print_nbgeqrt( A, qrpiv ); */
        check = 1;
        for (k=0; k<minMN; k++) {
            nb = 0;
            for (m=k; m < A->mt; m++) {
              if ( dplasma_qr_gettype(qrpiv, k, m) > 0 )
                    nb++;
            }

            if ( nb != dplasma_qr_getnbgeqrf( qrpiv, k, A->mt) ) {
                check = 0;
                printf(" ----------------------------------------------------\n"
                       "  - a = %d, p = %d, M = %d, N = %d\n"
                       "     Check number of geqrt:\n"
                       "       For k=%d => return %d instead of %d",
                       a, p, A->mt, A->nt, k, dplasma_qr_getnbgeqrf( qrpiv, k, A->mt), nb );
            }
        }

        ENDCHECK( check, 1 );
    }

    /*
     * Check indices of geqrt
     */
    {
        int prevm = -1;
        check = 1;
        for (k=0; k<minMN; k++) {
            /* dplasma_qr_print_geqrt_k( A, qrpiv, k ); */
            nb = dplasma_qr_getnbgeqrf( qrpiv, k, A->mt );
            prevm = -1;
            for (i=0; i < nb; i++) {

                m = dplasma_qr_getm( qrpiv, k, i );

                /*
                 * getm ahas to be the inverse of geti
                 */
                if ( i != dplasma_qr_geti( qrpiv, k, m) ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d && geti( k=%d, m=%d ) => i = %d\n",
                           a, p, A->mt, A->nt,
                           k, i, m, k, m, dplasma_qr_geti( qrpiv, k, m));
                }
                /* tile before the diagonal are factorized and
                 * the m is a growing list (not true with round-robin inside TS)
                 */
                else if ( (a == 1) && (( m < k ) || ( m < prevm )) ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d",
                           a, p, A->mt, A->nt, k, i, m);
                }
#if 0
                else if ( m != dplasma_qr_getinon0( qrpiv, k, i, A->mt ) ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d but should be %d",
                           a, p, A->mt, A->nt, k, i, m, dplasma_qr_getinon0( qrpiv, k, i, A->mt));
                }
#endif
                prevm = m;
            }
        }
        ENDCHECK( check, 2 );
    }

    /*
     * Check number of exit in next
     */
    {
        int s;
        check = 1;

        for (k=0; k<minMN; k++) {
            for(m=k; m<A->mt; m++) {
                nb = 0;
                for(s=A->mt; s>k; s--) {
                    if ( dplasma_qr_nextpiv(qrpiv, m, k, s) == A->mt )
                        nb++;
                }
                if ( nb > 1 ) {
                    dplasma_qr_print_next_k( A, qrpiv, k);
                    dplasma_qr_print_prev_k( A, qrpiv, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Next of line %d for step %d contains more than one exit:\n",
                           a, p, A->mt, A->nt,
                           m, k);
                    check = 0;
                    return 3;
                }
                else if ( nb == 0 ) {
                    dplasma_qr_print_next_k( A, qrpiv, k);
                    dplasma_qr_print_prev_k( A, qrpiv, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Next of line %d for step %d needs one exit:\n",
                           a, p, A->mt, A->nt,
                           m, k);
                    check = 0;
                    return 3;
                }
            }
        }
        ENDCHECK( check, 3 );
    }

    /*
     * Check number of exit in prev
     */
    {
        int s;
        check = 1;

        for (k=0; k<minMN; k++) {
            for(m=k; m<A->mt; m++) {
                nb = 0;
                for(s=k; s<A->mt; s++) {
                    if ( dplasma_qr_prevpiv(qrpiv, m, k, s) == A->mt )
                        nb++;
                }
                if ( nb > 1 ) {
                    dplasma_qr_print_next_k( A, qrpiv, k);
                    dplasma_qr_print_prev_k( A, qrpiv, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Prev of line %d for step %d contains more than one exit:\n",
                           a, p, A->mt, A->nt,
                           m, k);
                    check = 0;
                    return 3;
                }
                else if ( nb == 0 ) {
                    dplasma_qr_print_next_k( A, qrpiv, k);
                    dplasma_qr_print_prev_k( A, qrpiv, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Prev of line %d for step %d needs one exit:\n",
                           a, p, A->mt, A->nt,
                           m, k);
                    check = 0;
                    return 3;
                }
            }
        }
        ENDCHECK( check, 3 );
    }

    /*
     * Check next/prev
     */
    {
        int start, next, prev;
        check = 1;

        for (k=0; k<minMN; k++) {
            start = A->mt;
            for(m=k; m<A->mt; m++) {

                do {
                    next = dplasma_qr_nextpiv(qrpiv, m, k, start);
                    if ( next == A->mt )
                        prev = dplasma_qr_prevpiv(qrpiv, m, k, next);//prev = dplasma_qr_prevpiv(qrpiv, m, k, m);
                    else
                        prev = dplasma_qr_prevpiv(qrpiv, m, k, next);

                    if ( start != prev ) {
                        dplasma_qr_print_next_k( A, qrpiv, k);
                        dplasma_qr_print_prev_k( A, qrpiv, k);

                        printf(" ----------------------------------------------------\n"
                               "  - a = %d, p = %d, M = %d, N = %d\n"
                               "     Check next/prev:\n"
                               "       next( m=%d, k=%d, start=%d ) => %d && prev( m=%d, k=%d, start=%d ) => %d\n ( %d != %d )",
                               a, p, A->mt, A->nt,
                               m, k, start, next, m, k, next, prev, start, prev);
                        check = 0;
                        return 3;
                    }
                    start = next;
                } while ( start != A->mt );
            }
        }
        ENDCHECK( check, 3 );
    }

    return 0;
}

void dplasma_qr_print_type( tiled_matrix_desc_t *A, qr_piv_t *qrpiv )
{
    int minMN = min(A->mt, A->nt );
    int m, k;
    int lm = 0;
    int lmg = 0;
    int rank = 0;

    printf("\n------------ Localization = Type of pivot --------------\n");
    for(m=0; m<A->mt; m++) {
        printf("%3d | ", m);
        for (k=0; k<min(minMN, m+1); k++) {
            printf( "%3d ", dplasma_qr_gettype( qrpiv, k, m ) );
        }
        for (k=min(minMN, m+1); k<minMN; k++) {
            printf( "    " );
        }

        printf("    ");
        printf("%2d,%3d | ", rank, lmg);
        for (k=0; k<min(minMN, lmg+1); k++) {
            printf( "%3d ", dplasma_qr_gettype( qrpiv, k, lmg) );
        }
        for (k=min(minMN, lmg+1); k<minMN; k++) {
            printf( "    " );
        }
        lm++; lmg+=qrpiv->p;
        if ( lmg >= A->mt ) {
            rank++;
            lmg = rank;
            lm = 0;
        }
        printf("\n");
    }
}

void dplasma_qr_print_pivot( tiled_matrix_desc_t *A, qr_piv_t *qrpiv )
{
    int minMN = min(A->mt, A->nt );
    int m, k;
    int lm = 0;
    int lmg = 0;
    int rank = 0;
    printf("\n------------ Current Pivot--------------\n");
    for(m=0; m<A->mt; m++) {
        printf("%3d | ", m);
        for (k=0; k<min(minMN, m+1); k++) {
            printf( "%3d ", dplasma_qr_currpiv(qrpiv, m, k) );
        }
        for (k=min(minMN, m+1); k<minMN; k++) {
            printf( "    " );
        }

        printf("    ");
        printf("%2d,%3d | ", rank, lmg);
        for (k=0; k<min(minMN, lmg+1); k++) {
            printf( "%3d ", dplasma_qr_currpiv(qrpiv, lmg, k) );
        }
        for (k=min(minMN, lmg+1); k<minMN; k++) {
            printf( "    " );
        }
        lm++; lmg+=qrpiv->p;
        if ( lmg >= A->mt ) {
            rank++;
            lmg = rank;
            lm = 0;
        }
        printf("\n");
    }
}

void dplasma_qr_print_next_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k )
{
    int m, s;
    printf("\n------------ Next (k = %d)--------------\n", k);

    printf( "      " );
    for(s=A->mt; s>0; s--)
        printf( "%3d ", s );
    printf( "\n" );

    for(m=0; m<A->mt; m++) {
        printf("%3d | ", m);
        for(s=A->mt; s>0; s--) {
            printf( "%3d ", dplasma_qr_nextpiv(qrpiv, m, k, s) );
        }
        printf("\n");
    }
}

void dplasma_qr_print_prev_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k )
{
    int m, s;
    printf("\n------------ Prev (k = %d)--------------\n", k);

    printf( "      " );
    for(s=A->mt; s>-1; s--)
        printf( "%3d ", s );
    printf( "\n" );

    for(m=0; m<A->mt; m++) {
        printf("%3d | ", m);
        for(s=A->mt; s>-1; s--) {
            printf( "%3d ", dplasma_qr_prevpiv(qrpiv, m, k, s) );
        }
        printf("\n");
    }
}

void dplasma_qr_print_perm( tiled_matrix_desc_t *A, qr_piv_t *qrpiv )
{
    int minMN = min(A->mt, A->nt );
    int m, k;

    printf("\n------------ Permutation --------------\n");
    for (k=0; k<minMN; k++) {
        printf( "%3d ", k );
    }
    printf( "\n" );
    for (k=0; k<minMN; k++) {
        printf( "----" );
    }
    printf( "\n" );

    for (m=0; m < A->mt+1; m++) {
        for (k=0; k<minMN; k++) {
            printf( "%3d ", qrpiv->perm[ k*(A->mt+1) + m ] );
        }
        printf( "\n" );
    }
    printf( "\n" );
}

void dplasma_qr_print_nbgeqrt( tiled_matrix_desc_t *A, qr_piv_t *qrpiv )
{
    int minMN = min(A->mt, A->nt );
    int m, k, nb;

    printf("\n------------ Nb GEQRT per k --------------\n");
    printf(" k      : ");
    for (k=0; k<minMN; k++) {
        printf( "%3d ", k );
    }
    printf( "\n" );
    printf(" Compute: ");
    for (k=0; k<minMN; k++) {
        nb = 0;
        for (m=k; m < A->mt; m++) {
            if ( dplasma_qr_gettype(qrpiv, k, m) > 0 )
                nb++;
        }
        printf( "%3d ", nb );
    }
    printf( "\n" );
    printf(" Formula: ");
    for (k=0; k<minMN; k++) {
        printf( "%3d ", dplasma_qr_getnbgeqrf( qrpiv, k, A->mt) );
    }
    printf( "\n" );
}

void dplasma_qr_print_geqrt_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k )
{
    int i, m, nb;

    printf("\n------------ Liste of geqrt for k = %d --------------\n", k);

    printf( "  m:");
    nb = dplasma_qr_getnbgeqrf( qrpiv, k, A->mt );
    for (i=0; i < nb; i++) {
        m = dplasma_qr_getm( qrpiv, k, i );
        if ( i == dplasma_qr_geti( qrpiv, k, m) )
            printf( "%3d ", m );
        else
            printf( "x%2d ", dplasma_qr_geti( qrpiv, k, m) );
    }
    printf( "\n" );
}


/* static int dplasma_qr_getinon0( const qr_piv_t *qrpiv,  */
/*                                 const int k, int i, int mt )  */
/* { */
/*     int j; */
/*     for(j=k; j<mt; j++) { */
/*         if ( dplasma_qr_gettype( qrpiv, k, j ) != 0 ) */
/*             i--; */
/*         if ( i == -1 ) */
/*             break; */
/*     } */
/*     return qrpiv->perm[k*(qrpiv->desc->mt+1) + j]; */
/* } */

#define DAG_HEADER        "digraph G { orientation=portrait; \n"
#define DAG_FOOTER        "} // close graph\n"
#define DAG_LABELNODE     "%d [label=\"%d\",color=white,pos=\"-1.,-%d.!\"]\n"
#define DAG_LENGTHNODE    "l%d [label=\"%d\",color=white,pos=\"%d.,0.5!\"]\n"
#define DAG_INVISEDGE     "%d->%d [style=\"invis\"];\n"
#define DAG_STARTNODE     "p%d_m%d_k%d [shape=point,width=0.1, pos=\"%d.,-%d.!\",color=\"%s\"];\n"
#define DAG_NODE          "p%d_m%d_k%d [shape=point,width=0.1, pos=\"%d.,-%d.!\",color=\"%s\"];\n"
#define DAG_FIRSTEDGE_PIV "%d->p%d_m%d_k0\n"
#define DAG_FIRSTEDGE_TS  "%d->p%d_m%d_k0 [style=dotted,width=0.1]\n"
#define DAG_FIRSTEDGE_TT  "%d->p%d_m%d_k0 [style=dotted,width=0.1]\n"
#define DAG_EDGE_PIV      "p%d_m%d_k%d->p%d_m%d_k%d [width=0.1,color=\"%s\"]\n"
#define DAG_EDGE_TS       "p%d_m%d_k%d->p%d_m%d_k%d [style=dotted, width=0.1,color=\"%s\"]\n"
#define DAG_EDGE_TT       "p%d_m%d_k%d->p%d_m%d_k%d [style=dashed, width=0.1,color=\"%s\"]\n"

char *color[] = {
    "red",
    "blue",
    "green",
    "orange",
    "cyan",
    "purple",
    "yellow",
};
#define DAG_NBCOLORS 7

void dplasma_qr_print_dag( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, char *filename )
{
    int *pos, *next, *done;
    int k, m, n, lpos, prev, length;
    int minMN = min( A->mt, A->nt );
    FILE *f = fopen( filename, "w" );

    done = (int*)malloc( A->mt * sizeof(int) );
    pos  = (int*)malloc( A->mt * sizeof(int) );
    next = (int*)malloc( A->mt * sizeof(int) );
    memset(pos,  0, A->mt * sizeof(int) );
    memset(next, 0, A->mt * sizeof(int) );

    /* Print header */
    fprintf(f, DAG_HEADER ); /*, A->mt+2, minMN+2 );*/
    for(m=0; m < A->mt; m++) {
        fprintf(f, DAG_LABELNODE, m, m, m);
    }

    for(k=0; k<minMN; k++ ) {
        int nb2reduce = A->mt - k - 1;

        for(m=k; m < A->mt; m++) {
            fprintf(f, DAG_STARTNODE, m, A->mt, k, pos[m], m, color[ (m%qrpiv->p) % DAG_NBCOLORS ]);
            next[m] = dplasma_qr_nextpiv( qrpiv, m, k, A->mt);
        }

        while( nb2reduce > 0 ) {
            memset(done, 0, A->mt * sizeof(int) );
            for(m=A->mt-1; m > (k-1); m--) {
                n = next[m];
                if ( next[n] != A->mt )
                    continue;
                if ( n != A->mt ) {
                    lpos = max( pos[m], pos[n] );
                    lpos++;
                    pos[m] = lpos;
                    pos[n] = lpos;

                    fprintf(f, DAG_NODE, m, n, k, pos[m], m, color[ (m%qrpiv->p) % DAG_NBCOLORS ]);

                    prev = dplasma_qr_prevpiv( qrpiv, m, k, n );
                    fprintf(f, DAG_EDGE_PIV,
                           m, prev, k,
                           m, n,    k,
                           color[ (m%qrpiv->p) % DAG_NBCOLORS ]);

                    prev = dplasma_qr_prevpiv( qrpiv, n, k, n );
                    if ( dplasma_qr_gettype(qrpiv, k, n) == 0 )
                        fprintf(f, DAG_EDGE_TS,
                               n, prev, k,
                               m, n, k,
                               color[ (m%qrpiv->p) % DAG_NBCOLORS ]);
                    else
                        fprintf(f, DAG_EDGE_TT,
                               n, prev, k,
                               m, n, k,
                               color[ (m%qrpiv->p) % DAG_NBCOLORS ]);

                    next[m] = dplasma_qr_nextpiv( qrpiv, m, k, n);
                    done[m] = done[n] = 1;
                    nb2reduce--;
                }
            }
        }
    }

    length = 0;
    for(m=0; m < A->mt; m++) {
        length = max(length, pos[m]);
    }
    length++;
    for(k=0; k<length; k++)
        fprintf(f, DAG_LENGTHNODE, k, k, k);
    fprintf(f, DAG_FOOTER);

    printf("Tic Max = %d\n", length-1);

    fclose( f );
    free(pos);
    free(next);
}
