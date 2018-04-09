/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
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
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"

#include <math.h>
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

/* static int dplasma_qrtree_getinon0( const qr_piv_t *arg,  */
/*                                 const int k, int i, int mt ); */

#define ENDCHECK( test, ret )                   \
    if ( !test )                                \
        return ret;

int dplasma_qrtree_check( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree)
{
    int minMN = dplasma_imin(A->mt, A->nt );
    int i, m, k, nb;
    int check;

    int a = qrtree->a;
    int p = qrtree->p;

    /*
     * Check Formula for NB geqrt
     */
    {
        /* dplasma_qrtree_print_type( A, qrtree ); */
        /* dplasma_qrtree_print_nbgeqrt( A, qrtree ); */
        check = 1;
        for (k=0; k<minMN; k++) {
            nb = 0;
            for (m=k; m < A->mt; m++) {
                if ( qrtree->gettype( qrtree, k, m ) > 0 )
                    nb++;
            }

            if ( nb != qrtree->getnbgeqrf( qrtree, k ) ) {
                check = 0;
                printf(" ----------------------------------------------------\n"
                       "  - a = %d, p = %d, M = %d, N = %d\n"
                       "     Check number of geqrt:\n"
                       "       For k=%d => return %d instead of %d",
                       a, p, A->mt, A->nt, k, qrtree->getnbgeqrf( qrtree, k ), nb );
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
            /* dplasma_qrtree_print_geqrt_k( A, qrtree, k ); */
            nb = qrtree->getnbgeqrf( qrtree, k );
            prevm = -1;
            for (i=0; i < nb; i++) {

                m = qrtree->getm( qrtree, k, i );

                /*
                 * getm ahas to be the inverse of geti
                 */
                if ( i != qrtree->geti( qrtree, k, m) ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d && geti( k=%d, m=%d ) => i = %d\n",
                           a, p, A->mt, A->nt,
                           k, i, m, k, m, qrtree->geti( qrtree, k, m));
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
                else if ( m != qrtree->getinon0( qrtree, k, i, A->mt ) ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d but should be %d",
                           a, p, A->mt, A->nt, k, i, m, qrtree->getinon0( qrtree, k, i, A->mt));
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
                    if ( qrtree->nextpiv(qrtree, k, m, s) == A->mt )
                        nb++;
                }
                if ( nb > 1 ) {
                    dplasma_qrtree_print_next_k( A, qrtree, k);
                    dplasma_qrtree_print_prev_k( A, qrtree, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Next of line %d for step %d contains more than one exit:\n",
                           a, p, A->mt, A->nt,
                           m, k);
                    check = 0;
                    return 3;
                }
                else if ( nb == 0 ) {
                    dplasma_qrtree_print_next_k( A, qrtree, k);
                    dplasma_qrtree_print_prev_k( A, qrtree, k);

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
                    if ( qrtree->prevpiv(qrtree, k, m, s) == A->mt )
                        nb++;
                }
                if ( nb > 1 ) {
                    dplasma_qrtree_print_next_k( A, qrtree, k);
                    dplasma_qrtree_print_prev_k( A, qrtree, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Prev of line %d for step %d contains more than one exit:\n",
                           a, p, A->mt, A->nt,
                           m, k);
                    check = 0;
                    return 3;
                }
                else if ( nb == 0 ) {
                    dplasma_qrtree_print_next_k( A, qrtree, k);
                    dplasma_qrtree_print_prev_k( A, qrtree, k);

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
                    next = qrtree->nextpiv(qrtree, k, m, start);
                    if ( next == A->mt )
                        prev = qrtree->prevpiv(qrtree, k, m, m);
                    else
                        prev = qrtree->prevpiv(qrtree, k, m, next);

                    if ( start != prev ) {
                        dplasma_qrtree_print_next_k( A, qrtree, k);
                        dplasma_qrtree_print_prev_k( A, qrtree, k);

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

void dplasma_qrtree_print_type( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree )
{
    int minMN = dplasma_imin(A->mt, A->nt );
    int m, k;
    int lm = 0;
    int lmg = 0;
    int rank = 0;

    printf("\n------------ Localization = Type of pivot --------------\n");
    for(m=0; m<A->mt; m++) {
        printf("%3d | ", m);
        for (k=0; k<dplasma_imin(minMN, m+1); k++) {
            printf( "%3d ", qrtree->gettype( qrtree, k, m ) );
        }
        for (k=dplasma_imin(minMN, m+1); k<minMN; k++) {
            printf( "    " );
        }

        printf("    ");
        printf("%2d,%3d | ", rank, lmg);
        for (k=0; k<dplasma_imin(minMN, lmg+1); k++) {
            printf( "%3d ", qrtree->gettype( qrtree, k, lmg) );
        }
        for (k=dplasma_imin(minMN, lmg+1); k<minMN; k++) {
            printf( "    " );
        }
        lm++; lmg+=qrtree->p;
        if ( lmg >= A->mt ) {
            rank++;
            lmg = rank;
            lm = 0;
        }
        printf("\n");
    }
}

void dplasma_qrtree_print_pivot( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree )
{
    int minMN = dplasma_imin(A->mt, A->nt );
    int m, k;
    int lm = 0;
    int lmg = 0;
    int rank = 0;
    printf("\n------------ Current Pivot--------------\n");
    for(m=0; m<A->mt; m++) {
        printf("%3d | ", m);
        for (k=0; k<dplasma_imin(minMN, m+1); k++) {
            printf( "%3d ", qrtree->currpiv(qrtree, k, m) );
        }
        for (k=dplasma_imin(minMN, m+1); k<minMN; k++) {
            printf( "    " );
        }

        printf("    ");
        printf("%2d,%3d | ", rank, lmg);
        for (k=0; k<dplasma_imin(minMN, lmg+1); k++) {
            printf( "%3d ", qrtree->currpiv(qrtree, k, lmg) );
        }
        for (k=dplasma_imin(minMN, lmg+1); k<minMN; k++) {
            printf( "    " );
        }
        lm++; lmg+=qrtree->p;
        if ( lmg >= A->mt ) {
            rank++;
            lmg = rank;
            lm = 0;
        }
        printf("\n");
    }
}

void dplasma_qrtree_print_next_k( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, int k )
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
            printf( "%3d ", qrtree->nextpiv(qrtree, k, m, s) );
        }
        printf("\n");
    }
}

void dplasma_qrtree_print_prev_k( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, int k )
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
            printf( "%3d ", qrtree->prevpiv(qrtree, k, m, s) );
        }
        printf("\n");
    }
}

void dplasma_qrtree_print_perm( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, int *perm )
{
    int minMN = dplasma_imin(A->mt, A->nt );
    int m, k;
    (void)qrtree;

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
            printf( "%3d ", perm[ k*(A->mt+1) + m ] );
        }
        printf( "\n" );
    }
    printf( "\n" );
}

void dplasma_qrtree_print_nbgeqrt( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree )
{
    int minMN = dplasma_imin(A->mt, A->nt );
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
            if ( qrtree->gettype(qrtree, k, m) > 0 )
                nb++;
        }
        printf( "%3d ", nb );
    }
    printf( "\n" );
    printf(" Formula: ");
    for (k=0; k<minMN; k++) {
        printf( "%3d ", qrtree->getnbgeqrf( qrtree, k ) );
    }
    printf( "\n" );
}

void dplasma_qrtree_print_geqrt_k( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, int k )
{
    int i, m, nb;
    (void)A;

    printf("\n------------ Liste of geqrt for k = %d --------------\n", k);

    printf( "  m:");
    nb = qrtree->getnbgeqrf( qrtree, k );
    for (i=0; i < nb; i++) {
        m = qrtree->getm( qrtree, k, i );
        if ( i == qrtree->geti( qrtree, k, m) )
            printf( "%3d ", m );
        else
            printf( "x%2d ", qrtree->geti( qrtree, k, m) );
    }
    printf( "\n" );
}


/* static int dplasma_qrtree_getinon0( const dplasma_qrtree_t *qrtree,  */
/*                                 const int k, int i, int mt )  */
/* { */
/*     int j; */
/*     for(j=k; j<mt; j++) { */
/*         if ( dplasma_qrtree_gettype( qrtree, k, j ) != 0 ) */
/*             i--; */
/*         if ( i == -1 ) */
/*             break; */
/*     } */
/*     return qrtree->perm[k*(qrtree->desc->mt+1) + j]; */
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

void dplasma_qrtree_print_dag( parsec_tiled_matrix_dc_t *A, dplasma_qrtree_t *qrtree, char *filename )
{
    int *pos, *next, *done;
    int k, m, n, lpos, prev, length;
    int minMN = dplasma_imin( A->mt, A->nt );
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
            fprintf(f, DAG_STARTNODE, m, A->mt, k, pos[m], m, color[ (m%qrtree->p) % DAG_NBCOLORS ]);
            next[m] = qrtree->nextpiv( qrtree, k, m, A->mt);
        }

        while( nb2reduce > 0 ) {
            memset(done, 0, A->mt * sizeof(int) );
            for(m=A->mt-1; m > (k-1); m--) {
                n = next[m];
                if ( next[n] != A->mt )
                    continue;
                if ( n != A->mt ) {
                    lpos = dplasma_imax( pos[m], pos[n] );
                    lpos++;
                    pos[m] = lpos;
                    pos[n] = lpos;

                    fprintf(f, DAG_NODE, m, n, k, pos[m], m, color[ (m%qrtree->p) % DAG_NBCOLORS ]);

                    prev = qrtree->prevpiv( qrtree, k, m, n );
                    fprintf(f, DAG_EDGE_PIV,
                            m, prev, k,
                            m, n,    k,
                            color[ (m%qrtree->p) % DAG_NBCOLORS ]);

                    prev = qrtree->prevpiv( qrtree, k, n, n );
                    if ( qrtree->gettype(qrtree, k, n) == 0 )
                        fprintf(f, DAG_EDGE_TS,
                                n, prev, k,
                                m, n, k,
                                color[ (m%qrtree->p) % DAG_NBCOLORS ]);
                    else
                        fprintf(f, DAG_EDGE_TT,
                                n, prev, k,
                                m, n, k,
                                color[ (m%qrtree->p) % DAG_NBCOLORS ]);

                    next[m] = qrtree->nextpiv( qrtree, k, m, n);
                    done[m] = done[n] = 1;
                    nb2reduce--;
                }
            }
        }
    }

    length = 0;
    for(m=0; m < A->mt; m++) {
        length = dplasma_imax(length, pos[m]);
    }
    length++;
    for(k=0; k<length; k++)
        fprintf(f, DAG_LENGTHNODE, k, k, k);
    fprintf(f, DAG_FOOTER);

    printf("Tic Max = %d\n", length-1);

    fclose( f );
    free(pos);
    free(next);
    free(done);
}
