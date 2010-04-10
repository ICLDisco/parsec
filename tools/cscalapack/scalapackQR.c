#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <sys/time.h>

extern void   Cblacs_pinfo( int* mypnum, int* nprocs);
extern void   Cblacs_get( int context, int request, int* value);
extern int    Cblacs_gridinit( int* context, char * order, int np_row, int np_col);
extern void   Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
extern void   Cblacs_gridexit( int context);
extern void   Cblacs_exit( int error_code);
extern void blacs_pinfo_( int *mypnum, int *nprocs);
extern void blacs_get_( int *context, int *request, int* value);
extern void blacs_gridinit_( int* context, char *order, int *np_row, int *np_col);
extern void blacs_gridinfo_( int *context, int *np_row, int *np_col, int *my_row, int *my_col);
extern void blacs_gridexit_( int *context);
extern void pdgeqrf_( int *m, int *n, double *a, int *ia, int *ja, int *desca, double *tau, double *work, int *lwork, int *info );
extern void pdormqr_( char *side, char *trans, int *m, int *n, int *k, double *a, int *ia,
		int *ja, int *desca, double *tau, double *c, int *ic, int *jc, int *descc, double *work, int *lwork, int *info );
extern void pdtrsm_ ( char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *a, int *ia,
		int *ja, int *desca, double *b, int *ib, int *jb, int *descb );


extern float  pslange_( char *norm, int *m, int *n, float     *A, int *ia, int *ja, int *descA, float *work);
extern double pdlange_( char *norm, int *m, int *n, double    *A, int *ia, int *ja, int *descA, double *work);


extern void   pslacpy_( char *uplo, int *m, int *n, float     *A, int *ia, int *ja, int *descA,
                                                    float     *B, int *ib, int *jb, int *descB);
extern void   pdlacpy_( char *uplo, int *m, int *n, double     *A, int *ia, int *ja, int *descA,
                                                    double     *B, int *ib, int *jb, int *descB);

extern void   psgesv_( int *n, int *nrhs, float     *A, int *ia, int *ja, int *descA, int *ipiv,
                                          float     *B, int *ib, int *jb, int *descB, int *info);
extern void   pdgesv_( int *n, int *nrhs, double    *A, int *ia, int *ja, int *descA, int *ipiv,
                                          double    *B, int *ib, int *jb, int *descB, int *info);

extern void   psgemm_( char *transa, char *transb, int *M, int *N, int *K,
                                          float     *alpha,
                                          float     *A, int *ia, int *ja, int *descA,
                                          float     *B, int *ib, int *jb, int *descB,
                                          float     *beta,
                                          float     *C, int *ic, int *jc, int *descC );
extern void   pdgemm_( char *transa, char *transb, int *M, int *N, int *K,
                                          double    *alpha,
                                          double    *A, int *ia, int *ja, int *descA,
                                          double    *B, int *ib, int *jb, int *descB,
                                          double    *beta,
                                          double    *C, int *ic, int *jc, int *descC );

extern void   psgesvd_( char *jobu, char *jobvt, int *m, int *n,
                                  float     *A, int *ia, int *ja, int *descA,
                                  float     *s,
                                  float     *U, int *iu, int *ju, int *descU,
                                  float     *VT, int *ivt, int *jvt, int *descVT,
                                  float     *work, int *lwork, int *info);
extern void   pdgesvd_( char *jobu, char *jobvt, int *m, int *n,
                                  double    *A, int *ia, int *ja, int *descA,
                                  double    *s,
                                  double    *U, int *iu, int *ju, int *descU,
                                  double    *VT, int *ivt, int *jvt, int *descVT,
                                  double    *work, int *lwork, int *info);

extern void   pslaset_( char *uplo, int *m, int *n, float     *alpha, float     *beta, float     *A, int *ia, int *ja, int *descA );
extern void   pdlaset_( char *uplo, int *m, int *n, double    *alpha, double    *beta, double    *A, int *ia, int *ja, int *descA );

extern void   pselset_( float     *A, int *ia, int *ja, int *descA, float     *alpha);
extern void   pdelset_( double    *A, int *ia, int *ja, int *descA, double    *alpha);

extern void   pslawrite_( char **filenam, int *m, int *n, float  *A, int *ia, int *ja, int *descA, int *irwrit, int *icwrit, float  *work);
extern void   pdlawrite_( char **filenam, int *m, int *n, double *A, int *ia, int *ja, int *descA, int *irwrit, int *icwrit, double *work);

extern float  pslamch_( int *ictxt, char *cmach);
extern double pdlamch_( int *ictxt, char *cmach);

extern int    indxg2p_( int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern int    indxg2l_( int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern int    numroc_( int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void   descinit_( int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc,
				int *ictxt, int *lld, int *info);




static int max( int a, int b ){
        if (a>b) return(a); else return(b);
}

int main(int argc, char **argv) {
	int iam, nprocs, do_validation = 0;
	int myrank_mpi, nprocs_mpi;
	int ictxt, nprow, npcol, myrow, mycol;
	int np, nq, n, nb, nqrhs, nrhs;
	int i, j, k, info, itemp, seed;
	int descA[9], descB[9];
	double *A=NULL, *Acpy=NULL, *B=NULL, *X=NULL, *R=NULL, eps, *work=NULL;
    double AnormF, XnormF, RnormF, BnormF, residF;
	double *tau=NULL;
	int lwork;
	int izero=0,ione=1;
	double mone=(-1.0e0),pone=(1.0e0);
    /**/
	double MPIt1, MPIt2, MPIelapsed, GFLOPS, GFLOPS_per_proc ;
    /**/
	MPI_Init( &argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);
    /**/
	n = 100; nrhs = 1; nprow = 1; npcol = 1; nb = 64;
	for( i = 1; i < argc; i++ ) {
		if( strcmp( argv[i], "-n" ) == 0 ) {
			n      = atoi(argv[i+1]);
			i++;
		}
		if( strcmp( argv[i], "-nrhs" ) == 0 ) {
			nrhs   = atoi(argv[i+1]);
			i++;
		}
		if( strcmp( argv[i], "-p" ) == 0 ) {
			nprow  = atoi(argv[i+1]);
			i++;
		}
		if( strcmp( argv[i], "-q" ) == 0 ) {
			npcol  = atoi(argv[i+1]);
			i++;
		}
		if( strcmp( argv[i], "-nb" ) == 0 ) {
			nb     = atoi(argv[i+1]);
			i++;
		}
        if( strcmp( argv[i], "-v" ) == 0 ) {
            do_validation = 1;
        }
	}
    /**/
	if (nb>n)
		nb = n;
	if (nprow*npcol>nprocs_mpi){
		if (myrank_mpi==0)
			printf(" **** ERROR : we do not have enough processes available to make a p-by-q process grid ***\n");
        printf(" **** Bye-bye                                                                         ***\n");
		MPI_Finalize(); exit(1);
	}
    /**/
	/* no idea why I have problem with Cblacs on my computer, I am using blacsF77 interface here .... */
	blacs_pinfo_( &iam, &nprocs ) ;
	{ int im1 = -1; int i0 = 0; blacs_get_( &im1, &i0, &ictxt ); }
	blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
	blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

	//Cblacs_pinfo( &iam, &nprocs ) ;
	//Cblacs_get( -1, 0, &ictxt );
	//Cblacs_gridinit( &ictxt, "Row", nprow, npcol );
	//Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
    /**/
	if ( iam==0 ){
		printf("\n");
		printf("n\tnrhs\tnb\tp\tq\tinfo\tresid\ttime(s)  \tGFLOPS/sec\tGFLOPS/sec/proc\n");
	}
    /**/
    /*	
     *	if (iam==0)
     *		printf("\tn = %d\tnrhs = %d\t(%d,%d)\t%dx%d\n",n,nrhs,nprow,npcol,nb,nb);
     *	printf("Hello World, I am proc %d over %d for MPI, proc %d over %d for BLACS in position (%d,%d) in the process grid\n", 
     *	 		myrank_mpi,nprocs_mpi,iam,nprocs,myrow,mycol);
     */	 
    /*
     *
     *     Work only the process in the process grid
     *
     */
	if ((myrow < nprow)&(mycol < npcol)){
        /*
         *
         *     Compute the size of the local matrices (thanks to numroc)
         *
         */ 
		np    = numroc_( &n   , &nb, &myrow, &izero, &nprow );
		nq    = numroc_( &n   , &nb, &mycol, &izero, &npcol );
		nqrhs = numroc_( &nrhs, &nb, &mycol, &izero, &npcol );
        /*
         *
         *     Allocate and fill the matrices A and B
         *
         */ 

		seed = iam*n*(n+nrhs); srand(seed);
        /**/		
		A = (double *)calloc(np*nq,sizeof(double)) ;
		if (A==NULL){ printf("error of memory allocation A on proc %dx%d\n",myrow,mycol); exit(0); }
        /**/		
		Acpy = (double *)calloc(np*nq,sizeof(double)) ;
		if (Acpy==NULL){ printf("error of memory allocation Acpy on proc %dx%d\n",myrow,mycol); exit(0); }
        /**/		
		B = (double *)calloc(np*nqrhs,sizeof(double)) ;
		if (B==NULL){ printf("error of memory allocation B on proc %dx%d\n",myrow,mycol); exit(0); }
        /**/		
		X = (double *)calloc(np*nqrhs,sizeof(double)) ;
		if (X==NULL){ printf("error of memory allocation X on proc %dx%d\n",myrow,mycol); exit(0); }
        /**/		
		R = (double *)calloc(np*nqrhs,sizeof(double)) ;
		if (R==NULL){ printf("error of memory allocation R on proc %dx%d\n",myrow,mycol); exit(0); }
        /**/		
		tau = (double *)calloc(n,sizeof(double)) ;
		if (tau==NULL){ printf("error of memory allocation TAU on proc %dx%d\n",myrow,mycol); exit(0); }
        /**/		
		k = 0;
		for (i = 0; i < np; i++) {
			for (j = 0; j < nq; j++) {
				A[k] = ((double) rand()) / ((double) RAND_MAX) - 0.5 ;
				k++;	
			}
		}
		k = 0;
		for (i = 0; i < np; i++) {
			for (j = 0; j < nqrhs; j++) {
				B[k] = ((double) rand()) / ((double) RAND_MAX) - 0.5 ;
				k++;	
			}
		}
        /*
         *
         *     Initialize the array descriptor for the matrix A and B
         *
         */ 
		itemp = max( 1, np );
		descinit_( descA, &n, &n   , &nb, &nb, &izero, &izero, &ictxt, &itemp, &info );
		descinit_( descB, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &itemp, &info );
        /*
         *
         *     Make a copy of A and the rhs for checking purposes
         */
        pdlacpy_( "All", &n, &n   , A, &ione, &ione, descA, Acpy, &ione, &ione, descA );
        pdlacpy_( "All", &n, &nrhs, B, &ione, &ione, descB, X   , &ione, &ione, descB );
        /*
**********************************************************************
*     Call ScaLAPACK PDGESV routine
**********************************************************************
*/
        /**/
		lwork = -1;
		work = (double *)calloc(1,sizeof(double)) ;
		if (work==NULL){ printf("error of memory allocation WORK on proc %dx%d\n",myrow,mycol); exit(0); }
		pdgeqrf_( &n, &n, A, &ione, &ione, descA, tau, work, &lwork, &info );
		lwork = (int) work[0];
		free(work); work = NULL;
		work = (double *)calloc(lwork,sizeof(double)) ;
		if (work==NULL){ printf("error of memory allocation WORK on proc %dx%d\n",myrow,mycol); exit(0); }
        /**/		
		MPIt1 = MPI_Wtime();
		pdgeqrf_( &n, &n, A, &ione, &ione, descA, tau, work, &lwork, &info );
        /**/
		MPIt2 = MPI_Wtime();
		MPIelapsed=MPIt2-MPIt1;
		free(work); work = NULL;

        if( do_validation ) {
            /**/
            lwork = -1;
            work = (double *)calloc(1,sizeof(double)) ;
            if (work==NULL){ printf("error of memory allocation WORK on proc %dx%d\n",myrow,mycol); exit(0); }
            pdormqr_( "L", "T", &n, &nrhs, &n, A, &ione, &ione,
                      descA, tau, X, &ione, &ione, descB,
                      work, &lwork, &info );
            lwork = (int) work[0];
            free(work); work = NULL;
            work = (double *)calloc(lwork,sizeof(double)) ;
            if (work==NULL){ printf("error of memory allocation WORK on proc %dx%d\n",myrow,mycol); exit(0); }
            pdormqr_( "L", "T", &n, &nrhs, &n, A, &ione, &ione,
                      descA, tau, X, &ione, &ione, descB,
                      work, &lwork, &info );
            free(work); work=NULL;
            /**/
            pdtrsm_( "L", "U", "N", "N", &n, &nrhs, &pone, A, &ione, &ione, descA, X, &ione, &ione, descB );

            //fprintf(stderr,"%d ==> done \n",iam);Cblacs_gridexit( 0 ); MPI_Finalize(); exit(0);
            /*
             *     Compute residual ||A * X  - B|| / ( ||X|| * ||A|| * eps * N )
             *     Froebenius norm
             */
      		pdlacpy_( "All", &n, &nrhs, B, &ione, &ione, descB, R   , &ione, &ione, descB );
      		eps = pdlamch_( &ictxt, "Epsilon" );
            pdgemm_( "N", "N", &n, &nrhs, &n, &pone, Acpy, &ione, &ione, descA, X, &ione, &ione, descB,
                     &mone, R, &ione, &ione, descB);
            AnormF = pdlange_( "F", &n, &n   , A, &ione, &ione, descA, work);
            BnormF = pdlange_( "F", &n, &nrhs, B, &ione, &ione, descB, work);
            XnormF = pdlange_( "F", &n, &nrhs, X, &ione, &ione, descB, work);
            RnormF = pdlange_( "F", &n, &nrhs, R, &ione, &ione, descB, work);
            residF = RnormF / ( AnormF * XnormF * eps );
        }
        /**/
		GFLOPS = 4.0e0/3.e0*(((double) n)*((double) n)*((double) n))/1e+9/MPIelapsed;
		GFLOPS_per_proc = GFLOPS / (((double) nprow)*((double) npcol));
        /**/
		if ( iam==0 ){
			printf("%d\t%d\t%d\t%d\t%d\t%d\t%1.1f\t%f\t%f\t%f\n",
                   n, nrhs, nb, nprow, npcol, info, residF, MPIelapsed, GFLOPS, GFLOPS_per_proc);
		}
        /**/
		free(A);
		free(Acpy);
		free(B);
		free(X);
		free(tau);
	}
    /*
     *     Print ending messages
     */
	if ( iam==0 ){
		printf("\n");
	}
    /**/
	Cblacs_gridexit( 0 );
	MPI_Finalize();
	exit(0);
}
