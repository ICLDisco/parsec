/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_config.h"
#include "dplasma_cores.h"
#include "dplasma_zcores.h"

#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */
#if defined(HAVE_STDARG_H)
#include <stdarg.h>
#endif  /* defined(HAVE_STDARG_H) */
#include <stdio.h>
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif

#include "plasma.h"
#include "cblas.h"
#include "core_blas.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define  DLARFG      zlarfg_
extern void DLARFG(int *N, Dague_Complex64_t *ALPHA, Dague_Complex64_t *X, int *INCX, Dague_Complex64_t *TAU);

void band_to_trd_vmpi1(int N, int NB, Dague_Complex64_t *A, int LDA);
void band_to_trd_vmpi2(int N, int NB, Dague_Complex64_t *A, int LDA);
void band_to_trd_v8seq(int N, int NB, Dague_Complex64_t *A, int LDA, int INgrsiz, int INthgrsiz);
int TRD_seqgralgtype(int N, int NB, Dague_Complex64_t *A, int LDA, Dague_Complex64_t *C, Dague_Complex64_t *S, int i, int j, int m, int grsiz, int BAND);
int blgchase_ztrdv1(int NT, int N, int NB, Dague_Complex64_t *A, Dague_Complex64_t *V, Dague_Complex64_t *TAU, int sweep, int id, int blktile);
int blgchase_ztrdv2(int NT, int N, int NB, Dague_Complex64_t *A1, Dague_Complex64_t *A2, Dague_Complex64_t *V1, Dague_Complex64_t *TAU1, Dague_Complex64_t *V2, Dague_Complex64_t *TAU2, int sweep, int id, int blktile);

int CORE_zlarfx2(int side, int N,
                Dague_Complex64_t V,
                Dague_Complex64_t TAU,
                Dague_Complex64_t *C1, int LDC1,
                Dague_Complex64_t *C2, int LDC2);

int CORE_zlarfx2c(int uplo,
                Dague_Complex64_t V,
                Dague_Complex64_t TAU,
                Dague_Complex64_t *C1,
                Dague_Complex64_t *C2,
                Dague_Complex64_t *C3);

static void CORE_zhbtelr(int N, int NB, Dague_Complex64_t *A1, int LDA1, Dague_Complex64_t *A2, int LDA2, Dague_Complex64_t *V1, Dague_Complex64_t *TAU1, int st, int ed);
static void CORE_zhbtrce(int N, int NB, Dague_Complex64_t *A1, int LDA1, Dague_Complex64_t *A2, int LDA2, Dague_Complex64_t *V1, Dague_Complex64_t *TAU1, Dague_Complex64_t *V2, Dague_Complex64_t *TAU2, int st, int ed, int edglob);
static void CORE_zhbtlrx(int N, int NB, Dague_Complex64_t *A1, int LDA1, Dague_Complex64_t *A2, int LDA2, Dague_Complex64_t *V1, Dague_Complex64_t *TAU1, int st, int ed);

static void DLARFX_C(char side, int N, Dague_Complex64_t V, Dague_Complex64_t TAU, Dague_Complex64_t *C, int LDC);
static void TRD_type1bHL(int N, int NB, Dague_Complex64_t *A, int LDA, Dague_Complex64_t *V, Dague_Complex64_t *TAU, int st, int ed);
static void TRD_type2bHL(int N, int NB, Dague_Complex64_t *A, int LDA, Dague_Complex64_t *V, Dague_Complex64_t *TAU, int st, int ed);
static void TRD_type3bHL(int N, int NB, Dague_Complex64_t *A, int LDA, Dague_Complex64_t *V, Dague_Complex64_t *TAU, int st, int ed);






int CORE_zlarfx2(int side, int N,
                Dague_Complex64_t V,
                Dague_Complex64_t TAU,
                Dague_Complex64_t *C1, int LDC1,
                Dague_Complex64_t *C2, int LDC2)
{
    static Dague_Complex64_t zzero = 0.0;
    int    J;
    Dague_Complex64_t V2, T2, SUM;

    /* Quick return */
    /*
    if (N == 0)
        return PLASMA_SUCCESS;
    */
    if (TAU == zzero)
        return 0;

    /*
     * Special code for 2 x 2 Householder where V1 = I
    */
    if(side==PlasmaLeft){
        V2 = conj(V);
        T2 = TAU*conj(V2);
        for (J = 0; J < N ; J++){
           SUM         = C1[J*LDC1]   + V2*C2[J*LDC2];
           C1[J*LDC1]  = C1[J*LDC1]   - SUM*TAU;
           C2[J*LDC2]  = C2[J*LDC2]   - SUM*T2;
        }
    }else if(side==PlasmaRight){
        V2 = V;
        T2 = TAU*conj(V2);
        for (J = 0; J < N ; J++){
           SUM     = C1[J]   + V2*C2[J];
           C1[J]   = C1[J]   - SUM*TAU;
           C2[J]   = C2[J]   - SUM*T2;
        }
    }

    return 0;
}

/***************************************************************************//**
 *
 **/
int CORE_zlarfx2c(int uplo,
                Dague_Complex64_t V,
                Dague_Complex64_t TAU,
                Dague_Complex64_t *C1,
                Dague_Complex64_t *C2,
                Dague_Complex64_t *C3)
{
    static Dague_Complex64_t zzero = 0.0;
    Dague_Complex64_t T2, SUM, TEMP;

    /* Quick return */
    if (TAU == zzero)
        return 0;

   /*
    *        Special code for a diagonal block  C1
    *                                           C2  C3
    */
 if(uplo==PlasmaLower){   //do the corner Left then Right (used for the lower case tridiag)
      // L and R for the 2x2 corner
      //            C(N-1, N-1)  C(N-1,N)        C1  TEMP
      //            C(N  , N-1)  C(N  ,N)        C2  C3
      // For Left : use conj(TAU) and V.
      // For Right: nothing, keep TAU and V.
      // Left 1 ==> C1
      //            C2
      TEMP    = conj(C2[0]); // copy C2 here before modifying it.
      T2      = conj(TAU)*V;
      SUM     = C1[0]   + conj(V)*C2[0];
      C1[0]   = C1[0]   - SUM*conj(TAU);
      C2[0]   = C2[0]   - SUM*T2;
      // Left 2 ==> TEMP
      //            C3
      SUM     = TEMP    + conj(V)*C3[0];
      TEMP    = TEMP    - SUM*conj(TAU);
      C3[0]   = C3[0]   - SUM*T2;
      // Right 1 ==>  C1 TEMP.  NB: no need to compute corner (2,2)=TEMP
      T2      = TAU*conj(V);
      SUM     = C1[0]   + V*TEMP;
      C1[0]   = C1[0]   - SUM*TAU;
      // Right 2 ==> C2 C3
      SUM     = C2[0]   + V*C3[0];
      C2[0]   = C2[0]   - SUM*TAU;
      C3[0]   = C3[0]   - SUM*T2;
 }else if(uplo==PlasmaUpper){ // do the corner Right then Left (used for the upper case tridiag)
      //            C(N-1, N-1)  C(N-1,N)        C1    C2
      //            C(N  , N-1)  C(N  ,N)        TEMP  C3
      // For Left : use TAU       and conj(V).
      // For Right: use conj(TAU) and conj(V).
      // Right 1 ==> C1 C2
      V       = conj(V);
      TEMP    = conj(C2[0]); // copy C2 here before modifying it.
      T2      = conj(TAU)*conj(V);
      SUM     = C1[0]   + V*C2[0];
      C1[0]   = C1[0]   - SUM*conj(TAU);
      C2[0]   = C2[0]   - SUM*T2;
      // Right 2 ==> TEMP C3
      SUM     = TEMP    + V*C3[0];
      TEMP    = TEMP    - SUM*conj(TAU);
      C3[0]   = C3[0]   - SUM*T2;
      // Left 1 ==> C1
      //            TEMP. NB: no need to compute corner (2,1)=TEMP
      T2      = TAU*V;
      SUM     = C1[0]   + conj(V)*TEMP;
      C1[0]   = C1[0]   - SUM*TAU;
      // Left 2 ==> C2
      //            C3
      SUM     = C2[0]   + conj(V)*C3[0];
      C2[0]   = C2[0]   - SUM*TAU;
      C3[0]   = C3[0]   - SUM*T2;
 }

 return 0;
}












///////////////////////////////////////////////////////////
//                  DLARFX en C 
///////////////////////////////////////////////////////////
static void DLARFX_C(char side, int N, Dague_Complex64_t V, Dague_Complex64_t TAU, Dague_Complex64_t *C, int LDC)
{
 Dague_Complex64_t T2, SUM, TEMP;
 int    J, pt;

/*
 *        Special code for 2 x 2 Householder
 */


 T2 = TAU*V;
 if(side=='L'){       
     for (J = 0; J < N ; J++){
        pt      = LDC*J;
        SUM     = C[pt]   + V*C[pt+1];
        C[pt]   = C[pt]   - SUM*TAU;
        C[pt+1] = C[pt+1] - SUM*T2;
     }
 }else if(side=='R'){
     for (J = 0; J < N ; J++){
        pt      = J+LDC;
        SUM     = C[J]   + V*C[pt];
        C[J]    = C[J]   - SUM*TAU;
        C[pt]   = C[pt]  - SUM*T2;
     }
 }else if(side=='B'){       
     TEMP    = C[ LDC * (N-2) +1];
     for (J = 0; J < N-1 ; J++){
        pt      = LDC*J;
        SUM     = C[pt]   + V*C[pt+1];
        C[pt]   = C[pt]   - SUM*TAU;
        C[pt+1] = C[pt+1] - SUM*T2;
     }
     // L and R for the 2x2 corner
     //            C(1, N-1)  C(1,N)        C(1, N-1)  TEMP
     //            C(2, N-1)  C(2,N)        C(2, N-1)  C(2,N)
     // Left 1 ==> C(1,N-1) C(2,N-1) deja fait dans la boucle
     //pt      = LDC * (N-2);
     //TEMP    = C[pt+1];
     //SUM     = C[pt]   + V*C[pt+1];
     //C[pt]   = C[pt]   - SUM*TAU;
     //C[pt+1] = C[pt+1] - SUM*T2;
     // Left 2 ==> TEMP C(2,N)
     pt      = LDC * (N-1) +1;
     SUM     = TEMP    + V*C[pt];
     TEMP    = TEMP    - SUM*TAU;
     C[pt]   = C[pt]   - SUM*T2;
     // Right 1 ==>  C(1,N-1) TEMP.  NB: no need to compute corner (2,2)
     J       = LDC * (N-2);
     SUM     = C[J]   + V*TEMP;
     C[J]    = C[J]   - SUM*TAU;
     // Right 2 ==> C(2,N-1) C(2,N)
     J      = LDC * (N-2) + 1;
     pt     = LDC * (N-1) + 1;
     SUM    = C[J]   + V*C[pt];
     C[J]   = C[J]   - SUM*TAU;
     C[pt]  = C[pt]  - SUM*T2;
 }
}
///////////////////////////////////////////////////////////
 
///////////////////////////////////////////////////////////
#define A1(m,n)   &(A1[((m)-(n)) + LDA1*(n)])
#define A2(m,n)   &(A2[((m)-(n)) + LDA2*((n)-NB)])
#define V1(m)     &(V1[m-st])
#define TAU1(m)   &(TAU1[m-st])
#define V2(m)     &(V2[m-st])
#define TAU2(m)   &(TAU2[m-st])
///////////////////////////////////////////////////////////
//                  TYPE 1-BAND Householder
///////////////////////////////////////////////////////////
static void CORE_zhbtelr(int N, int NB, Dague_Complex64_t *A1, int LDA1, Dague_Complex64_t *A2, int LDA2, Dague_Complex64_t *V1, Dague_Complex64_t *TAU1, int st, int ed) {
  int    J1, J2, KDM1, LDX;
  int    len, len1, len2, t1ed, t2st;
  int    i, IONE, ITWO; 
  IONE=1;
  ITWO=2; 
  (void)N;

  KDM1 = NB-1;
  LDX = LDA1-1;
  LDX = LDA2-1;
  /* **********************************************************************************************
   *   Annihiliate, then LEFT:
   * ***********************************************************************************************/
  for (i = ed; i >= st+1 ; i--){
     /* generate Householder to annihilate a(i+k-1,i) within the band */
     *V1(i)          = *A1(i, (st-1));
     *A1(i, (st-1))  = 0.0;
     DLARFG( &ITWO, A1((i-1),(st-1)), V1(i), &IONE, TAU1(i) );

     J1   = st;
     J2   = i-2;
     t1ed = min(J2,KDM1);
     t2st = max(J1, NB); 
     len1 = t1ed - J1 +1;
     len2 = J2 - t2st +1;
     // printf("Type 1L st %d   ed %d    i %d    J1 %d   J2 %d  t1ed %d  t2st %d len1 %d len2 %d \n", st, ed, i, J1,J2,t1ed,t2st,len1,len2);
     /* apply reflector from the left (horizontal row) and from the right for only the diagonal 2x2.*/
     if(len2>=0){
        /* part of the left(if len2>0) and the corner are on tile T2 */    
	if(len2>0) CORE_zlarfx2(PlasmaLeft, len2 , *V1(i), conj(*TAU1(i)), A2((i-1), t2st), LDX, A2(i, t2st), LDX);
        CORE_zlarfx2c(PlasmaLower, *V1(i), *TAU1(i), A2(i-1,i-1), A2(i,i-1), A2(i,i)); 
        if(len1>0) CORE_zlarfx2(PlasmaLeft, len1 , *V1(i), conj(*TAU1(i)), A1(i-1, J1), LDX, A1(i, J1), LDX);
     }else if(len2==-1){
        /* the left is on tile T1, and only A(i,i) of the corner is on tile T2 */    
        CORE_zlarfx2c(PlasmaLower, *V1(i), *TAU1(i), A1(i-1,i-1), A1(i,i-1), A2(i,i)); 
        if(len1>0) CORE_zlarfx2(PlasmaLeft, len1 , *V1(i), conj(*TAU1(i)), A1(i-1, J1), LDX, A1(i, J1), LDX);
     }else{
        /* the left and the corner are on tile T1, nothing on tile T2 */    
        CORE_zlarfx2c(PlasmaLower, *V1(i), *TAU1(i), A1(i-1,i-1), A1(i,i-1), A1(i,i)) ; 
        if(len1>0) CORE_zlarfx2(PlasmaLeft, len1 , *V1(i), conj(*TAU1(i)), A1((i-1), J1), LDX, A1(i, J1), LDX);
     }
  }
  /* **********************************************************************************************
   *   APPLY RIGHT ON THE REMAINING ELEMENT OF KERNEL 1
   * ***********************************************************************************************/
  for (i = ed; i >= st+1 ; i--){
     J1    = i+1;
     J2    = ed;
     len   = J2-J1+1;
     if(len>0){
   	if(i>NB)
           /* both column (i-1) and i are on tile T2 */   
           CORE_zlarfx2(PlasmaRight, len, *V1(i), *TAU1(i), A2(J1, i-1), LDX, A2(J1, i), LDX);
        else if(i==NB)
           /* column (i-1) is on tile T1 while column i is on tile T2 */   
           CORE_zlarfx2(PlasmaRight, len, *V1(i), *TAU1(i), A1(J1, i-1), LDX, A2(J1, i), LDX);
        else
           /* both column (i-1) and i are on tile T1 */   
           CORE_zlarfx2(PlasmaRight, len, *V1(i), *TAU1(i), A1(J1, i-1), LDX, A1(J1, i), LDX);
     }
  }
}
///////////////////////////////////////////////////////////







///////////////////////////////////////////////////////////
//                  TYPE 2-BAND Householder
///////////////////////////////////////////////////////////
static void CORE_zhbtrce(int N, int NB, Dague_Complex64_t *A1, int LDA1, Dague_Complex64_t *A2, int LDA2, Dague_Complex64_t *V1, Dague_Complex64_t *TAU1, Dague_Complex64_t *V2, Dague_Complex64_t *TAU2, int st, int ed, int edglob) {
  int    J1, J2, J3, KDM1, LDX, pt;
  int    len, len1, len2, t1ed, t2st, iglob;
  int    i, IONE, ITWO; 
  Dague_Complex64_t V,T,SUM;
  IONE=1;
  ITWO=2; 

  iglob = edglob+1;
  LDX   = LDA1-1;
  KDM1  = NB-1;  
  /* **********************************************************************************************
   *   Right:
   * ***********************************************************************************************/
  for (i = ed; i >= st+1 ; i--){
     /* apply Householder from the right. and create newnnz outside the band if J3 < N */
     iglob = iglob -1;
     J1  = ed+1;
     if((iglob+NB)>= N){
        J2 = i +(N-iglob-1);
        J3 = J2;         
     }else{
        J2 = i + KDM1;
	J3 = J2+1;
     }
     len   = J2-J1+1;
     /* printf("Type 2R st %d   ed %d    i %d   J1 %d   J2 %d  len %d iglob %d  \n",st,ed,i,J1,J2,len,iglob);*/

     if(len>0){
   	if(i>NB){
           /* both column (i-1) and i are on tile T2 */   
           CORE_zlarfx2(PlasmaRight, len, *V1(i), *TAU1(i), A2(J1, i-1), LDX, A2(J1, i), LDX);
           /* if nonzero element need to be created outside the band (if index < N) then create and eliminate it. */
           if(J3>J2){
              /* new nnz at TEMP=V2(i)  */
              V          = *V1(i);
              T          = *TAU1(i) * conj(V);
              SUM        =  V * (*A2(J3, i));
              *V2(i)     = -SUM * (*TAU1(i));
              *A2(J3, i) = *A2(J3, i) - SUM * T;
              /* generate Householder to annihilate a(j+kd,j-1) within the band */
              DLARFG( &ITWO, A2(J2,i-1), V2(i), &IONE, TAU2(i) );
           }
	}else if(i==NB){
           /* column (i-1) is on tile T1 while column i is on tile T2 */   
           CORE_zlarfx2(PlasmaRight, len, *V1(i), *TAU1(i), A1(J1, i-1), LDX, A2(J1, i), LDX);
           /* if nonzero element need to be created outside the band (if index < N) then create and eliminate it. */
           if(J3>J2){
              /* new nnz at TEMP=V2(i)  */
              V          = *V1(i);
              T          = *TAU1(i) * conj(V);
              SUM        =  V * (*A2(J3, i));
              *V2(i)     = -SUM * (*TAU1(i));
              *A2(J3, i) = *A2(J3, i) - SUM * T;
              /* generate Householder to annihilate a(j+kd,j-1) within the band */
              DLARFG( &ITWO, A1(J2, i-1), V2(i), &IONE, TAU2(i) );
           }
	}else{
           /* both column (i-1) and i are on tile T1 */   
           CORE_zlarfx2(PlasmaRight, len, *V1(i), *TAU1(i), A1(J1, i-1), LDX, A1(J1, i), LDX);
           /* if nonzero element need to be created outside the band (if index < N) then create and eliminate it. */
           if(J3>J2){
              /* new nnz at TEMP=V2(i)  */
              V              = *V1(i);
              T              = *TAU1(i) * conj(V);
              SUM            =  V * (*A1(J3, i));
              *V2(i)         = -SUM * (*TAU1(i));
              *A1(J3, i)    = *A1(J3, i) - SUM * T;
              /* generate Householder to annihilate a(j+kd,j-1) within the band */
              DLARFG( &ITWO, A1(J2, i-1), V2(i), &IONE, TAU2(i) );
           }
        }
     }
  }
	      // if(id==1) return;
  
  /* **********************************************************************************************
   *   APPLY LEFT ON THE REMAINING ELEMENT OF KERNEL 1
   * ***********************************************************************************************/
  iglob = edglob+1;
  for (i = ed; i >= st+1 ; i--){
     iglob = iglob -1;
     if((iglob+NB)< N){ /* mean that J3>J2 and so a nnz has been created and so a left is required. */
        J1   = i;
        J2   = ed;
        t1ed = min(J2,KDM1);
        t2st = max(J1, NB); 
        len1 = t1ed - J1 +1;
        len2 = J2 - t2st +1;
        pt   = i + KDM1; /* pt correspond to the J2 position of the corresponding right done above */
        //printf("Type 2L st %d   ed %d    i %d    J1 %d   J2 %d  t1ed %d  t2st %d len1 %d len2 %d \n", st, ed, i, J1,J2,t1ed,t2st,len1,len2);

	/* apply reflector from the left (horizontal row) and from the right for only the diagonal 2x2.*/
        if(len2>0){
           /* part of the left(if len2>0) and the corner are on tile T2 */    
   	   CORE_zlarfx2(PlasmaLeft, len2 , *V2(i), conj(*TAU2(i)), A2(pt, t2st), LDX, A2(pt+1, t2st), LDX);
           if(len1>0) CORE_zlarfx2(PlasmaLeft, len1 , *V2(i), conj(*TAU2(i)), A1(pt, J1), LDX, A1(pt+1, J1), LDX);
        }else if(len1>0){
           /* the left and the corner are on tile T1, nothing on tile T2 */    
           CORE_zlarfx2(PlasmaLeft, len1 , *V2(i), conj(*TAU2(i)), A1(pt, J1), LDX, A1(pt+1, J1), LDX);
        }
     }
  }
}
///////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////
//                  TYPE 1-BAND Householder
///////////////////////////////////////////////////////////
static void CORE_zhbtlrx(int N, int NB, Dague_Complex64_t *A1, int LDA1, Dague_Complex64_t *A2, int LDA2, Dague_Complex64_t *V1, Dague_Complex64_t *TAU1, int st, int ed) {
  int    J1, J2, KDM1, LDX;
  int    len, len1, len2, t1ed, t2st;
  int    i;

  (void)N;
  KDM1 = NB-1;
  LDX = LDA1-1;
  LDX = LDA2-1;
  /* **********************************************************************************************
   *   Annihiliate, then LEFT:
   * ***********************************************************************************************/
  for (i = ed; i >= st+1 ; i--){
     J1   = st;
     J2   = i-2;
     t1ed = min(J2,KDM1);
     t2st = max(J1, NB); 
     len1 = t1ed - J1 +1;
     len2 = J2 - t2st +1;
     //printf("Type 3L st %d   ed %d    i %d    J1 %d   J2 %d  t1ed %d  t2st %d len1 %d len2 %d \n", st, ed, i, J1,J2,t1ed,t2st,len1,len2);
     /* apply reflector from the left (horizontal row) and from the right for only the diagonal 2x2.*/
     if(len2>=0){
        /* part of the left(if len2>0) and the corner are on tile T2 */    
	if(len2>0) CORE_zlarfx2(PlasmaLeft, len2 , *V1(i), conj(*TAU1(i)), A2((i-1), t2st), LDX, A2(i, t2st), LDX);
        CORE_zlarfx2c(PlasmaLower, *V1(i), *TAU1(i), A2(i-1,i-1), A2(i,i-1), A2(i,i)); 
        if(len1>0) CORE_zlarfx2(PlasmaLeft, len1 , *V1(i), conj(*TAU1(i)), A1(i-1, J1), LDX, A1(i, J1), LDX);
     }else if(len2==-1){
        /* the left is on tile T1, and only A(i,i) of the corner is on tile T2 */    
        CORE_zlarfx2c(PlasmaLower, *V1(i), *TAU1(i), A1(i-1,i-1), A1(i,i-1), A2(i,i)); 
        if(len1>0) CORE_zlarfx2(PlasmaLeft, len1 , *V1(i), conj(*TAU1(i)), A1(i-1, J1), LDX, A1(i, J1), LDX);
     }else{
        /* the left and the corner are on tile T1, nothing on tile T2 */    
        CORE_zlarfx2c(PlasmaLower, *V1(i), *TAU1(i), A1(i-1,i-1), A1(i,i-1), A1(i,i)) ; 
        if(len1>0) CORE_zlarfx2(PlasmaLeft, len1 , *V1(i), conj(*TAU1(i)), A1((i-1), J1), LDX, A1(i, J1), LDX);
     }
  }
  /* **********************************************************************************************
   *   APPLY RIGHT ON THE REMAINING ELEMENT OF KERNEL 1
   * ***********************************************************************************************/
  for (i = ed; i >= st+1 ; i--){
     J1    = i+1;
     J2    = ed;
     len   = J2-J1+1;
     if(len>0){
   	if(i>NB)
           /* both column (i-1) and i are on tile T2 */   
           CORE_zlarfx2(PlasmaRight, len, *V1(i), *TAU1(i), A2(J1, i-1), LDX, A2(J1, i), LDX);
        else if(i==NB)
           /* column (i-1) is on tile T1 while column i is on tile T2 */   
           CORE_zlarfx2(PlasmaRight, len, *V1(i), *TAU1(i), A1(J1, i-1), LDX, A2(J1, i), LDX);
        else
           /* both column (i-1) and i are on tile T1 */   
           CORE_zlarfx2(PlasmaRight, len, *V1(i), *TAU1(i), A1(J1, i-1), LDX, A1(J1, i), LDX);
     }
  }
}
///////////////////////////////////////////////////////////
#undef A1
#undef A2
#undef V1
#undef TAU1
#undef V2
#undef TAU2











///////////////////////////////////////////////////////////
//                  TYPE 1-BAND Householder
///////////////////////////////////////////////////////////
//// add -1 because of C
#define A(m,n)   &(A[((m)-(n)) + LDA*((n)-1)])
#define V(m)     &(V[m-1])
#define TAU(m)   &(TAU[m-1])
static void TRD_type1bHL(int N, int NB, Dague_Complex64_t *A, int LDA, Dague_Complex64_t *V, Dague_Complex64_t *TAU, int st, int ed) {
  int    J1, J2, len, LDX;
  int    i, IONE, ITWO; 
  IONE=1;
  ITWO=2; 
  (void)NB;

  if(ed <= st){
    printf("TRD_type 1bH: ERROR st and ed %d  %d \n",st,ed);
    exit(-10);
  }

  LDX = LDA-1;
  for (i = ed; i >= st+1 ; i--){
     // generate Householder to annihilate a(i+k-1,i) within the band
     *V(i)          = *A(i, (st-1));
     *A(i, (st-1))  = 0.0;
     DLARFG( &ITWO, A((i-1),(st-1)), V(i), &IONE, TAU(i) );

     // apply reflector from the left (horizontal row) and from the right for only the diagonal 2x2. 
     J1  = st;
     J2  = i;
     len = J2-J1+1;
     printf("voici J1 %d J2 %d len %d \n",J1,J2,len);
     DLARFX_C('B', len , *V(i), *TAU(i), A((i-1),J1   ), LDX);
  }

  for (i = ed; i >= st+1 ; i--){
     len = min(ed,N)-i;
     if(len>0)DLARFX_C('R', len, *V(i), *TAU(i), A((i+1),(i-1)), LDX);
  }


}
#undef A
#undef V
#undef TAU
///////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////
//                  TYPE 2-BAND Householder
///////////////////////////////////////////////////////////
//// add -1 because of C
#define A(m,n)   &(A[((m)-(n)) + LDA*((n)-1)])
#define V(m)     &(V[m-1])
#define TAU(m)   &(TAU[m-1])
static void TRD_type2bHL(int N, int NB, Dague_Complex64_t *A, int LDA, Dague_Complex64_t *V, Dague_Complex64_t *TAU, int st, int ed) {
  int    J1, J2, J3, KDM2, len, LDX;
  int    i, IONE, ITWO; 
  IONE=1;
  ITWO=2;


  if(ed <= st){
    printf("TRD_type 2H: ERROR st and ed %d  %d \n",st,ed);
    exit(-10);;
  }

  LDX  = LDA -1;
  KDM2 = NB-2;
  for (i = ed; i >= st+1 ; i--){
     // apply Householder from the right. and create newnnz outside the band if J3 < N
     J1  = ed+1;
     J2  = min((i+1+KDM2), N);
     J3  = min((J2+1), N);
     len = J2-J1+1;
     DLARFX_C('R', len, *V(i), *TAU(i), A(J1,(i-1)), LDX);

     // if nonzero element a(j+kd,j-1) has been created outside the band (if index < N) then eliminate it.
     len    = J3-J2; // soit 1 soit 0
     if(len>0){ 
         //  new nnz at TEMP=V(J3)  
         *V(J3)     =            - *A(J3,(i)) * (*TAU(i)) * (*V(i)); 
         *A(J3,(i)) = *A(J3,(i)) + *V(J3)  * (*V(i)); //ATTENTION THIS replacement IS VALID IN FLOAT CASE NOT IN COMPLEX
         // generate Householder to annihilate a(j+kd,j-1) within the band
         DLARFG( &ITWO, A(J2,(i-1)), V(J3), &IONE, TAU(J3) );
     }
  }
	       //if(id==1) return;

  
  for (i = ed; i >= st+1 ; i--){
     J2  = min((i+1+KDM2), N);
     J3  = min((J2+1), N);
     len    = J3-J2;
     if(len>0){
        len = min(ed,N)-i+1;
        DLARFX_C('L', len , *V(J3), *TAU(J3), A(J2, i), LDX);
     }
  }

}
#undef A
#undef V
#undef TAU
///////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////
//                  TYPE 3-BAND Householder
///////////////////////////////////////////////////////////
//// add -1 because of C
#define A(m,n)   &(A[((m)-(n)) + LDA*((n)-1)])
#define V(m)     &(V[m-1])
#define TAU(m)   &(TAU[m-1])
static void TRD_type3bHL(int N, int NB, Dague_Complex64_t *A, int LDA, Dague_Complex64_t *V, Dague_Complex64_t *TAU, int st, int ed) {
  int    J1, J2, len, LDX;
  int    i;
  (void)NB;

  if(ed <= st){
    printf("TRD_type 3H: ERROR st and ed %d  %d \n",st,ed);
    exit(-10);
  }

  LDX = LDA-1;
  for (i = ed; i >= st+1 ; i--){
     // apply rotation from the left. faire le DROT horizontal
     J1  = st;
     J2  = i;
     len = J2-J1+1;
     //len = NB+1;
     DLARFX_C('B', len , *V(i), *TAU(i), A((i-1),J1   ), LDX);     
  }

  for (i = ed; i >= st+1 ; i--){
     len = min(ed,N)-i;
     if(len>0)DLARFX_C('R', len, *V(i), *TAU(i), A((i+1),(i-1)), LDX);
  }
}
#undef A
#undef V
#undef TAU
///////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////
//                  grouping sched wrapper call
///////////////////////////////////////////////////////////
int TRD_seqgralgtype(int N, int NB, Dague_Complex64_t *A, int LDA, Dague_Complex64_t *C, Dague_Complex64_t *S, int i, int j, int m, int grsiz, int BAND) {
  int    k,shift=3;
  int    myid,colpt,stind,edind,blklastind,stepercol;
  (void) BAND;


  
  k   = shift/grsiz;
  stepercol =  k*grsiz == shift ? k:k+1;


   for (k = 1; k <=grsiz; k++){ 
      myid = (i-j)*(stepercol*grsiz) +(m-1)*grsiz + k;	
      if(myid%2 ==0){
           colpt      = (myid/2)*NB+1+j-1;
           stind      = colpt-NB+1;
           edind      = min(colpt,N);
           blklastind = colpt;
           if(stind>=edind){
               printf("TRD_seqalg ERROR---------> st>=ed  %d  %d \n\n",stind, edind);
               return -10;
           }
      }else{
           colpt      = ((myid+1)/2)*NB + 1 +j -1 ;
           stind      = colpt-NB+1;
           edind      = min(colpt,N);
           if( (stind>=edind-1) && (edind==N) )
               blklastind=N;
           else
               blklastind=0;
           if(stind>=edind){
               printf("TRD_seqalg ERROR---------> st>=ed  %d  %d \n\n",stind, edind);
               return -10;

           }
      }

          if(myid == 1)
              TRD_type1bHL(N, NB, A, LDA, C, S, stind, edind);
          else if(myid%2 == 0)
              TRD_type2bHL(N, NB, A, LDA, C, S, stind, edind);
          else if(myid%2 == 1)
              TRD_type3bHL(N, NB, A, LDA, C, S, stind, edind);
          else{
              printf("COUCOU ERROR myid/2 %d\n",myid);
               return -10;
          }

      if(blklastind >= (N-1))  break;
      //if(myid==2)return;
  }   // END for k=1:grsiz
return 0;
}
///////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                  REDUCTION BAND TO TRIDIAG
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void band_to_trd_v8seq(int N, int NB, Dague_Complex64_t *A, int LDA, int INgrsiz, int INthgrsiz) {
  int myid, grsiz, shift, stt, st, ed, stind, edind, BAND;
  int blklastind, colpt;
  int stepercol,mylastid;
  Dague_Complex64_t *C, *S;
  int i,j,m;
  int thgrsiz, thgrnb, thgrid, thed;
  int INFO;
  INFO=-1;
  BAND = 0;
  C   = malloc(N*sizeof(Dague_Complex64_t));
  S   = malloc(N*sizeof(Dague_Complex64_t));
  memset(C,0,N*sizeof(Dague_Complex64_t));
  memset(S,0,N*sizeof(Dague_Complex64_t));

  grsiz   = INgrsiz;
  thgrsiz = INthgrsiz;

  shift   = 3;
  if(grsiz==0)grsiz   = 6;
  if(thgrsiz==0)thgrsiz = N;
  
  if(LDA != (NB+1))
  {
      printf(" ERROR LDA not equal NB+1 and this code is special for LDA=NB+1. LDA=%d NB+1=%d \n",LDA,NB+1);
      return;
  }
  printf("  Version -8seq- grsiz %4d     thgrsiz %4d       N %5d      NB %5d    BAND %5d\n",grsiz,thgrsiz, N, NB, BAND);


  i = shift/grsiz;
  stepercol =  i*grsiz == shift ? i:i+1;

  i       = (N-2)/thgrsiz;
  thgrnb  = i*thgrsiz == (N-2) ? i:i+1;

  for (thgrid = 1; thgrid<=thgrnb; thgrid++){
     stt  = (thgrid-1)*thgrsiz+1;
     thed = min( (stt + thgrsiz -1), (N-2));
     for (i = stt; i <= N-2; i++){
        ed=min(i,thed);
        if(stt>ed)break;
        for (m = 1; m <=stepercol; m++){ 
            st=stt;
            for (j = st; j <=ed; j++){
                 myid     = (i-j)*(stepercol*grsiz) +(m-1)*grsiz + 1;
                 mylastid = myid+grsiz-1;
                 INFO = TRD_seqgralgtype(N, NB, A, LDA, C, S, i, j, m, grsiz, BAND);
                 if(INFO!=0){
			 printf("ERROR band_to_trd_v8seq INFO=%d\n",INFO);
			 return;
		 }
                 if(mylastid%2 ==0){
                     blklastind      = (mylastid/2)*NB+1+j-1;
                 }else{
                      colpt      = ((mylastid+1)/2)*NB + 1 +j -1 ;
                      stind      = colpt-NB+1;
                      edind      = min(colpt,N);
                      if( (stind>=edind-1) && (edind==N) )
                          blklastind=N;
                      else
                          blklastind=0;
                 }
                 if(blklastind >= (N-1))  stt=stt+1;
    	   } // END for j=st:ed
        } // END for m=1:stepercol
     } // END for i=1:N-2
  } // END for thgrid=1:thgrnb

} // END FUNCTION
///////////////////////////////////////////////////////////////////////////////////////////////////////////////







///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int blgchase_ztrdv1(int NT, int N, int NB, Dague_Complex64_t *A, Dague_Complex64_t *V, Dague_Complex64_t *TAU, int sweep, int id, int blktile) {
  int    /*edloc,*/ stloc, st, ed, KDM1, LDA;

  (void)NT;
  KDM1   = NB-1;
  LDA    = NB+1;
  /* generate the indiceslocal and global*/
  stloc  = (sweep+1)%NB;
  if(stloc==0) stloc=NB;
  /*if(id==NT-1)
     edloc = NB-1;
  else
     edloc = stloc + KDM1;
  */

  st = min(id*NB+stloc, N-1);
  ed = min(st+KDM1, N-1);
  /*
   * i     = (N-1)%ed;
   * edloc = i%NB+NB;
   */

  /* quick return in case of last tile */
  if(st==ed)
     return 0;

  /* because the kernel have been writted for fortran, add 1 */
   st = st +1;
   ed = ed +1;
  /* code for all tiles */
  if(id==blktile){
     TRD_type1bHL(N, NB, A, LDA, V, TAU, st, ed);
     TRD_type2bHL(N, NB, A, LDA, V, TAU, st, ed);
  }else{
     TRD_type3bHL(N, NB, A, LDA, V, TAU, st, ed);
     //if(id==6) return;
     TRD_type2bHL(N, NB, A, LDA, V, TAU, st, ed);
  }
  return 0;
}

#if 0
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void band_to_trd_vmpi1(int N, int NB, Dague_Complex64_t *A, int LDA) {
  int NT;
  Dague_Complex64_t *V, *TAU;
  int blktile, S, id, sweep;
  V   = malloc(N*sizeof(Dague_Complex64_t));
  TAU = malloc(N*sizeof(Dague_Complex64_t));
  memset(V,0,N*sizeof(Dague_Complex64_t));
  memset(TAU,0,N*sizeof(Dague_Complex64_t));


  NT = N/NB;
  if(NT*NB != N){
      printf("ERROR NT*NB not equal N \n");
      return;
  }

  //printf("voici NT, N NB %d %d %d\n",NT,N,NB);
  for (blktile = 0; blktile<NT; blktile++){
     for (S = 0; S<NB; S++){
        sweep = blktile*NB + S ;
        for (id = blktile; id<NT; id++){
               blgchase_ztrdv1 (NT, N, NB, A, V, TAU, sweep , id, blktile);
	}
     }
  }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif




///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int blgchase_ztrdv2(int NT, int N, int NB, Dague_Complex64_t *A1, Dague_Complex64_t *A2, Dague_Complex64_t *V1, Dague_Complex64_t *TAU1, Dague_Complex64_t *V2, Dague_Complex64_t *TAU2, int sweep, int id, int blktile) {
  int    edloc, stloc, st, ed, KDM1, LDA;

  KDM1   = NB-1;
  LDA    = NB+1;
  /* generate the indiceslocal and global*/
  stloc  = (sweep+1)%NB;
  if(stloc==0) stloc=NB;
  if(id==NT-1)
     edloc = NB-1;
  else
     edloc = stloc + KDM1;

  st = min(id*NB+stloc, N-1);
  ed = min(st+KDM1, N-1);
  /*
   * i     = (N-1)%ed;
   * edloc = i%NB+NB;
   */

  /* quick return in case of last tile */
  if(st==ed)
     return 0;


  /* code for all tiles */
  if(id==blktile){
     CORE_zhbtelr(N, NB, A1, LDA, A2, LDA, V1, TAU1, stloc, edloc);
     CORE_zhbtrce(N, NB, A1, LDA, A2, LDA, V1, TAU1, V2, TAU2, stloc, edloc, ed);
  }else{
     CORE_zhbtlrx(N, NB, A1, LDA, A2, LDA, V1, TAU1, stloc, edloc);
     CORE_zhbtrce(N, NB, A1, LDA, A2, LDA, V1, TAU1, V2, TAU2, stloc, edloc, ed);
  }
  return 0;
}

#if 0
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void band_to_trd_vmpi2(int N, int NB, Dague_Complex64_t *A, int LDA) {
    int NT;
    Dague_Complex64_t *V, *TAU;
    int blktile, S, id, sweep;
    V   = malloc(N*sizeof(Dague_Complex64_t));
    TAU = malloc(N*sizeof(Dague_Complex64_t));
    memset(V,0,N*sizeof(Dague_Complex64_t));
    memset(TAU,0,N*sizeof(Dague_Complex64_t));


    NT = N/NB;
    if(NT*NB != N){
        printf("ERROR NT*NB not equal N \n");
        return;
    }

    //printf("voici NT, N NB %d %d %d\n",NT,N,NB);
    for (blktile = 0; blktile<NT; blktile++){
        for (S = 0; S<NB; S++){
            sweep = blktile*NB + S ;
            for (id = blktile; id<NT; id++){
                //printf("voici  blktile %d    S %d     id %d   sweep %d   \n",blktile, S, id, sweep); 
                blgchase_ztrdv2 (NT, N, NB,
                                A+(id*NB*LDA), A+((id+1)*NB*LDA),
                                V+(id*NB), TAU+(id*NB),
                                V+((id+1)*NB), TAU+((id+1)*NB),
                                sweep, id, blktile);
            }
            //if(sweep==0) return;
        }
    }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif




