/* 
 * -- High Performance Computing Linpack Benchmark (HPL)                
 *    HPL - 2.0 - September 10, 2008                          
 *    Antoine P. Petitet                                                
 *    University of Tennessee, Knoxville                                
 *    Innovative Computing Laboratory                                 
 *    (C) Copyright 2000-2008 All Rights Reserved                       
 *                                                                      
 * -- Copyright notice and Licensing terms:                             
 *                                                                      
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:                                                             
 *                                                                      
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.        
 *                                                                      
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. 
 *                                                                      
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 */ 

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define    HPL_MULT0         1284865837
#define    HPL_MULT1         1481765933
#define    HPL_IADD0         1
#define    HPL_IADD1         0
#define    HPL_DIVFAC        2147483648.0
#define    HPL_POW16         65536.0
#define    HPL_HALF          0.5

void HPL_ladd
(
   int *                            J,
   int *                            K,
   int *                            I
)
{
/* 
 * Purpose
 * =======
 *
 * HPL_ladd adds  without carry two long positive integers  K and J and
 * puts the result into I. The long integers  I, J, K are encoded on 64
 * bits using an array of 2 integers.  The 32-lower bits  are stored in
 * the  first  entry  of each array,  the 32-higher bits  in the second
 * entry.
 *
 * Arguments
 * =========
 *
 * J       (local input)                 int *
 *         On entry, J is an integer array of dimension 2 containing the
 *         encoded long integer J.
 *
 * K       (local input)                 int *
 *         On entry, K is an integer array of dimension 2 containing the
 *         encoded long integer K.
 *
 * I       (local output)                int *
 *         On entry, I is an integer array of dimension 2. On exit, this
 *         array contains the encoded long integer result.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   unsigned int        itmp0, itmp1;
   unsigned int        ktmp0 = K[0] & 65535, ktmp1 = (unsigned)K[0] >> 16;
   unsigned int        ktmp2 = K[1] & 65535, ktmp3 = (unsigned)K[1] >> 16;
   unsigned int        jtmp0 = J[0] & 65535, jtmp1 = (unsigned)J[0] >> 16;
   unsigned int        jtmp2 = J[1] & 65535, jtmp3 = (unsigned)J[1] >> 16;

/* ..
 * .. Executable Statements ..
 */
/*
 *    K[1] K[0] K  I[0]  = (K[0]+J[0]) % 2^32
 *    XXXX XXXX    carry = (K[0]+J[0]) / 2^32
 *
 * +  J[1] J[0] J  I[1] = K[1] + J[1] + carry
 *    XXXX XXXX    I[1] = I[1] % 2^32
 *    -------------
 *    I[1] I[0]
 *    0XXX XXXX I
 */
   itmp0 = ktmp0 + jtmp0;
   itmp1 = itmp0 >> 16;         I[0] = itmp0 - (itmp1 << 16 );
   itmp1 += ktmp1 + jtmp1;      I[0] |= (itmp1 & 65535) << 16;
   itmp0 = (itmp1 >> 16) + ktmp2 + jtmp2;
   I[1] = itmp0 - ((itmp0 >> 16 ) << 16);
   itmp1 = (itmp0 >> 16) + ktmp3 + jtmp3;
   I[1] |= (itmp1 & 65535) << 16;
/*
 * End of HPL_ladd
 */
}

void HPL_lmul
(
   int *                            K,
   int *                            J,
   int *                            I
)
{
/* 
 * Purpose
 * =======
 *
 * HPL_lmul multiplies  without carry two long positive integers K and J
 * and puts the result into I. The long integers  I, J, K are encoded on
 * 32 bits using an array of 2 integers. The 32-lower bits are stored in
 * the first entry of each array, the 32-higher bits in the second entry
 * of each array. For efficiency purposes, the  intrisic modulo function
 * is inlined.
 *
 * Arguments
 * =========
 *
 * K       (local input)                 int *
 *         On entry, K is an integer array of dimension 2 containing the
 *         encoded long integer K.
 *
 * J       (local input)                 int *
 *         On entry, J is an integer array of dimension 2 containing the
 *         encoded long integer J.
 *
 * I       (local output)                int *
 *         On entry, I is an integer array of dimension 2. On exit, this
 *         array contains the encoded long integer result.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   int                        r, c;
   unsigned int               kk[4], jj[4], res[5];
/* ..
 * .. Executable Statements ..
 */
/*
 * Addition is done with 16 bits at a time. Multiplying two 16-bit
 * integers yields a 32-bit result. The lower 16-bits of the result
 * are kept in I, and the higher 16-bits are carried over to the
 * next multiplication.
 */
   for (c = 0; c < 2; ++c) {
     kk[2*c] = K[c] & 65535;
     kk[2*c+1] = ((unsigned)K[c] >> 16) & 65535;
     jj[2*c] = J[c] & 65535;
     jj[2*c+1] = ((unsigned)J[c] >> 16) & 65535;
   }

   res[0] = 0;
   for (c = 0; c < 4; ++c) {
     res[c+1] = (res[c] >> 16) & 65535;
     res[c] &= 65535;
     for (r = 0; r < c+1; ++r) {
       res[c] = kk[r] * jj[c-r] + (res[c] & 65535);
       res[c+1] += (res[c] >> 16) & 65535;
     }
   }

   for (c = 0; c < 2; ++c)
     I[c] = (int)(((res[2*c+1] & 65535) << 16) | (res[2*c] & 65535));
/*
 * End of HPL_lmul
 */
}

void HPL_xjumpm
(
   const int                        JUMPM,
   int *                            MULT,
   int *                            IADD,
   int *                            IRANN,
   int *                            IRANM,
   int *                            IAM,
   int *                            ICM
)
{
/* 
 * Purpose
 * =======
 *
 * HPL_xjumpm computes  the constants  A and C  to jump JUMPM numbers in
 * the random sequence: X(n+JUMPM) = A*X(n)+C.  The constants encoded in
 * MULT and IADD  specify  how to jump from one entry in the sequence to
 * the next.
 *
 * Arguments
 * =========
 *
 * JUMPM   (local input)                 const int
 *         On entry,  JUMPM  specifies  the  number  of entries  in  the
 *         sequence to jump over. When JUMPM is less or equal than zero,
 *         A and C are not computed, IRANM is set to IRANN corresponding
 *         to a jump of size zero.
 *
 * MULT    (local input)                 int *
 *         On entry, MULT is an array of dimension 2,  that contains the
 *         16-lower  and 15-higher bits of the constant  a  to jump from
 *         X(n) to X(n+1) = a*X(n) + c in the random sequence.
 *
 * IADD    (local input)                 int *
 *         On entry, IADD is an array of dimension 2,  that contains the
 *         16-lower  and 15-higher bits of the constant  c  to jump from
 *         X(n) to X(n+1) = a*X(n) + c in the random sequence.
 *
 * IRANN   (local input)                 int *
 *         On entry, IRANN is an array of dimension 2. that contains the
 *         16-lower and 15-higher bits of the encoding of X(n).
 *
 * IRANM   (local output)                int *
 *         On entry,  IRANM  is an array of dimension 2.   On exit, this
 *         array  contains respectively  the 16-lower and 15-higher bits
 *         of the encoding of X(n+JUMPM).
 *
 * IAM     (local output)                int *
 *         On entry, IAM is an array of dimension 2. On exit, when JUMPM
 *         is  greater  than  zero,  this  array  contains  the  encoded
 *         constant  A  to jump from  X(n) to  X(n+JUMPM)  in the random
 *         sequence. IAM(0:1)  contains  respectively  the  16-lower and
 *         15-higher  bits  of this constant  A. When  JUMPM  is less or
 *         equal than zero, this array is not referenced.
 *
 * ICM     (local output)                int *
 *         On entry, ICM is an array of dimension 2. On exit, when JUMPM
 *         is  greater  than  zero,  this  array  contains  the  encoded
 *         constant  C  to jump from  X(n)  to  X(n+JUMPM) in the random
 *         sequence. ICM(0:1)  contains  respectively  the  16-lower and
 *         15-higher  bits  of this constant  C. When  JUMPM  is less or
 *         equal than zero, this array is not referenced.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   int                        j[2], k;
/* ..
 * .. Executable Statements ..
 */
   if( JUMPM > 0 )
   {
      IAM[0] = MULT[0]; IAM[1] = MULT[1];   /* IAM   = MULT;          */
      ICM[0] = IADD[0]; ICM[1] = IADD[1];   /* ICM   = IADD;          */
      for( k = 1; k <= JUMPM-1; k++ )
      {
         HPL_lmul( IAM, MULT, j );          /* j     = IAM   * MULT;  */
         IAM[0] = j[0]; IAM[1] = j[1];      /* IAM   = j;             */
         HPL_lmul( ICM, MULT, j );          /* j     = ICM   * MULT;  */
         HPL_ladd( IADD, j, ICM );          /* ICM   = IADD  + j;     */
      }
      HPL_lmul( IRANN, IAM, j );            /* j     = IRANN * IAM;   */
      HPL_ladd( j, ICM, IRANM );            /* IRANM = j     + ICM;   */
   }
   else
   {                                        /* IRANM = IRANN          */
      IRANM[0] = IRANN[0]; IRANM[1] = IRANN[1];
   }
/*
 * End of HPL_xjumpm
 */
}

/*
 * ---------------------------------------------------------------------
 * Static variables
 * ---------------------------------------------------------------------
 */
static int       ias[2], ics[2], irand[2];

void HPL_setran
(
   const int                        OPTION,
   int *                            IRAN
)
{
/* 
 * Purpose
 * =======
 *
 * HPL_setran initializes  the random generator with the encoding of the
 * first number X(0) in the sequence,  and the constants a and c used to
 * compute the next element in the sequence: X(n+1) = a*X(n) + c.  X(0),
 * a and c are stored in the static variables  irand, ias and ics.  When
 * OPTION is 0 (resp. 1 and 2),  irand  (resp. ia and ic)  is set to the
 * values of the input array IRAN.  When OPTION is 3, IRAN is set to the
 * current value of irand, and irand is then incremented.
 *
 * Arguments
 * =========
 *
 * OPTION  (local input)                 const int
 *         On entry, OPTION  is an integer that specifies the operations
 *         to be performed on the random generator as specified above.
 *
 * IRAN    (local input/output)          int *
 *         On entry,  IRAN is an array of dimension 2, that contains the
 *         16-lower and 15-higher bits of a random number.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   int                        j[2];
/* ..
 * .. Executable Statements ..
 */
   if(      OPTION == 3 )
   {                                       /* return current value */
      IRAN[0] = irand[0]; IRAN[1] = irand[1];
      HPL_lmul( irand, ias, j );         /* j     = irand * ias;   */
      HPL_ladd( j, ics, irand );         /* irand = j     + ics;   */
   } 
   else if( OPTION == 0 ) { irand[0] = IRAN[0]; irand[1] = IRAN[1]; }
   else if( OPTION == 1 ) { ias  [0] = IRAN[0]; ias  [1] = IRAN[1]; }
   else if( OPTION == 2 ) { ics  [0] = IRAN[0]; ics  [1] = IRAN[1]; }
/*
 * End of HPL_setran
 */
}

void HPL_jumpit
(
   int *                            MULT,
   int *                            IADD,
   int *                            IRANN,
   int *                            IRANM
)
{
/* 
 * Purpose
 * =======
 *
 * HPL_jumpit jumps in the random sequence from the number  X(n) encoded
 * in IRANN to the number  X(m)  encoded in  IRANM using the constants A
 * and C encoded in MULT and IADD: X(m) = A * X(n) + C.  The constants A
 * and C obviously depend on m and n,  see  the function  HPL_xjumpm  in
 * order to initialize them.
 *
 * Arguments
 * =========
 *
 * MULT    (local input)                 int *
 *         On entry, MULT is an array of dimension 2, that contains the
 *         16-lower and 15-higher bits of the constant A.
 *
 * IADD    (local input)                 int *
 *         On entry, IADD is an array of dimension 2, that contains the
 *         16-lower and 15-higher bits of the constant C.
 *
 * IRANN   (local input)                 int *
 *         On entry,  IRANN  is an array of dimension 2,  that contains 
 *         the 16-lower and 15-higher bits of the encoding of X(n).
 *
 * IRANM   (local output)                int *
 *         On entry,  IRANM  is an array of dimension 2.  On exit, this
 *         array contains respectively the 16-lower and  15-higher bits
 *         of the encoding of X(m).
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   int                          j[2];
/* ..
 * .. Executable Statements ..
 */
   HPL_lmul( IRANN, MULT, j );              /* j     = IRANN * MULT;  */
   HPL_ladd( j, IADD, IRANM );              /* IRANM = j     + IADD;  */
   HPL_setran( 0, IRANM );                  /* irand = IRANM          */
/*
 * End of HPL_jumpit
 */
}

/*
 * MnumrocI computes the # of local indexes  np_ residing in the process
 * of coordinate  proc_  corresponding to the interval of global indexes
 * i_:i_+n_-1  assuming  that the global index 0 resides in  the process
 * src_,  and that the indexes are distributed from src_ using the para-
 * meters inb_, nb_ and nprocs_.
 */
#define    MnumrocI( np_, n_, i_, inb_, nb_, proc_, src_, nprocs_ )    \
           {                                                           \
              if( ( (src_) >= 0 ) && ( (nprocs_) > 1 ) )               \
              {                                                        \
                 int inb__, mydist__, n__, nblk__, quot__, src__;      \
                 if( ( inb__ = (inb_) - (i_) ) <= 0 )                  \
                 {                                                     \
                    nblk__ = (-inb__) / (nb_) + 1;                     \
                    src__  = (src_) + nblk__;                          \
                    src__ -= ( src__ / (nprocs_) ) * (nprocs_);        \
                    inb__ += nblk__*(nb_);                             \
                    if( ( n__ = (n_) - inb__ ) <= 0 )                  \
                    {                                                  \
                       if( (proc_) == src__ ) np_ = (n_);              \
                       else                   np_ = 0;                 \
                    }                                                  \
                    else                                               \
                    {                                                  \
                       if( ( mydist__ = (proc_) - src__ ) < 0 )        \
                          mydist__ += (nprocs_);                       \
                       nblk__    = n__ / (nb_) + 1;                    \
                       mydist__ -= nblk__ -                            \
                          (quot__ = (nblk__ / (nprocs_))) * (nprocs_); \
                       if( mydist__ < 0 )                              \
                       {                                               \
                          if( (proc_) != src__ )                       \
                             np_ = (nb_) + (nb_) * quot__;             \
                          else                                         \
                             np_ = inb__ + (nb_) * quot__;             \
                       }                                               \
                       else if( mydist__ > 0 )                         \
                       {                                               \
                          np_ = (nb_) * quot__;                        \
                       }                                               \
                       else                                            \
                       {                                               \
                          if( (proc_) != src__ )                       \
                             np_ = n__ +(nb_)+(nb_)*(quot__ - nblk__); \
                          else                                         \
                             np_ = (n_)+      (nb_)*(quot__ - nblk__); \
                       }                                               \
                    }                                                  \
                 }                                                     \
                 else                                                  \
                 {                                                     \
                    if( ( n__ = (n_) - inb__ ) <= 0 )                  \
                    {                                                  \
                       if( (proc_) == (src_) ) np_ = (n_);             \
                       else                    np_ = 0;                \
                    }                                                  \
                    else                                               \
                    {                                                  \
                       if( ( mydist__ = (proc_) - (src_) ) < 0 )       \
                          mydist__ += (nprocs_);                       \
                       nblk__    = n__ / (nb_) + 1;                    \
                       mydist__ -= nblk__ -                            \
                          ( quot__ = (nblk__ / (nprocs_)) )*(nprocs_); \
                       if( mydist__ < 0 )                              \
                       {                                               \
                          if( (proc_) != (src_) )                      \
                             np_ = (nb_) + (nb_) * quot__;             \
                          else                                         \
                             np_ = inb__ + (nb_) * quot__;             \
                       }                                               \
                       else if( mydist__ > 0 )                         \
                       {                                               \
                          np_ = (nb_) * quot__;                        \
                       }                                               \
                       else                                            \
                       {                                               \
                          if( (proc_) != (src_) )                      \
                             np_ = n__ +(nb_)+(nb_)*(quot__ - nblk__); \
                          else                                         \
                             np_ = (n_)+      (nb_)*(quot__ - nblk__); \
                       }                                               \
                    }                                                  \
                 }                                                     \
              }                                                        \
              else                                                     \
              {                                                        \
                 np_ = (n_);                                           \
              }                                                        \
           }

#define    Mnumroc( np_, n_, inb_, nb_, proc_, src_, nprocs_ )         \
           MnumrocI( np_, n_, 0, inb_, nb_, proc_, src_, nprocs_ )


void HPL_pdmatgen
(
   const int                        nprow,
   const int                        npcol,
   const int                        myrow,
   const int                        mycol,
   const int                        M,
   const int                        N,
   const int                        NB,
   double *                         A,
   const int                        LDA,
   const int                        ISEED
)
{
/* 
 * Purpose
 * =======
 *
 * HPL_pdmatgen generates (or regenerates) a parallel random matrix A.
 *  
 * The  pseudo-random  generator uses the linear congruential algorithm:
 * X(n+1) = (a * X(n) + c) mod m  as  described  in the  Art of Computer
 * Programming, Knuth 1973, Vol. 2.
 *
 * Arguments
 * =========
 *
 * nprow   (local input)                 const int
 *         On entry, the number of processors per row.
 *
 * npcol   (local input)                 const int
 *         On entry, the number of processors per column.
 *
 * myrow   (local input)                 const int
 *         On entry, this processor row position on the
 *         processors grid.
 *
 * mycol   (local input)                 const int
 *         On entry, this processor column position on the
 *         processors grid.
 *
 * GRID    (local input)                 const HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * M       (global input)                const int
 *         On entry,  M  specifies  the number  of rows of the matrix A.
 *         M must be at least zero.
 *
 * N       (global input)                const int
 *         On entry,  N specifies the number of columns of the matrix A.
 *         N must be at least zero.
 *
 * NB      (global input)                const int
 *         On entry,  NB specifies the blocking factor used to partition
 *         and distribute the matrix A. NB must be larger than one.
 *
 * A       (local output)                double *
 *         On entry,  A  points  to an array of dimension (LDA,LocQ(N)).
 *         On exit, this array contains the coefficients of the randomly
 *         generated matrix.
 *
 * LDA     (local input)                 const int
 *         On entry, LDA specifies the leading dimension of the array A.
 *         LDA must be at least max(1,LocP(M)).
 *
 * ISEED   (global input)                const int
 *         On entry, ISEED  specifies  the  seed  number to generate the
 *         matrix A. ISEED must be at least zero.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   int                        iadd [2], ia1  [2], ia2  [2], ia3  [2],
                              ia4  [2], ia5  [2], ib1  [2], ib2  [2],
                              ib3  [2], ic1  [2], ic2  [2], ic3  [2],
                              ic4  [2], ic5  [2], iran1[2], iran2[2],
                              iran3[2], iran4[2], itmp1[2], itmp2[2],
                              itmp3[2], jseed[2], mult [2];
   int                        ib, iblk, ik, jb, jblk, jk, jump1, jump2,
                              jump3, jump4, jump5, jump6, jump7, lmb,
                              lnb, mblks, mp, nblks, nq;
/* ..
 * .. Executable Statements ..
 */
   mult [0] = HPL_MULT0; mult [1] = HPL_MULT1;
   iadd [0] = HPL_IADD0; iadd [1] = HPL_IADD1;
   jseed[0] = ISEED;     jseed[1] = 0;
/*
 * Generate an M by N matrix starting in process (0,0)
 */
   Mnumroc( mp, M, NB, NB, myrow, 0, nprow );
   Mnumroc( nq, N, NB, NB, mycol, 0, npcol );

   if( ( mp <= 0 ) || ( nq <= 0 ) ) return;
/*
 * Local number of blocks and size of the last one
 */
   mblks = ( mp + NB - 1 ) / NB; lmb = mp - ( ( mp - 1 ) / NB ) * NB;
   nblks = ( nq + NB - 1 ) / NB; lnb = nq - ( ( nq - 1 ) / NB ) * NB;
/*
 * Compute multiplier/adder for various jumps in random sequence
 */
   jump1 = 1;  jump2 = nprow * NB; jump3 = M; jump4 = npcol * NB;
   jump5 = NB; jump6 = mycol;      jump7 = myrow * NB;

   HPL_xjumpm( jump1, mult, iadd, jseed, iran1, ia1,   ic1   );
   HPL_xjumpm( jump2, mult, iadd, iran1, itmp1, ia2,   ic2   );
   HPL_xjumpm( jump3, mult, iadd, iran1, itmp1, ia3,   ic3   );
   HPL_xjumpm( jump4, ia3,  ic3,  iran1, itmp1, ia4,   ic4   );
   HPL_xjumpm( jump5, ia3,  ic3,  iran1, itmp1, ia5,   ic5   );
   HPL_xjumpm( jump6, ia5,  ic5,  iran1, itmp3, itmp1, itmp2 );
   HPL_xjumpm( jump7, mult, iadd, itmp3, iran1, itmp1, itmp2 );
   HPL_setran( 0, iran1 ); HPL_setran( 1, ia1 ); HPL_setran( 2, ic1 );
/*
 * Save value of first number in sequence
 */
   ib1[0] = iran1[0]; ib1[1] = iran1[1];
   ib2[0] = iran1[0]; ib2[1] = iran1[1];
   ib3[0] = iran1[0]; ib3[1] = iran1[1];

   for( jblk = 0; jblk < nblks; jblk++ )
   {
      jb = ( jblk == nblks - 1 ? lnb : NB );
      for( jk = 0; jk < jb; jk++ )
      {
         for( iblk = 0; iblk < mblks; iblk++ )
         {
            ib = ( iblk == mblks - 1 ? lmb : NB );
            for( ik = 0; ik < ib; A++, ik++ ) *A = HPL_rand();
            HPL_jumpit( ia2, ic2, ib1, iran2 );
            ib1[0] = iran2[0]; ib1[1] = iran2[1];
         }
         A += LDA - mp;
         HPL_jumpit( ia3, ic3, ib2, iran3 );
         ib1[0] = iran3[0]; ib1[1] = iran3[1];
         ib2[0] = iran3[0]; ib2[1] = iran3[1];
      }
      HPL_jumpit( ia4, ic4, ib3, iran4 );
      ib1[0] = iran4[0]; ib1[1] = iran4[1];
      ib2[0] = iran4[0]; ib2[1] = iran4[1];
      ib3[0] = iran4[0]; ib3[1] = iran4[1];
   }
/*
 * End of HPL_pdmatgen
 */
}

