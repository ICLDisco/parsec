#ifndef _DPLASMAF77_H_
#define _DPLASMAF77_H_

/** ****************************************************************************
 *  Determine FORTRAN names
 */
#if defined(NOCHANGE)
#define DPLASMA_F77NAME(lcname, UCNAME)        dplasmaf77_##lcname
#elif defined(UPCASE)
#define DPLASMA_F77NAME(lcname, UCNAME)        DPLASMAF77_##UCNAME
#else /* ADD_ */
#define DPLASMA_F77NAME(lcname, UCNAME)        dplasmaf77_##lcname##_
#endif

#define DPLASMA_ZF77NAME( _lname_ , _UNAME_ ) DPLASMA_F77NAME( z##_lname_, Z##_UNAME_ )
#define DPLASMA_CF77NAME( _lname_ , _UNAME_ ) DPLASMA_F77NAME( c##_lname_, C##_UNAME_ )
#define DPLASMA_DF77NAME( _lname_ , _UNAME_ ) DPLASMA_F77NAME( d##_lname_, D##_UNAME_ )
#define DPLASMA_SF77NAME( _lname_ , _UNAME_ ) DPLASMA_F77NAME( s##_lname_, S##_UNAME_ )

#endif
