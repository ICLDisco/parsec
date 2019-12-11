/*
 * Copyright (c) 2016-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_ATOMIC_EXTERNAL_H_HAS_BEEN_INCLUDED
#define PARSEC_ATOMIC_EXTERNAL_H_HAS_BEEN_INCLUDED

#if defined(PARSEC_ATOMIC_ACCESS_TO_INTERNALS_ALLOWED)
#error "This file should never be used while building PaRSEC internally"
#endif  /* defined(PARSEC_ATOMIC_ACCESS_TO_INTERNALS_ALLOWED) */

BEGIN_C_DECLS

/* Memory Barriers */

PARSEC_DECLSPEC void parsec_mfence(void);
PARSEC_DECLSPEC void parsec_atomic_wmb(void);
PARSEC_DECLSPEC void parsec_atomic_rmb(void);

/* Compare and Swap */

PARSEC_DECLSPEC int
parsec_atomic_cas_int32(volatile int32_t* location,
                        int32_t old_value,
                        int32_t new_value);
PARSEC_DECLSPEC int
parsec_atomic_cas_int64(volatile int64_t* location,
                        int64_t old_value,
                        int64_t new_value);

#if defined(PARSEC_HAVE_INT128)
PARSEC_DECLSPEC int
parsec_atomic_cas_int128(volatile __int128_t* location,
                         __int128_t old_value,
                         __int128_t new_value);
#endif  /* defined(PARSEC_HAVE_INT128) */

PARSEC_DECLSPEC int
parsec_atomic_cas_ptr(volatile void* location,
                      void* old_value,
                      void* new_value);

/* Mask Operations */

PARSEC_DECLSPEC int32_t parsec_atomic_fetch_or_int32(int32_t*, int32_t);
PARSEC_DECLSPEC int64_t parsec_atomic_fetch_or_int64(int64_t*, int64_t);
PARSEC_DECLSPEC int32_t parsec_atomic_fetch_and_int32(int32_t*, int32_t);
PARSEC_DECLSPEC int64_t parsec_atomic_fetch_and_int64(int64_t*, int64_t);

#if defined(PARSEC_HAVE_INT128)
PARSEC_DECLSPEC __int128_t parsec_atomic_fetch_or_int128(__int128_t*, __int128_t);
PARSEC_DECLSPEC __int128_t parsec_atomic_fetch_and_int128(__int128_t*, __int128_t);
#endif

/* Integer Operations */

PARSEC_DECLSPEC int32_t parsec_atomic_fetch_add_int32( volatile int32_t *, int32_t );
PARSEC_DECLSPEC int32_t parsec_atomic_fetch_inc_int32( volatile int32_t * );
PARSEC_DECLSPEC int32_t parsec_atomic_fetch_sub_int32( volatile int32_t *, int32_t );
PARSEC_DECLSPEC int32_t parsec_atomic_fetch_dec_int32( volatile int32_t * );

PARSEC_DECLSPEC int64_t parsec_atomic_fetch_add_int64( volatile int64_t *, int64_t );
PARSEC_DECLSPEC int64_t parsec_atomic_fetch_inc_int64( volatile int64_t * );
PARSEC_DECLSPEC int64_t parsec_atomic_fetch_sub_int64( volatile int64_t *, int64_t );
PARSEC_DECLSPEC int64_t parsec_atomic_fetch_dec_int64( volatile int64_t * );

#if defined(PARSEC_HAVE_INT128)
PARSEC_DECLSPEC __int128_t parsec_atomic_fetch_add_int128( volatile __int128_t *, __int128_t );
PARSEC_DECLSPEC __int128_t parsec_atomic_fetch_inc_int128( volatile __int128_t * );
PARSEC_DECLSPEC __int128_t parsec_atomic_fetch_sub_int128( volatile __int128_t *, __int128_t );
PARSEC_DECLSPEC __int128_t parsec_atomic_fetch_dec_int128( volatile __int128_t * );
#endif  /* defined(PARSEC_HAVE_INT128) */

/* Locks */

typedef volatile int parsec_atomic_lock_t;
PARSEC_DECLSPEC void parsec_atomic_lock( parsec_atomic_lock_t* atomic_lock );
PARSEC_DECLSPEC void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock );
PARSEC_DECLSPEC int parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock );

END_C_DECLS

#endif  /* PARSEC_ATOMIC_EXTERNAL_H_HAS_BEEN_INCLUDED */
