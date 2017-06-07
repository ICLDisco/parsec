/*
 * Copyright (c) 2016-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef ATOMIC_EXTERNAL_H_HAS_BEEN_INCLUDED
#define ATOMIC_EXTERNAL_H_HAS_BEEN_INCLUDED

#if defined(PARSEC_ATOMIC_ACCESS_TO_INTERNALS_ALLOWED)
#error "This file should never be used while building PaRSEC internally"
#endif  /* defined(PARSEC_ATOMIC_ACCESS_TO_INTERNALS_ALLOWED) */

BEGIN_C_DECLS

PARSEC_DECLSPEC void parsec_mfence(void);
PARSEC_DECLSPEC void parsec_atomic_wmb(void);
PARSEC_DECLSPEC void parsec_atomic_rmb(void);

PARSEC_DECLSPEC int32_t
parsec_atomic_cas_32b(volatile uint32_t* location,
                      uint32_t old_value,
                      uint32_t new_value);
PARSEC_DECLSPEC int32_t
parsec_atomic_cas_64b(volatile uint64_t* location,
                      uint64_t old_value,
                      uint64_t new_value);

#if defined(HAVE_UINT128b)
PARSEC_DECLSPEC int32_t
parsec_atomic_cas_128b(volatile __uint128_t* location,
                       __uint128_t old_value,
                       __uint128_t new_value);
#endif  /* defined(HAVE_UINT128b) */

PARSEC_DECLSPEC int32_t
parsec_atomic_cas_ptr(volatile void* location,
                      void* old_value,
                      void* new_value);

PARSEC_DECLSPEC uint32_t parsec_atomic_bor(uint32_t*, uint32_t);

PARSEC_DECLSPEC uint32_t parsec_atomic_add_32b( volatile uint32_t *location, int32_t );
PARSEC_DECLSPEC uint32_t parsec_atomic_sub_32b( volatile uint32_t *location, int32_t );
PARSEC_DECLSPEC uint32_t parsec_atomic_inc_32b( volatile uint32_t *location );
PARSEC_DECLSPEC uint32_t parsec_atomic_dec_32b( volatile uint32_t *location );

PARSEC_DECLSPEC uint64_t parsec_atomic_add_64b( volatile uint64_t *location, int64_t );
PARSEC_DECLSPEC uint64_t parsec_atomic_sub_64b( volatile uint64_t *location, int64_t );
PARSEC_DECLSPEC uint64_t parsec_atomic_inc_64b( volatile uint64_t *location );
PARSEC_DECLSPEC uint64_t parsec_atomic_dec_64b( volatile uint64_t *location );

#if defined(HAVE_UINT128b)
PARSEC_DECLSPEC __uint128_t parsec_atomic_add_128b( volatile __uint128_t *location, __uint128_t );
PARSEC_DECLSPEC __uint128_t parsec_atomic_sub_128b( volatile __uint128_t *location, __uint128_t );
PARSEC_DECLSPEC __uint128_t parsec_atomic_inc_128b( volatile __uint128_t *location );
PARSEC_DECLSPEC __uint128_t parsec_atomic_dec_128b( volatile __uint128_t *location );
#endif  /* defined(HAVE_UINT128b) */

typedef int parsec_atomic_lock_t;
PARSEC_DECLSPEC void parsec_atomic_lock( parsec_atomic_lock_t* atomic_lock );
PARSEC_DECLSPEC void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock );
PARSEC_DECLSPEC long parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock );

END_C_DECLS

#endif  /* ATOMIC_EXTERNAL_H_HAS_BEEN_INCLUDED */
