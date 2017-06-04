#
# Check if there is support for 128b types
#
include(CheckTypeSize)
CHECK_TYPE_SIZE( __uint128_t UINT128b )

#
# C11 include support for atomic operations via stdatomic.h but only when
# __STDC_NO_ATOMICS__ is not defined.
if( PARSEC_HAVE_STD_C1x )
  CHECK_C_SOURCE_COMPILES("
      #if __STDC_VERSION__ >= 201112L
      int main(void) { return 0; }
      #else
      #error Not C11 compliant
      #endif
      " PARSEC_COMPILER_C11_COMPLIANT)
  if( PARSEC_COMPILER_C11_COMPLIANT )
    CHECK_C_SOURCE_COMPILES("
        #if defined(__STDC_NO_ATOMICS__)
        #error Compiler is C11 compliant but does not support atomic operations
        #else
        int main(void) { return 0; }
        #endif
        " PARSEC_STDC_HAVE_C11_ATOMICS)
        if( PARSEC_STDC_HAVE_C11_ATOMICS )
# Some "C11" compilers do not define __STDC_NO_ATOMICS__ even when they don't support
# atomics (e.g. gcc-6.3.1, when using -fopenmp. See Issue #123).
          check_include_files(stdatomic.h PARSEC_ATOMIC_USE_C11_ATOMICS)
        endif( PARSEC_STDC_HAVE_C11_ATOMICS )
  endif( PARSEC_COMPILER_C11_COMPLIANT )
endif( PARSEC_HAVE_STD_C1x )
if( PARSEC_ATOMIC_USE_C11_ATOMICS )
  CHECK_C_SOURCE_COMPILES("
    #include <stdint.h>
    #include <stdatomic.h>
    int main(void) {
        int32_t where = 0, expected = 0;
        if( !atomic_compare_exchange_strong( (_Atomic int32_t*)&where, &expected, 1 ) )
            return -1;
        return 0;
    }
    " PARSEC_ATOMIC_USE_C11_32)
if(PARSEC_ATOMIC_USE_C11_32)
  CHECK_C_SOURCE_COMPILES("
    #include <stdint.h>
    #include <stdatomic.h>
    int main(void) {
        int64_t where = 0, expected = 0;
        if( !atomic_compare_exchange_strong( (_Atomic int64_t*)&where, &expected, 1 ) )
            return -1;
        return 0;
    }
    " PARSEC_ATOMIC_USE_C11_64)
  #
  # Do we need special flags to support 128 bits atomics ?
  #
  if( HAVE_UINT128b )
    CHECK_C_SOURCE_COMPILES("
       #include <stdatomic.h>
       int main(void) {
            __int128_t where = 0, expected = 0;
            if( !atomic_compare_exchange_strong( (_Atomic __uint128_t*)&where, &expected, 1 ) )
                return -1;
            return 0;
        }
        " PARSEC_ATOMIC_USE_C11_128)
    if( NOT PARSEC_ATOMIC_USE_C11_128 ) # try again with -mcx16
      include(CMakePushCheckState)
      CMAKE_PUSH_CHECK_STATE()
      SET( CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mcx16" )
      UNSET( PARSEC_ATOMIC_USE_C11_128 CACHE )
      CHECK_C_SOURCE_COMPILES("
            #include <stdatomic.h>
            int main(void) {
                __int128_t where = 0, expected;
                if( !atomic_compare_exchange_strong( (_Atomic __uint128_t*)&where, &expected, 1 ) )
                    return -1;
                return 0;
            }
            " PARSEC_ATOMIC_USE_C11_128)
      CMAKE_POP_CHECK_STATE()
      if( PARSEC_ATOMIC_USE_C11_128 )
	SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcx16" CACHE STRING "" FORCE)
      endif( PARSEC_ATOMIC_USE_C11_128 )
    endif( NOT PARSEC_ATOMIC_USE_C11_128 )
    if( NOT PARSEC_ATOMIC_USE_C11_128 ) # try again with -latomic
      include(CMakePushCheckState)
      CMAKE_PUSH_CHECK_STATE()
      list(APPEND CMAKE_REQUIRED_LIBRARIES atomic)
      UNSET( PARSEC_ATOMIC_USE_C11_128 CACHE )
      CHECK_C_SOURCE_COMPILES("
            #include <stdatomic.h>
            int main(void) {
                __int128_t where = 0, expected = 0;
                if( !atomic_compare_exchange_strong( (_Atomic __int128_t*)&where, &expected, 1 ) )
                    return -1;
                return 0;
            }
            " PARSEC_ATOMIC_USE_C11_128)
      CMAKE_POP_CHECK_STATE()
      if( PARSEC_ATOMIC_USE_C11_128 )
	list(APPEND EXTRA_LIBS "-latomic")
	list(APPEND CMAKE_C_STANDARD_LIBRARIES "-latomic")
      endif( PARSEC_ATOMIC_USE_C11_128 )
    endif( NOT PARSEC_ATOMIC_USE_C11_128 )
  endif(HAVE_UINT128b)
endif(PARSEC_ATOMIC_USE_C11_32)
endif( PARSEC_ATOMIC_USE_C11_ATOMICS )

#
# Check if the compiler supports __sync_bool_compare_and_swap.
#
if(NOT PARSEC_ATOMIC_USE_C11_32 OR NOT PARSEC_ATOMIC_USE_C11_64 OR NOT PARSEC_ATOMIC_USE_C11_128)

  include(CheckCSourceRuns)

# Gcc style atomics?
CHECK_C_SOURCE_COMPILES("
      #include <stdint.h>
      int main(void) {
         int32_t where = 0;
         if (!__sync_bool_compare_and_swap(&where, 0, 1))
            return -1;
         return 0;
      }
      " PARSEC_ATOMIC_USE_GCC_32_BUILTINS)
  # As far as I know, compilers that do not support C11 do not support the
  # libatomic extension either, so we do not recheck with -latomic as above
if( PARSEC_ATOMIC_USE_GCC_32_BUILTINS )
    CHECK_C_SOURCE_COMPILES("
        #include <stdint.h>
        int main(void) {
           int64_t where = 0;
           if (!__sync_bool_compare_and_swap(&where, 0, 1))
              return -1;
           return 0;
        }
        " PARSEC_ATOMIC_USE_GCC_64_BUILTINS)
endif( PARSEC_ATOMIC_USE_GCC_32_BUILTINS )
if( PARSEC_ATOMIC_USE_GCC_64_BUILTINS )
  CHECK_C_SOURCE_COMPILES("
        #include <stdint.h>
        int main(void) {
            __int128_t where = 0;
            if( !__sync_bool_compare_and_swap(&where, 0, 1))
                return -1;
            return 0;
        }
        " PARSEC_ATOMIC_USE_GCC_128_BUILTINS)
  if(HAVE_UINT128b)
    if( NOT PARSEC_ATOMIC_USE_GCC_128_BUILTINS ) # try again with -mcx16
      include(CMakePushCheckState)
      CMAKE_PUSH_CHECK_STATE()
      SET( CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mcx16" )
      UNSET( PARSEC_ATOMIC_USE_GCC_128_BUILTINS CACHE )
      CHECK_C_SOURCE_COMPILES("
            #include <stdint.h>
            int main(void) {
                __int128_t where = 0;
                if( !__sync_bool_compare_and_swap(&where, 0, 1))
                    return -1;
                return 0;
            }
            " PARSEC_ATOMIC_USE_GCC_128_BUILTINS)
      CMAKE_POP_CHECK_STATE()
      if( PARSEC_ATOMIC_USE_GCC_128_BUILTINS )
	SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcx16" CACHE STRING "" FORCE)
      endif( PARSEC_ATOMIC_USE_GCC_128_BUILTINS )
    endif( NOT PARSEC_ATOMIC_USE_GCC_128_BUILTINS )
  endif(HAVE_UINT128b)
endif( PARSEC_ATOMIC_USE_GCC_64_BUILTINS )
  
# Xlc style atomics?
CHECK_C_SOURCE_COMPILES("
      #include <stdint.h>

      int main(void)
      {
         int32_t where = 0, old = where;

         if (!__compare_and_swap(&where, &old, 1))
            return -1;

         return 0;
      }
      " PARSEC_ATOMIC_USE_XLC_32_BUILTINS)
if( PARSEC_ATOMIC_USE_XLC_32_BUILTINS )
  CHECK_C_SOURCE_COMPILES("
        #include <stdint.h>

        int main(void)
        {
           long where = 0, old = where;

           if (!__compare_and_swaplp(&where, &old, 1))
              return -1;

           return 0;
        }
        " PARSEC_ATOMIC_USE_XLC_64_BUILTINS)
endif( PARSEC_ATOMIC_USE_XLC_32_BUILTINS )

# MIPS style atomics?
CHECK_C_SOURCE_COMPILES("
      #include <stdint.h>

      int main(void)
      {
         uint32_t where  = 0;
         if (!__sync_compare_and_swap(&where, 0, 1))
            return -1;

         return 0;
      }
      " PARSEC_ATOMIC_USE_MIPOSPRO_32_BUILTINS)
if( PARSEC_ATOMIC_USE_MIPOSPRO_32_BUILTINS )
  CHECK_C_SOURCE_COMPILES("
        #include <stdint.h>

        int main(void)
        {
           uint64_t where  = 0;
           if (!__sync_compare_and_swap(&where, 0, 1))
              return -1;

           return 0;
        }
        " PARSEC_ATOMIC_USE_MIPOSPRO_64_BUILTINS)
endif( PARSEC_ATOMIC_USE_MIPOSPRO_32_BUILTINS )

# SUN OS style atomics?
CHECK_C_SOURCE_COMPILES("
      #include <atomic.h>
      #include <stdint.h>

      int main(void)
      {
         uint_t where = 0;
         if (0 != atomic_cas_uint(&where, 0, 1))
            return -1;

         return 0;
      }
      " PARSEC_ATOMIC_USE_SUN_32)
if( PARSEC_ATOMIC_USE_SUN_32 )
    CHECK_C_SOURCE_COMPILES("
        #include <atomic.h>
        #include <stdint.h>

        int main(void)
        {
           uint64_t where = 0;
           if (0 != atomic_cas_uint(&where, 0, 1))
              return -1;

           return 0;
        }
        " PARSEC_ATOMIC_USE_SUN_64)
endif( PARSEC_ATOMIC_USE_SUN_32 )

# Apple style atomics?
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  CHECK_FUNCTION_EXISTS(OSAtomicCompareAndSwap32 PARSEC_HAVE_COMPARE_AND_SWAP_32)
  CHECK_FUNCTION_EXISTS(OSAtomicCompareAndSwap64 PARSEC_HAVE_COMPARE_AND_SWAP_64)
endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")

endif(NOT PARSEC_ATOMIC_USE_C11_32 OR NOT PARSEC_ATOMIC_USE_C11_64 OR NOT PARSEC_ATOMIC_USE_C11_128)

if( PARSEC_ATOMIC_USE_SUN_32 OR PARSEC_ATOMIC_USE_MIPOSPRO_32_BUILTINS OR PARSEC_ATOMIC_USE_GCC_32_BUILTINS OR PARSEC_ATOMIC_USE_C11_32)
  set( PARSEC_HAVE_COMPARE_AND_SWAP_32 1 CACHE INTERNAL "Atomic operation on 32 bits are supported")
endif( PARSEC_ATOMIC_USE_SUN_32 OR PARSEC_ATOMIC_USE_MIPOSPRO_32_BUILTINS OR PARSEC_ATOMIC_USE_GCC_32_BUILTINS OR PARSEC_ATOMIC_USE_C11_32)

if( PARSEC_ATOMIC_USE_SUN_64 OR PARSEC_ATOMIC_USE_MIPOSPRO_64_BUILTINS OR PARSEC_ATOMIC_USE_GCC_64_BUILTINS OR PARSEC_ATOMIC_USE_C11_64)
  set( PARSEC_HAVE_COMPARE_AND_SWAP_64 1 CACHE INTERNAL "Atomic operation on 64 bits are supported")
endif( PARSEC_ATOMIC_USE_SUN_64 OR PARSEC_ATOMIC_USE_MIPOSPRO_64_BUILTINS OR PARSEC_ATOMIC_USE_GCC_64_BUILTINS OR PARSEC_ATOMIC_USE_C11_64)

if( PARSEC_ATOMIC_USE_GCC_128_BUILTINS OR PARSEC_ATOMIC_USE_C11_128)
  set( PARSEC_HAVE_COMPARE_AND_SWAP_128 1 CACHE INTERNAL "Atomic operation on 128 bits are supported")
endif( PARSEC_ATOMIC_USE_GCC_128_BUILTINS OR PARSEC_ATOMIC_USE_C11_128)


if( PARSEC_HAVE_COMPARE_AND_SWAP_32 )
  message( STATUS "\t support for 32 bits atomics - found")
endif( PARSEC_HAVE_COMPARE_AND_SWAP_32 )

if( PARSEC_HAVE_COMPARE_AND_SWAP_64 )
  message( STATUS "\t support for 64 bits atomics - found")
endif( PARSEC_HAVE_COMPARE_AND_SWAP_64 )

if( PARSEC_HAVE_COMPARE_AND_SWAP_128 )
  message( STATUS "\t support for 128 bits atomics - found")
endif( PARSEC_HAVE_COMPARE_AND_SWAP_128 )

if( CMAKE_SIZEOF_VOID_P MATCHES "8" )
  if( PARSEC_HAVE_COMPARE_AND_SWAP_32 AND NOT PARSEC_HAVE_COMPARE_AND_SWAP_64 )
    message( FATAL_ERROR "64 bits OS with support for 32 bits atomics but without support for 64 bits atomics")
  endif( PARSEC_HAVE_COMPARE_AND_SWAP_32 AND NOT PARSEC_HAVE_COMPARE_AND_SWAP_64 )
  if( NOT PARSEC_HAVE_COMPARE_AND_SWAP_128 )
    message( WARNING "128 bit atomics not found but pointers are 64 bits. Some list operations will not be optimized")
  endif( NOT PARSEC_HAVE_COMPARE_AND_SWAP_128 )
endif( CMAKE_SIZEOF_VOID_P MATCHES "8" )


