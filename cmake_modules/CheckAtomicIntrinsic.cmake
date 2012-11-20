#
# Check if the compiler supports __sync_bool_compare_and_swap.
#
if(NOT HAVE_COMPARE_AND_SWAP_32 AND NOT HAVE_COMPARE_AND_SWAP_64)
    

include(CheckCSourceCompiles)

# Gcc style atomics?
CHECK_C_SOURCE_COMPILES("
      #include <stdint.h>

      int main( int argc, char** argv)
      {
         int32_t where = 0;

         if (!__sync_bool_compare_and_swap(&where, 0, 1))
            return -1;
         
         return 0;
      }
      " DAGUE_ATOMIC_USE_GCC_32_BUILTINS)
if( DAGUE_ATOMIC_USE_GCC_32_BUILTINS )
  CHECK_C_SOURCE_COMPILES("
        #include <stdint.h>

        int main( int argc, char** argv)
        {
           int64_t where = 0;

           if (!__sync_bool_compare_and_swap(&where, 0, 1))
              return -1;

           return 0;
        }
        " DAGUE_ATOMIC_USE_GCC_64_BUILTINS)
endif( DAGUE_ATOMIC_USE_GCC_32_BUILTINS )

# Xlc style atomics?
CHECK_C_SOURCE_COMPILES("
      #include <stdint.h>

      int main( int argc, char** argv)
      {
         int32_t where = 0, old = where;

         if (!__compare_and_swap(&where, &old, 1))
            return -1;

         return 0;
      }
      " DAGUE_ATOMIC_USE_XLC_32_BUILTINS)
if( DAGUE_ATOMIC_USE_XLC_32_BUILTINS )
  CHECK_C_SOURCE_COMPILES("
        #include <stdint.h>

        int main( int argc, char** argv)
        {
           long where = 0, old = where;

           if (!__compare_and_swaplp(&where, &old, 1))
              return -1;

           return 0;
        }
        " DAGUE_ATOMIC_USE_XLC_64_BUILTINS)
endif( DAGUE_ATOMIC_USE_XLC_32_BUILTINS )

# MIPS style atomics?
CHECK_C_SOURCE_COMPILES("
      #include <stdint.h>

      int main(int, const char**)
      {
         uint32_t where  = 0;
         if (!__sync_compare_and_swap(&where, 0, 1))
            return -1;

         return 0;
      }
      " DAGUE_ATOMIC_USE_MIPOSPRO_32_BUILTINS)
if( DAGUE_ATOMIC_USE_MIPOSPRO_32_BUILTINS )
  CHECK_C_SOURCE_COMPILES("
        #include <stdint.h>

        int main(int, const char**)
        {
           uint64_t where  = 0;
           if (!__sync_compare_and_swap(&where, 0, 1))
              return -1;

           return 0;
        }
        " DAGUE_ATOMIC_USE_MIPOSPRO_64_BUILTINS)
endif( DAGUE_ATOMIC_USE_MIPOSPRO_32_BUILTINS )

# SUN OS style atomics? 
CHECK_C_SOURCE_COMPILES("
      #include <atomic.h>
      #include <stdint.h>

      int main(int, const char**)
      {
         uint_t where = 0;
         if (0 != atomic_cas_uint(&where, 0, 1))
            return -1;

         return 0;
      }
      " DAGUE_ATOMIC_USE_SUN_32)
if( DAGUE_ATOMIC_USE_SUN_32 )
  CHECK_C_SOURCE_COMPILES("
        #include <atomic.h>
        #include <stdint.h>

        int main(int, const char**)
        {
           uint64_t where = 0;
           if (0 != atomic_cas_uint(&where, 0, 1))
              return -1;

           return 0;
        }
        " DAGUE_ATOMIC_USE_SUN_64)
endif( DAGUE_ATOMIC_USE_SUN_32 )

# Apple style atomics?
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  CHECK_FUNCTION_EXISTS(OSAtomicCompareAndSwap32 HAVE_COMPARE_AND_SWAP_32)
  CHECK_FUNCTION_EXISTS(OSAtomicCompareAndSwap64 HAVE_COMPARE_AND_SWAP_64)
endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")

if( DAGUE_ATOMIC_USE_SUN_32 OR DAGUE_ATOMIC_USE_MIPOSPRO_32_BUILTINS OR DAGUE_ATOMIC_USE_GCC_32_BUILTINS )
  set( HAVE_COMPARE_AND_SWAP_32 1 CACHE INTERNAL "Atomic operation on 32 bits are supported")
endif( DAGUE_ATOMIC_USE_SUN_32 OR DAGUE_ATOMIC_USE_MIPOSPRO_32_BUILTINS OR DAGUE_ATOMIC_USE_GCC_32_BUILTINS )

if( DAGUE_ATOMIC_USE_SUN_64 OR DAGUE_ATOMIC_USE_MIPOSPRO_64_BUILTINS OR DAGUE_ATOMIC_USE_GCC_64_BUILTINS )
  set( HAVE_COMPARE_AND_SWAP_64 1 CACHE INTERNAL "Atomic operation on 64 bits are supported")
endif( DAGUE_ATOMIC_USE_SUN_64 OR DAGUE_ATOMIC_USE_MIPOSPRO_64_BUILTINS OR DAGUE_ATOMIC_USE_GCC_64_BUILTINS )

if( HAVE_COMPARE_AND_SWAP_32 )
  message( STATUS "\t support for 32 bits atomics - found")
endif( HAVE_COMPARE_AND_SWAP_32 )

if( HAVE_COMPARE_AND_SWAP_64 )
  message( STATUS "\t support for 64 bits atomics - found")
endif( HAVE_COMPARE_AND_SWAP_64 )

if( CMAKE_SIZEOF_VOID_P MATCHES "8" AND HAVE_COMPARE_AND_SWAP_32 AND NOT HAVE_COMPARE_AND_SWAP_64 )
  error( "64 bits OS with support for 32 bits atomics but without support for 64 bits atomics")
endif( CMAKE_SIZEOF_VOID_P MATCHES "8" AND HAVE_COMPARE_AND_SWAP_32 AND NOT HAVE_COMPARE_AND_SWAP_64 )

endif(NOT HAVE_COMPARE_AND_SWAP_32 AND NOT HAVE_COMPARE_AND_SWAP_64)

