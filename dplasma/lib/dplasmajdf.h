#ifndef _DPLASMAJDF_H_
#define _DPLASMAJDF_H_

#include <core_blas.h>
#include "dague.h"
#include "dplasma.h"
#include "memory_pool.h"

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

#ifdef DAGUE_CALL_TRACE
#   include <stdlib.h>
#   include <stdio.h>
#   define plasma_const( x )  plasma_lapack_constants[x]
#   define printlog(str, ...) fprintf(stderr, "thread %d VP %d " str "\n", \
                                      context->th_id, context->virtual_process->vp_id, __VA_ARGS__)
#   define OUTPUT(ARG)  printf ARG
#else
#   define printlog(...) do {} while(0)
#   define OUTPUT(ARG)
#endif

#ifdef DAGUE_DRY_RUN
#define DRYRUN( body )
#else
#define DRYRUN( body ) body
#endif

#ifndef HAVE_MPI
#define TEMP_TYPE MPITYPE
#undef MPITYPE
#define MPITYPE ((dague_datatype_t)QUOTEME(TEMP_TYPE))
#undef TEMP_TYPE
#endif  /* HAVE_MPI */


#endif /* _DPLASMAJDF_H_ */

