#ifndef _DPLASMAJDF_H_
#define _DPLASMAJDF_H_

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

#ifdef DAGUE_CALL_TRACE
#   include <stdlib.h>
#   include <stdio.h>
#   define printlog(...) fprintf(stderr, __VA_ARGS__)
#   define OUTPUT(ARG)  printf ARG
#else
#   define printlog(...) do {} while(0)
#   define OUTPUT(ARG)
#endif

#ifdef DAGUE_DRY_RUN
#undef CORE
#define CORE(FN, ARGS)
#define DRYRUN( body )
#else
#define DRYRUN( body ) body
#endif


/** PRECISION GENERATION, DEPRECATED: TODO: REMOVE */
#if   defined(PRECISION_z) 
#define TYPENAME   PLASMA_Complex64_t
#define TYPELETTER z
#define PRECNAME   double
#define PRECLETTER d
#define CORE(FN, ARGS) CORE_z##FN ARGS
#define dagueprefix(fn) dague_z##fn
#define DAGUEprefix(fn) DAGUE_z##fn
#define MPITYPE  MPI_DOUBLE_COMPLEX
#define DAGUE_TYPE_ENUM matrix_ComplexDouble

#elif defined(PRECISION_c)
#define TYPENAME   PLASMA_Complex32_t
#define TYPELETTER c
#define PRECNAME   float
#define PRECLETTER s
#define CORE(FN, ARGS) CORE_c##FN ARGS
#define dagueprefix(fn) dague_c##fn
#define DAGUEprefix(fn) DAGUE_c##fn
#define MPITYPE  MPI_COMPLEX
#define DAGUE_TYPE_ENUM matrix_ComplexFloat

#elif defined(PRECISION_d)
#define TYPENAME   double
#define TYPELETTER d
#define PRECNAME   double
#define PRECLETTER d
#define CORE(FN, ARGS) CORE_d##FN ARGS
#define dagueprefix(fn) dague_d##fn
#define DAGUEprefix(fn) DAGUE_d##fn
#define MPITYPE  MPI_DOUBLE
#define DAGUE_TYPE_ENUM matrix_RealDouble

#elif defined(PRECISION_s) 
#define TYPENAME   float
#define TYPELETTER s
#define PRECNAME   float
#define PRECLETTER s
#define CORE(FN, ARGS) CORE_s##FN ARGS
#define dagueprefix(fn) dague_s##fn
#define DAGUEprefix(fn) DAGUE_s##fn
#define MPITYPE  MPI_FLOAT
#define DAGUE_TYPE_ENUM matrix_RealFloat
#else
/*#error "Precision is not selected. You have to define PRECISION_[zcdf]"*/
#endif


#ifndef USE_MPI
#define TEMP_TYPE MPITYPE
#undef MPITYPE
#define MPITYPE ((dague_remote_dep_datatype_t)QUOTEME(TEMP_TYPE))
#undef TEMP_TYPE
#endif  /* USE_MPI */


#endif /* _DPLASMAJDF_H_ */

