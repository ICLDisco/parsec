#ifndef _DPLASMAJDF_H_
#define _DPLASMAJDF_H_

#ifdef DAGUE_CALL_TRACE
#   include <stdlib.h>
#   include <stdio.h>
#   define printlog(...) fprintf(stderr, __VA_ARGS__)
#   define OUTPUT(ARG)  printf ARG
#else
#   define printlog(...) do {} while(0)
#   define OUTPUT(ARG)
#endif

#if (defined DAGCOMPLEX) 
#if (defined DAGDOUBLE)
#define TYPENAME   PLASMA_Complex64_t
#define TYPELETTER z
#define PRECNAME   double
#define PRECLETTER d
#define CORE(FN, ARGS) CORE_z##FN ARGS
#define dagueprefix(fn) dague_z##fn
#define DAGUEprefix(fn) DAGUE_z##fn
#else 
#define TYPENAME   PLASMA_Complex32_t
#define TYPELETTER c
#define PRECNAME   float
#define PRECLETTER s
#define CORE(FN, ARGS) CORE_c##FN ARGS
#define dagueprefix(fn) dague_c##fn
#define DAGUEprefix(fn) DAGUE_c##fn
#endif
#else
#if (defined DAGDOUBLE)
#define TYPENAME   double
#define TYPELETTER d
#define PRECNAME   double
#define PRECLETTER d
#define CORE(FN, ARGS) CORE_d##FN ARGS
#define dagueprefix(fn) dague_d##fn
#define DAGUEprefix(fn) DAGUE_d##fn
#else 
#define TYPENAME   float
#define TYPELETTER s
#define PRECNAME   float
#define PRECLETTER s
#define CORE(FN, ARGS) CORE_s##FN ARGS
#define dagueprefix(fn) dague_s##fn
#define DAGUEprefix(fn) DAGUE_s##fn
#endif
#endif

#ifdef DAGUE_DRY_RUN
#undef CORE
#define CORE(FN, ARGS)
#endif
    
#endif /* _DPLASMAJDF_H_ */
