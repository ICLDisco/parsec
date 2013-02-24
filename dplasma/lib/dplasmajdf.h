#ifndef _DPLASMAJDF_H_
#define _DPLASMAJDF_H_

#ifdef DAGUE_CALL_TRACE
#   include <stdlib.h>
#   include <stdio.h>
#   define plasma_const( x )  plasma_lapack_constants[x]
#   define printlog(str, ...) fprintf(stderr, "thread %d VP %d " str "\n", \
                                      context->th_id, context->virtual_process->vp_id, __VA_ARGS__)
#else
#   define printlog(...) do {} while(0)
#endif

#endif /* _DPLASMAJDF_H_ */

