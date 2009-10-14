#ifndef _params_h
#define _params_h

#include "dep.h"

/**< Remark: (sym_type == SYM_INOUT) iff (sym_type & SYM_IN) && (sym_type & SYM_OUT) */
#define SYM_IN    1
#define SYM_OUT   2
#define SYM_INOUT 3

#define MAX_DEP_IN_COUNT  3
#define MAX_DEP_OUT_COUNT 3

typedef struct {
    char          *sym_name;
    unsigned char  sym_type;
    dep_t         *dep_in[MAX_DEP_IN_COUNT];
    dep_t         *dep_out[MAX_DEP_OUT_COUNT];
} param_t;

#endif
