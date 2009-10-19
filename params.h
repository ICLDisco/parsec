/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _params_h
#define _params_h

typedef struct param param_t;

#include "dep.h"

/**< Remark: (sym_type == SYM_INOUT) iff (sym_type & SYM_IN) && (sym_type & SYM_OUT) */
#define SYM_IN    1
#define SYM_OUT   2
#define SYM_INOUT 3

#define MAX_DEP_IN_COUNT  5
#define MAX_DEP_OUT_COUNT 5

struct param {
    char          *sym_name;
    unsigned char  sym_type;
    dep_t         *dep_in[MAX_DEP_IN_COUNT];
    dep_t         *dep_out[MAX_DEP_OUT_COUNT];
};

void param_dump(const param_t *p, const char *prefix);

#endif
