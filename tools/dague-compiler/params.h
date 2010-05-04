/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _params_h
#define _params_h

typedef struct param param_t;

#include "dep.h"

/**< Remark: (sym_type == SYM_INOUT) if (sym_type & SYM_IN) && (sym_type & SYM_OUT) */
#define SYM_IN     0x01
#define SYM_OUT    0x02
#define SYM_INOUT  (SYM_IN | SYM_OUT)

#define ACCESS_READ     0x01
#define ACCESS_WRITE    0x02
#define ACCESS_RW       (ACCESS_READ | ACCESS_WRITE)

#define MAX_DEP_IN_COUNT  10
#define MAX_DEP_OUT_COUNT 10

struct param {
    char*               name;
    unsigned char       sym_type;
    unsigned char       access_type;
    unsigned char       param_mask;
    const dep_t*        dep_in[MAX_DEP_IN_COUNT];
    const dep_t*        dep_out[MAX_DEP_OUT_COUNT];
};

void param_dump(const param_t *p, const char *prefix);

/**
 *
 */
param_t* DAGuE_find_or_create_param(struct DAGuE_t* function, char* param_name);

#endif
