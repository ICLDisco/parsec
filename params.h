/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _params_h
#define _params_h

typedef struct param param_t;

#include "dep.h"
#include "remote_dep.h"

/**< Remark: (sym_type == SYM_INOUT) if (sym_type & SYM_IN) && (sym_type & SYM_OUT) */
#define SYM_IN     0x01
#define SYM_OUT    0x02
#define SYM_INOUT  (SYM_IN | SYM_OUT)

#define MAX_DEP_IN_COUNT  10
#define MAX_DEP_OUT_COUNT 10

struct param {
    char*                           name;
    struct dplasma_t*               function;
    unsigned char                   sym_type;
    unsigned char                   param_mask;
    dep_t*                          dep_in[MAX_DEP_IN_COUNT];
    dep_t*                          dep_out[MAX_DEP_OUT_COUNT];
    dplasma_remote_dep_datatype_t*  type;
};

void param_dump(const param_t *p, const char *prefix);

/**
 *
 */
param_t* dplasma_find_or_create_param(struct dplasma_t* function, char* param_name);

#endif
