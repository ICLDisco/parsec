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

#define MAX_DEP_IN_COUNT  5
#define MAX_DEP_OUT_COUNT 5

struct param {
    char          *sym_name;
    unsigned char  sym_type;
    dep_t         *dep_in[MAX_DEP_IN_COUNT];
    dep_t         *dep_out[MAX_DEP_OUT_COUNT];
};

void param_dump(const param_t *p, const char *prefix);

/**
 * helper to dump the c structure representing the dplasma object
 * 
 * can add anything to init_func_body that will be run in the constructor at init time
 *
 * @returns a (static) string representing the (unique) name of 
 *          the params to use to point to this param.
 *          the special value "NULL" if p is null
 */
char *dump_c_param(FILE *out, const param_t *p,  char *init_func_body, int init_func_body_size);

#endif
