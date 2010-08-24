/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague.h"
#include "remote_dep.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

void param_dump(const param_t *p, const struct dague_object *dague_object, const char *prefix)
{
    int i, length;
    char *pref2 = (char*)malloc(strlen(prefix)+8);

    length = printf("%s%s %s%s ", 
                    prefix, (p->name == NULL) ? "" : p->name, 
                    (p->sym_type & SYM_IN)  ? "IN"  : "  ",
                    (p->sym_type & SYM_OUT) ? "OUT" : "   ");

    if( NULL != p->dep_in[0] ) {
        dep_dump(p->dep_in[0], dague_object, "<- " );
        for( i = sprintf( pref2, "%s", prefix ); i < length; pref2[i] = ' ', i++ );
        sprintf( pref2 + length, "<- " );
    }
    for(i = 1; NULL != p->dep_in[i] && i < MAX_DEP_IN_COUNT; i++) {
        dep_dump( p->dep_in[i], dague_object, pref2 );
    }

    sprintf( pref2 + length, "-> " );
    for(i = 0; i < MAX_DEP_OUT_COUNT && NULL != p->dep_out[i]; i++) {
        dep_dump( p->dep_out[i], dague_object, pref2 );
    }
}
