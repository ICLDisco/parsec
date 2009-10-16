#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "params.h"

void param_dump(const param_t *p, const char *prefix)
{
    int i;
    char *pref2 = (char*)malloc(strlen(prefix)+8);
    pref2[0] = '\0';

    printf("%s%s %s%s ", 
           prefix, p->sym_name, 
           (p->sym_type & SYM_IN)  ? "IN"  : "  ",
           (p->sym_type & SYM_OUT) ? "OUT" : "   ");

    for(i = 0; NULL != p->dep_in[i] && i < MAX_DEP_IN_COUNT; i++) {
        dep_dump( p->dep_in[i], pref2 );
        sprintf(pref2, "%s       ", prefix);
    }
    for(i = 0; NULL != p->dep_out[i] && i < MAX_DEP_OUT_COUNT; i++) {
        dep_dump( p->dep_out[i], pref2 );
        sprintf(pref2, "%s       ", prefix);
    }
}
