#include <stdio.h>
#include <string.h>
#include <stdlib.h>
 
#include "dplasma.h"

void dplasma_dump(const dplasma_t *d, const char *prefix)
{
    int i;
    char *pref2 = malloc(strlen(prefix)+3);

    sprintf(pref2, "%s  ", prefix);
    printf("%sDplasma Function: %s\n", prefix, d->name);

    printf("%s Parameter Variables:\n", prefix);
    for(i = 0; NULL != d->locals[i] && i < MAX_LOCAL_COUNT; i++) {
        symbol_dump(d->locals[i], pref2);
    }

    free(pref2);
}
