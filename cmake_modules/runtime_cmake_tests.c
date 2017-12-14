#include <stdio.h>
#include <stdlib.h>

#include "parsec/parsec_config.h"
#include "parsec/sys/atomic.h"

int main()
{
#if defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B)
   printf("PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B=TRUE\n");
   return EXIT_SUCCESS;
#endif
   printf("PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B=FALSE\n");
   return EXIT_FAILURE;
}
