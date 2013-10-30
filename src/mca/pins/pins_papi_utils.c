#include "dague_config.h"
#if defined(HAVE_PAPI)
#include <papi.h>
#endif
#include "pins_papi_utils.h"

#if defined(HAVE_PAPI)
static int init_done = 0;
static int thread_init_done = 0;
#endif  /* defined(HAVE_PAPI) */

void pins_papi_init(dague_context_t * master_context)
{
    (void)master_context;
#if defined(HAVE_PAPI)
    if (!init_done) {
        init_done = 1;
        PAPI_library_init(PAPI_VER_CURRENT); /* this has to happen before threads get created */
        PAPI_set_debug(PAPI_VERB_ECONT);
        int t_init = PAPI_thread_init(( unsigned long ( * )( void ) ) ( pthread_self )); 
        if (t_init != PAPI_OK)
            DEBUG(("PAPI Thread Init failed with error code %d (%s)!\n", t_init, PAPI_strerror(t_init)));
    }
#endif /* HAVE_PAPI */
}


void pins_papi_thread_init(dague_execution_unit_t * exec_unit)
{
    (void)exec_unit;
#if defined(HAVE_PAPI)
    if (!thread_init_done) {
        thread_init_done = 1;
        int rv = PAPI_register_thread();
        if (rv != PAPI_OK)
            DEBUG(("PAPI_register_thread failed with error %s\n", PAPI_strerror(rv)));
    }
#endif /* HAVE_PAPI */
}
