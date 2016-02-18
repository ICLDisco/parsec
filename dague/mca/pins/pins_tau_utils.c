#include "dague_config.h"
#if defined(DAGUE_HAVE_TAU)
#include "TAU.h"
#endif
#include "pins_tau_utils.h"

#if defined(DAGUE_HAVE_TAU)
static int init_done = 0;
static int thread_init_done = 0;
#endif  /* defined(DAGUE_HAVE_TAU) */

void pins_tau_init(dague_context_t * master_context)
{
    (void)master_context;
#if defined(DAGUE_HAVE_TAU)
    if (!init_done) {
        init_done = 1;
        TAU_INIT(pargc, pargv);
        TAU_DB_PURGE();
        TAU_PROFILE_SET_NODE(0);
    }
#endif /* DAGUE_HAVE_TAU */
}


void pins_tau_thread_init(dague_execution_unit_t * exec_unit)
{
    (void)exec_unit;
#if defined(DAGUE_HAVE_TAU)
    if (!thread_init_done) {
        thread_init_done = 1;
        TAU_REGISTER_THREAD();
    }
#endif /* DAGUE_HAVE_TAU */
}
