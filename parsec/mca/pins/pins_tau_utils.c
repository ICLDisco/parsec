#include "parsec_config.h"
#if defined(PARSEC_HAVE_TAU)
#include "TAU.h"
#endif
#include "pins_tau_utils.h"

#if defined(PARSEC_HAVE_TAU)
static int init_done = 0;
static int thread_init_done = 0;
#endif  /* defined(PARSEC_HAVE_TAU) */

void pins_tau_init(parsec_context_t * master_context)
{
    (void)master_context;
#if defined(PARSEC_HAVE_TAU)
    if (!init_done) {
        init_done = 1;
        TAU_INIT(pargc, pargv);
        TAU_DB_PURGE();
        TAU_PROFILE_SET_NODE(0);
    }
#endif /* PARSEC_HAVE_TAU */
}


void pins_tau_thread_init(parsec_execution_unit_t * exec_unit)
{
    (void)exec_unit;
#if defined(PARSEC_HAVE_TAU)
    if (!thread_init_done) {
        thread_init_done = 1;
        TAU_REGISTER_THREAD();
    }
#endif /* PARSEC_HAVE_TAU */
}
