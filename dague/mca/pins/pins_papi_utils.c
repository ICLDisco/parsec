#include "dague_config.h"
#if defined(HAVE_PAPI)
#include <papi.h>
#endif
#include "pins_papi_utils.h"
#include "dague/utils/output.h"

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

int pins_papi_mca_string_parse(dague_execution_unit_t * exec_unit, char* mca_param_string, char*** event_names)
{
    int num_counters = 0;
    #if defined(HAVE_PAPI)
    char *mca_param_name, *token, *saveptr = NULL;
    int socket, core;

    num_counters = 0;

    mca_param_name = strdup(mca_param_string);
    token = strtok_r(mca_param_name, ":", &saveptr);

    while(token != NULL) {
        socket = core = 0;

        if(token[0] == 'S') {
            if(token[1] != '*') {
                if(atoi(&token[1]) == exec_unit->socket_id)
                    socket = 1;
            } else
                socket = 1;
        }

        token = strtok_r(NULL, ":", &saveptr);

        if(token[0] == 'C') {
            if(token[1] != '*') {
                if(atoi(&token[1]) == (exec_unit->core_id % CORES_PER_SOCKET))
                    core = 1;
            } else
                core = 1;
        }

        token = strtok_r(NULL, ",", &saveptr);

        if(socket == 1 && core == 1) {
            if(num_counters == 0) {
                *event_names = (char**)malloc(sizeof(char*));
                event_names[0][0] = strdup(token);
            } else {
                event_names[0] = (char**)realloc(event_names[0], (num_counters+1) * sizeof(char*));
                event_names[0][num_counters] = strdup(token);
            }
            num_counters++;
        }
        token = strtok_r(NULL, ":", &saveptr);
    }

    free(mca_param_name);
#endif /* HAVE_PAPI */
    return num_counters;
}

int pins_papi_create_eventset(dague_execution_unit_t * exec_unit, int* eventset, char** event_names, int** native_events, int num_events)
{
    #if defined(HAVE_PAPI)
    int err;
    *eventset = PAPI_NULL;
    *native_events = (int*)malloc(num_events * sizeof(int));

    /* Create an empty eventset */
    if( PAPI_OK != (err = PAPI_create_eventset(eventset)) ) {
        dague_output(0, "pins_thread_init_papi_socket: thread %d couldn't create the PAPI event set; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        return -1;
    }

    int i;
    for(i = 0; i < num_events; i++) {
        native_events[0][i] = PAPI_NULL;
        /* Convert event name to code */
        if(PAPI_OK != PAPI_event_name_to_code(event_names[i], &native_events[0][i]) )
            break;

        /* Add event to the eventset */
        if( PAPI_OK != (err = PAPI_add_event(*eventset, native_events[0][i])) ) {
            dague_output(0, "pins_thread_init_papi_socket: failed to add event %s; ERROR: %s\n",
                         event_names[i], PAPI_strerror(err));
            break;
        }
    }
#endif /* HAVE_PAPI */
    return num_events;
}
