/**
 * This is a simple example on how to use the parsec profiling system standalone
 *
 * Compilation:
 *  Include dirs must point to a directory with profiling.h (in dague/ directory)
 *    (NB profiling.h is standalone, you can also copy it where you need it)
 *  The program must be linked with libdague.a and libdague-base.a
 *  The libraries must be compiled with DAGUE_PROF_TRACE enabled
 *
 * Execution:
 *  Just execute the binary without arguments. This produces a sp-0.prof-* file
 *  that holds the trace example
 *
 * Explanation of the code:
 *  This code uses NB_THREADS that each produce EVENTS_PER_THREAD event pairs
 *  Each event pair is either of type A or B
 *  Events of type A have no information structure
 *  Events of type B have an information structure with an int (i), and a double (d)
 *  Events are correctly paired (no overlapping of events inside the same thread), but
 *    this is not required by the profiling system if events are correctly identified.
 */

#include <pthread.h>
#include <unistd.h>
#include "profiling.h"

#define NB_THREADS         4
#define EVENTS_PER_THREAD 10

typedef struct {
    pthread_t                 pthread_id;
    int                       thread_index;
    dague_thread_profiling_t *prof;
} per_thread_info_t;

typedef struct {
    int i;
    double d;
} event_b_info_t;

#define EVENT_B_INFO_CONVERTER "i{int};d{double}"

/**
 * Declare a pair of int for each event type: a start key and an end key
 */
static int event_a_startkey, event_a_endkey;
static int event_b_startkey, event_b_endkey;

static void *run_thread(void *_arg)
{
    per_thread_info_t *ti = (per_thread_info_t*)_arg;
    int i;

    /** This code is thread-specific
     *  Each thread must use its own dague_thread_profiling_t * head to trace events
     *  If two threads share the same dague_thread_profiling_t * or two threads use the
     *  same dague_thread_profiling_t * to trace events, the trace will be corrupted
     *
     *  4096 is the size of the events page; memory allocation and potentially I/O
     *       flush may happen every time events fill up these pages.
     *       Multiple of the page size is mandatory
     *       Take a value that can fit all your events if you can
     *       Otherwise, take the largest value that does not hinder your application.
     *  format, printf arguments: to build a unique human-readable name for the thread
     */
    ti->prof = dague_profiling_thread_init(4096, "This is the name of thread %d", ti->thread_index);

    /**
     *  You can save runtime-specific information per threads, in the form of key/value pair
     */
    dague_profiling_thread_add_information(ti->prof,
                                           "This is a thread-specific information key",
                                           "This is the corresponding value");

    for(i = 0; i < EVENTS_PER_THREAD; i++) {
        if( rand() % 2 == 0 ) {
            /**
             * This is how to trace an event without additional information.
             *  the dague_thread_profiling_t * must be the one of *this* thread
             *  startkey / endkey, depending if this is starting a state or ending one.
             *   NB: a state that is started must be ended; a state that is ended must have been started before
             *  i is an identifier of the event.
             *   If it is not possible (or not efficient) to create a unique identifier, events are matched
             *   as follows: the end of a start event matches the start event if it has the same object_id,
             *   the same event_id, and happens after the start. Matching algorithm searches first on the current
             *   thread, then on all other threads, in an arbitrary order.
             *  PROFILE_OBJECT_ID_NULL is passed to say if the object_id should be ignored when matching
             *   this event. Here, i is sufficiently unique since the end always happens on the same thread as
             *   the start.
             */
            dague_profiling_trace_flags(ti->prof, event_a_startkey, i, PROFILE_OBJECT_ID_NULL, NULL, 0);
            usleep(rand() % 300);
            dague_profiling_trace_flags(ti->prof, event_a_endkey, i, PROFILE_OBJECT_ID_NULL, NULL, 0);
        } else {
            event_b_info_t info;
            info.i = i;
            info.d = (double)ti->thread_index;
            dague_profiling_trace_flags(ti->prof, event_b_startkey, i, PROFILE_OBJECT_ID_NULL, NULL, 0);
            usleep(rand() % 300);
            /** This is how to trace an event with additional information
             *  The information can be attached to the start or the end or both parts of a state
             *  One passes the pointer to a structure with the additional fields;
             *  That structure must be at least of size the size passed as argument to the
             *  dictionary creation below.
             *  The event is marked with having additional information.
             */
            dague_profiling_trace_flags(ti->prof, event_b_endkey, i, PROFILE_OBJECT_ID_NULL, &info,
                                        DAGUE_PROFILING_EVENT_HAS_INFO);
        }
    }

    return NULL;
}

int main()
{
    int i;
    per_thread_info_t thread_info[NB_THREADS];

    /** First, there is a sequential part (no threads) */

    /** We initialize the system */
    dague_profiling_init();

    /** MPI should be initialized before the dbp_start call, if it is a distributed application
     *  first argument sp is the base name for the trace file
     *   It will be named sp-<%d>.prof-XXXX where <%d> is the MPI rank (0 if no MPI), and XXXXX is a random value
     *  second argument "Demonstration..." is a human readable string to qualify the trace
     */
    dague_profiling_dbp_start( "sp", "Demonstration of basic PaRSEC profiling system" );

    /** Each Event type must be defined before any event is traced
     *  They are defined by being added to a dictionary.
     *  In case of distributed run, the dictionaries must match exactly
     *
     *  First parameter Event A is the human readable name
     *  Then there is an HTML suggested color for tracing that state
     *  The size (number of bytes) of additional informations related to this event when there is one
     *  A converter string. In this case, the structure has an int, i and a double, d, so the string
     *   is "i{int};d{double}"
     *  The call returns the startkey and the endkey corresponding to the new event
     */
    dague_profiling_add_dictionary_keyword("Event A", "#FF0000", 0, NULL, &event_a_startkey, &event_a_endkey);
    dague_profiling_add_dictionary_keyword("Event B", "#0000FF",
                                           sizeof(event_b_info_t), EVENT_B_INFO_CONVERTER,
                                           &event_b_startkey, &event_b_endkey);

    /**
     * Process-level key/value pairs can be added to remember parameters of the run for example
     */
    dague_profiling_add_information("This is a global information key", "This is the global information value");

    /** profiling_start() defines the time 0. It must be called, or no event will be traced */
    dague_profiling_start();

    /** After this step, multithreaded execution can start */

    for(i = 1; i < NB_THREADS; i++) {
        thread_info[i].thread_index = i;
        pthread_create(&thread_info[i].pthread_id, NULL, run_thread, &thread_info[i]);
    }

    thread_info[0].thread_index = 0;
    run_thread(&thread_info[0]);

    for(i = 1; i < NB_THREADS; i++)
        pthread_join(thread_info[i].pthread_id, NULL);

    /** dbp_dump() will flush the trace file. fini() will also flush if it was not done before
     *  dbp_dump() and fini() are not thread safe and no thread should use any profiling routine
     *  while those are called, or after fini() is called
     */
    dague_profiling_dbp_dump();
    dague_profiling_fini();
}
