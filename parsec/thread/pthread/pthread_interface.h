/**
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "pthread.h"

struct  parsec_thread_s {
    int rank;
    pthread_t thread;
};

typedef struct parsec_thread_s parsec_thread_t;

#define PARSEC_THREAD_LIBRARY_INIT(pargc, pargv) { \
    }

#define PARSEC_THREAD_LIBRARY_FINI() { \
    }

#define PARSEC_THREAD_SET_CONCURRENCY( nb_threads ) do {      \
        pthread_setconcurrency(nb_threads);               \
    } while(0)

#define PARSEC_THREAD_CONTEXT_MALLOC2( context, nb_threads ) do {        \
        context->parsec_threads = (parsec_thread_t*)malloc(nb_threads * sizeof(parsec_thread_t)); \
        context->monitoring_steering_threads = (parsec_thread_t*)malloc(3 * sizeof(parsec_thread_t)); \
    } while(0)



/*pthread_self(void);*/
/*pthread_cond_wait to prevent the thread from being scheduled*/


#define PARSEC_THREAD_CONTEXT_MALLOC( context, nb_threads ) do {         \
        context->parsec_threads = NULL;                                  \
        if ( nb_threads > 1 )                                           \
            context->parsec_threads = (parsec_thread_t*)malloc(nb_threads * sizeof(parsec_thread_t)); \
    } while(0)

#define PARSEC_THREAD_STREAM_CREATE( parsec_thread, t ) {                 \
    }

#define PARSEC_THREAD_STREAM_INIT( parsec_thread, sched_init_func ) {     \
	}

#define PARSEC_THREAD_COMM_STREAM_INIT( parsec_context, parsec_thread ) {  \
    }

#define PARSEC_THREAD_STREAM_MASTER_CREATE( parsec_thread, t ) {          \
	}

#define PARSEC_THREAD_PUSH_THREAD( parsec_thread, thread_func, thread_args ) do { \
		pthread_attr_t thread_attr; \
		pthread_attr_init(&thread_attr); \
		pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM); \
		pthread_create( &parsec_thread.thread, \
		                &thread_attr, \
		                (THREAD_FUNC_TYPE)thread_func, \
		                thread_args); \
	} while(0)

#define PARSEC_THREAD_PUSH_THREAD_AND_WAIT( parsec_thread, thread_func, thread_args ) do { \
		void *ret = NULL; \
		PARSEC_THREAD_PUSH_THREAD(parsec_thread, thread_func, thread_args); \
		pthread_join( parsec_thread.thread, &ret ); \
	} while(0)

#define PARSEC_THREAD_CREATE( parsec_thread, thread_init, thread_args ) do { \
        pthread_attr_t thread_attr;                                     \
        pthread_attr_init(&thread_attr);                                \
        pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);      \
        pthread_create( &parsec_thread.thread,                           \
                        &thread_attr,                                   \
                        thread_init,                                    \
                        thread_args);                                   \
    } while(0)

#define PARSEC_THREAD_MASTER_CREATE( parsec_thread, t_func, t_args ) do { \
        (t_func)( t_args );                                             \
    } while(0)

#define PARSEC_THREAD_CREATE_JOIN( t_func, t_args ) do {                 \
		void *ret = NULL; \
		pthread_t thread; \
		pthread_attr_t thread_attr; \
		pthread_attr_init(&thread_attr); \
		pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM); \
        pthread_create( &thread,                                        \
                        &thread_attr,                                   \
                        (THREAD_FUNC_TYPE)t_func,                       \
                        t_args);                                        \
        pthread_join( thread, &ret );                                   \
    } while(0)

#define PARSEC_THREAD_BINDING_WAIT( barrier ) do {  \
        parsec_barrier_wait( barrier );          \
    } while(0)

#define PARSEC_THREAD_THREAD_JOIN( parsec_thread, p_ret ) do {    \
        pthread_join( parsec_thread.thread, p_ret );      \
    } while(0)

#define PARSEC_THREAD_STREAM_JOIN( parsec_thread ) {      \
    }

#define PARSEC_THREAD_DELETE( context, nb_threads ) do {  \
        free( context->parsec_threads );                  \
        context->parsec_threads = NULL;                   \
    } while(0)

#define PARSEC_THREAD_INIT( parsec_thread ) do {  \
    } while(0)

#define PARSEC_THREAD_STANDALONE_INIT( parsec_thread, t ) do {      \
    } while(0)

#define PARSEC_THREAD_SET_MIGRATABLE( bool ) do {                \
	} while(0)

typedef pthread_mutex_t parsec_thread_mutex_t;
typedef pthread_cond_t  parsec_thread_cond_t;
#define PARSEC_THREAD_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define PARSEC_THREAD_MUTEX_NULL        PTHREAD_MUTEX_INITIALIZER

#define PARSEC_THREAD_CONTEXT_WAIT( parsec_thread, context_wait_fn, eu )  \
    do { context_wait_fn( eu ); } while(0)

#define PARSEC_THREAD_YIELD() do {                       \
        sched_yield();                                  \
    } while(0)

#define PARSEC_THREAD_PAUSE( time ) do { \
        nanosleep(&time, NULL);                         \
    } while(0)




static inline int PARSEC_THREAD_MUTEX_CREATE( parsec_thread_mutex_t* p_mutex, const void* attr ) {
    int ret;
    do {
        ret = pthread_mutex_init( p_mutex, (const pthread_mutexattr_t*)attr );
    } while(0);
    return ret;
}

#define PARSEC_THREAD_MUTEX_DESTROY( p_mutex )           \
    do { pthread_mutex_destroy( p_mutex ); } while(0)

#define PARSEC_THREAD_MUTEX_LOCK( p_mutex )              \
    do { pthread_mutex_lock( p_mutex ); } while(0)

#define PARSEC_THREAD_MUTEX_UNLOCK( mutex )              \
    do { pthread_mutex_unlock( mutex ); } while(0)

static inline int PARSEC_THREAD_COND_CREATE( parsec_thread_cond_t* p_cond, const void* attr ) {
    int ret;
    do {
        ret = pthread_cond_init( p_cond, (const pthread_condattr_t*)attr );
    } while(0);
    return ret;
}

#define PARSEC_THREAD_COND_DESTROY( p_cond )             \
    do { pthread_cond_destroy( p_cond ); } while(0)

#define PARSEC_THREAD_COND_BROADCAST( cond )             \
    do { pthread_cond_broadcast( cond ); } while(0)

#define PARSEC_THREAD_COND_WAIT( cond, mutex )           \
    do { pthread_cond_wait( cond, mutex ); } while(0)

#define PARSEC_THREAD_COND_SIGNAL( p_cond )              \
    do { pthread_cond_signal( p_cond ); } while(0)



