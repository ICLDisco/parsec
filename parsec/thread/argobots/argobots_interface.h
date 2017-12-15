/**
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "abt.h"

#define HANDLE_ABT_ERROR(ret,msg)                       \
    if (ret != ABT_SUCCESS) {                           \
        fprintf(stderr, "ERROR[%d]: %s\n", ret, msg);   \
        exit(EXIT_FAILURE);                             \
    }

enum stream_state {
    STREAM_READY, /*setting the order to run*/
    STREAM_RUNNING,
    STREAM_STOPPING, /*setting the order to stop*/
    STREAM_GOING_TO_BED, /*migrating from its stream to the comm stream*/
    STREAM_SLEEPING,
    STREAM_DESTROYED
};

struct parsec_thread_s {
    void            *context;
    int              rank;
    ABT_xstream      stream;
    ABT_sched        sched;
    ABT_pool         pool;
    /* int         nb_threads;  /\* number of ULT in this stream *\/ */
    /* ABT_thread* threads; */
    volatile enum stream_state status; /*when put to sleep, a stream will not pick more task, it will block on future and wait for a wake up*/
    ABT_future       future;
    ABT_cond         cond;
    ABT_mutex        mutex;
    ABT_thread       thread[1];
};

#define STACKSIZE (4096*1024)
#define ULTNUMBER 16

typedef struct parsec_thread_s parsec_thread_t;


void parsec_thread_check_status(void *eu);
/* void* __listener_thread(void *arguments); */



#define PARSEC_THREAD_LIBRARY_INIT(pargc, pargv) do {   \
    ABT_init(*pargc, *pargv);                           \
    } while(0)

#define PARSEC_THREAD_LIBRARY_FINI() do {       \
    ABT_finalize();                             \
    } while(0)

#define PARSEC_THREAD_PARSEC_THREAD_INIT( parsec_thread ) do { \
        (parsec_thread).stream     = ABT_XSTREAM_NULL;         \
        (parsec_thread).sched      = ABT_SCHED_NULL;           \
        (parsec_thread).pool       = ABT_POOL_NULL;            \
        (parsec_thread).status     = STREAM_READY;             \
        (parsec_thread).future     = ABT_FUTURE_NULL;          \
    } while(0)

#define PARSEC_THREAD_CONTEXT_MALLOC2( parsec_context, nb_threads ) do { \
        (parsec_context)->parsec_threads =                              \
            (parsec_thread_t*)malloc((nb_threads) * sizeof(parsec_thread_t)); \
        int t;                                                          \
        for ( t = 0; t < (nb_threads); ++t ) {                          \
            (parsec_context)->parsec_threads[t].context    = parsec_context; \
            (parsec_context)->parsec_threads[t].rank       = t;         \
            PARSEC_THREAD_PARSEC_THREAD_INIT( (parsec_context)->parsec_threads[t] ); \
            INIT_FUTURE_COND((parsec_context)->parsec_threads[t]);      \
        }                                                               \
                                                                        \
        (parsec_context)->monitoring_steering_threads =                 \
            (parsec_thread_t*)malloc(sizeof(parsec_thread_t) + 2*sizeof(ABT_thread)); \
        (parsec_context)->monitoring_steering_threads[0].context    = parsec_context; \
        (parsec_context)->monitoring_steering_threads[0].rank       = nb_threads; \
        PARSEC_THREAD_PARSEC_THREAD_INIT( (parsec_context)->monitoring_steering_threads[0] ); \
        INIT_FUTURE_COND((parsec_context)->monitoring_steering_threads[0]); \
        (parsec_context)->monitoring_steering_threads[0].thread[0] =    \
            (parsec_context)->monitoring_steering_threads[0].thread[1] = \
            (parsec_context)->monitoring_steering_threads[0].thread[2] = ABT_THREAD_NULL; \
    } while(0)

#define PARSEC_THREAD_CONTEXT_FREE( parsec_context ) do {                 \
        int t, nb_threads;                                              \
        PARSEC_THREAD_GET_NUMBER( parsec_context, &nb_threads );          \
        for ( t = 0; t < nb_threads; ++t )                              \
            PARSEC_THREAD_FREE( (parsec_context)->parsec_threads[t] );     \
        free( (parsec_context)->parsec_threads );                         \
    } while(0)

#define PARSEC_THREAD_SCHED_INIT( parsec_thread, sched_init_func ) do { \
        sched_init_func( (parsec_thread).context, &(parsec_thread) );   \
    } while(0)

#define PARSEC_THREAD_SCHED_FREE( parsec_thread, sched_free_func ) do { \
        if ( (parsec_thread).sched != ABT_SCHED_NULL ) {                \
            sched_free_func( &(parsec_thread) );                        \
            (parsec_thread).sched = ABT_SCHED_NULL;                     \
        }                                                               \
    } while(0)

#define PARSEC_THREAD_STREAM_INIT( parsec_thread, sched_init_func ) do { \
        int ret;                                                        \
        PARSEC_THREAD_SCHED_INIT( parsec_thread, sched_init_func );     \
        if ( 0 == (parsec_thread).rank ) {                              \
            ret = ABT_xstream_self( &(parsec_thread).stream );          \
            HANDLE_ABT_ERROR( ret, "ABT_xstream_self" );                \
            ret = ABT_xstream_set_main_sched( (parsec_thread).stream, (parsec_thread).sched ); \
            HANDLE_ABT_ERROR( ret, "ABT_xstream_set_main_sched" );      \
        }                                                               \
        else if ( 0 < (parsec_thread).rank ) {                          \
            ret = ABT_xstream_create_with_rank( (parsec_thread).sched,  \
                                                (parsec_thread).rank,   \
                                                &(parsec_thread).stream ); \
            HANDLE_ABT_ERROR( ret, "ABT_xstream_create_with_rank" );    \
        }                                                               \
        else {                                                          \
            ret = ABT_xstream_create( (parsec_thread).sched,            \
                                      &(parsec_thread).stream );        \
            HANDLE_ABT_ERROR( ret, "ABT_xstream_create" );              \
            ret = ABT_xstream_get_rank( (parsec_thread).stream,         \
                                        &(parsec_thread).rank );        \
            HANDLE_ABT_ERROR( ret, "ABT_xstream_get_rank" );            \
        }                                                               \
        ret = ABT_sched_get_pools( (parsec_thread).sched, 1, 0,         \
                                   &(parsec_thread).pool );             \
        HANDLE_ABT_ERROR( ret, "ABT_sched_get_pools" );                 \
        ret = ABT_cond_create( &(parsec_thread).cond );                 \
        HANDLE_ABT_ERROR( ret, "ABT_cond_create" );                     \
        ret = ABT_mutex_create( &(parsec_thread).mutex );               \
        HANDLE_ABT_ERROR( ret, "ABT_mutex_create" );                    \
    } while(0)

#define PARSEC_THREAD_STREAM_RECREATE( parsec_thread ) do {             \
        int ret;                                                        \
        ret = ABT_xstream_create_with_rank( (parsec_thread).sched,      \
                                            (parsec_thread).rank,       \
                                            &(parsec_thread).stream );  \
        HANDLE_ABT_ERROR( ret, "ABT_xstream_create_with_rank" );        \
        (parsec_thread).status = STREAM_READY;                          \
    } while(0)

#define PARSEC_THREAD_PUSH_THREAD( parsec_thread, thread_func, thread_args ) do { \
        ABT_thread ult;                                                 \
        ABT_thread_attr attr;                                           \
        ABT_thread_attr_create(&attr);                                  \
        ABT_thread_attr_set_stacksize(attr, STACKSIZE);                 \
        int ret = ABT_thread_create( (parsec_thread).pool,              \
                                     (THREAD_FUNC_TYPE)thread_func,     \
                                     thread_args,                       \
                                     attr,                              \
                                     &ult );                            \
        HANDLE_ABT_ERROR( ret, "ABT_thread_create" );                   \
        if ( (parsec_thread).status == STREAM_READY )                   \
            (parsec_thread).status = STREAM_RUNNING;                    \
    } while(0)

#define PARSEC_THREAD_PUSH_THREAD_AND_WAIT( parsec_thread, thread_func, thread_args ) do { \
        ABT_thread_attr attr;                                           \
        ABT_thread_attr_create(&attr);                                  \
        ABT_thread_attr_set_stacksize(attr, STACKSIZE);                 \
        int ret = ABT_thread_create( (parsec_thread).pool,              \
                                     (void (*)(void *))thread_func,     \
                                     thread_args,                       \
                                     attr,                              \
                                     (parsec_thread).thread );          \
        HANDLE_ABT_ERROR( ret, "ABT_thread_create" );                   \
        if ( (parsec_thread).status == STREAM_READY )                   \
            (parsec_thread).status = STREAM_RUNNING;                    \
        ABT_thread_join( (parsec_thread).thread[0] );                   \
        ABT_thread_free( (parsec_thread).thread );                      \
    } while(0)

#define PARSEC_THREAD_PUSH_TASK( parsec_thread, task_func, task_args ) do { \
        ABT_task task;                                                  \
        int ret = ABT_task_create( (parsec_thread).pool,                \
                                   (void (*)(void *))task_func,         \
                                   task_args,                           \
                                   &task );                             \
        HANDLE_ABT_ERROR( ret, "ABT_task_create" );                     \
    } while(0)

#define PARSEC_THREAD_JOIN_THREAD( parsec_thread, p_ret ) do {          \
        (void)p_ret;                                                    \
        int ret;                                                        \
        ret = ABT_thread_join(  (parsec_thread).thread[0] );            \
        HANDLE_ABT_ERROR( ret, "ABT_thread_join" );                     \
        ret = ABT_thread_free( (parsec_thread).thread );                \
    } while(0)

#define PARSEC_THREAD_STREAM_FREE( parsec_thread ) do {                 \
        int ret;                                                        \
        if ( (parsec_thread).stream != ABT_XSTREAM_NULL ) {             \
            ret = ABT_xstream_join( parsec_thread.stream );             \
            HANDLE_ABT_ERROR( ret, "ABT_cond_create" );                 \
            ret = ABT_xstream_free( &(parsec_thread).stream );          \
            HANDLE_ABT_ERROR( ret, "ABT_cond_create" );                 \
            (parsec_thread).stream = ABT_XSTREAM_NULL;                  \
            (parsec_thread).status = STREAM_SLEEPING;                   \
        }                                                               \
    } while(0)

#define PARSEC_THREAD_FREE( parsec_thread ) do {                        \
        if ((parsec_thread).future != ABT_FUTURE_NULL)                  \
            ABT_future_free(&(parsec_thread).future);                   \
        (parsec_thread).future = ABT_FUTURE_NULL;                       \
        if ((parsec_thread).cond != ABT_COND_NULL)                      \
            ABT_cond_free(&(parsec_thread).cond);                       \
        (parsec_thread).cond = ABT_COND_NULL;                           \
        if ((parsec_thread).mutex != ABT_MUTEX_NULL)                    \
            ABT_mutex_free(&(parsec_thread).mutex);                     \
        (parsec_thread).mutex = ABT_MUTEX_NULL;                         \
        (parsec_thread).rank = -1;                                      \
        (parsec_thread).status = STREAM_DESTROYED;                      \
        PARSEC_THREAD_SCHED_FREE( parsec_thread );                      \
        PARSEC_THREAD_STREAM_FREE( parsec_thread );                     \
    } while(0)

static inline void PARSEC_THREAD_CONTEXT_WAIT( parsec_thread_t parsec_thread,
                                              void (*context_wait_fn)(void*),
                                              void* eu )
{
    ABT_thread ult;
    ABT_thread_attr attr;
    ABT_thread_attr_create(&attr);
    ABT_thread_attr_set_stacksize(attr, STACKSIZE);
    ABT_thread_create((parsec_thread).pool,
                      context_wait_fn,
                      eu,
                      attr,
                      &ult);
    if ( (parsec_thread).status == STREAM_READY )                        
        (parsec_thread).status = STREAM_RUNNING;                         
    ABT_thread_join( ult );
    ABT_thread_free( &ult );
}

#define PARSEC_THREAD_SET_CONCURRENCY( nb_threads ) {   \
    }

#define PARSEC_THREAD_SET_MIGRATABLE( bool ) do {               \
        int ret;                                                \
        ABT_thread me;                                          \
        if ( bool )                                             \
            ret = ABT_thread_set_migratable( me, ABT_TRUE );    \
        else                                                    \
            ret = ABT_thread_set_migratable( me, ABT_FALSE );   \
        HANDLE_ABT_ERROR( ret, "ABT_thread_set_migratable" );   \
    } while(0)

#define PARSEC_THREAD_YIELD() do {              \
        ABT_thread_yield();                     \
    } while(0)

#define PARSEC_THREAD_PAUSE( time ) do {        \
        ABT_thread_yield();                     \
    } while(0)


extern void* __listener_thread(void* arguments);
#define PARSEC_THREAD_PUSH_LISTENER( parsec_context, parsec_thread ) do { \
        ABT_thread_attr attr;                                           \
        ABT_thread_attr_create(&attr);                                  \
        ABT_thread_attr_set_stacksize(attr, STACKSIZE);                 \
        int ret = ABT_thread_create( (parsec_thread).pool,              \
                                     (void (*)(void *))__listener_thread, \
                                     (void*)parsec_context,             \
                                     attr,                              \
                                     (parsec_thread).thread );          \
        HANDLE_ABT_ERROR( ret, "ABT_thread_create" );                   \
        if ( (parsec_thread).status == STREAM_READY )                   \
            (parsec_thread).status = STREAM_RUNNING;                    \
    } while(0)

extern void* __monitoring_thread(void* arguments);
#define PARSEC_THREAD_PUSH_MONITORING( parsec_context, parsec_thread ) do { \
        ABT_thread_attr attr;                                           \
        ABT_thread_attr_create(&attr);                                  \
        ABT_thread_attr_set_stacksize(attr, STACKSIZE);                 \
        int ret = ABT_thread_create( (parsec_thread).pool,              \
                                     (void (*)(void *))__monitoring_thread, \
                                     (void*)parsec_context,             \
                                     attr,                              \
                                     (parsec_thread).thread+1 );        \
        HANDLE_ABT_ERROR( ret, "ABT_thread_create" );                   \
        if ( (parsec_thread).status == STREAM_READY )                   \
            (parsec_thread).status = STREAM_RUNNING;                    \
    } while(0)

extern void* __publisher_thread(void* arguments);
#define PARSEC_THREAD_PUSH_PUBLISHER( parsec_context, parsec_thread ) do { \
        ABT_thread_attr attr;                                           \
        ABT_thread_attr_create(&attr);                                  \
        ABT_thread_attr_set_stacksize(attr, STACKSIZE);                 \
        int ret = ABT_thread_create( (parsec_thread).pool,              \
                                     (void (*)(void *))__publisher_thread, \
                                     (void*)parsec_context,             \
                                     attr,                              \
                                     (parsec_thread).thread+2 );        \
        HANDLE_ABT_ERROR( ret, "ABT_thread_create" );                   \
        if ( (parsec_thread).status == STREAM_READY )                   \
            (parsec_thread).status = STREAM_RUNNING;                    \
    } while(0)


#ifdef DEMO_SC
#define CREATE_FUTURE_COND(parsec_thread) {         \
        (parsec_thread).future = ABT_FUTURE_NULL;   \
        (parsec_thread).cond   = ABT_COND_NULL;     \
        (parsec_thread).mutex  = ABT_MUTEX_NULL;    \
    }

#define INIT_FUTURE_COND(parsec_thread)  {                              \
        ret = ABT_cond_create( &(parsec_thread).cond );                 \
        HANDLE_ABT_ERROR( ret, "ABT_cond_create" );                     \
        ret = ABT_mutex_create( &(parsec_thread).mutex );               \
        HANDLE_ABT_ERROR( ret, "ABT_mutex_create" );                    \
    }

#define FREE_FUTURE_COND(parsec_thread) {                    \
        if ((parsec_thread).future != ABT_FUTURE_NULL)       \
            ABT_future_free(&(parsec_thread).future);        \
        (parsec_thread).future = ABT_FUTURE_NULL;            \
        ABT_cond_free(&(parsec_thread).cond);                \
        (parsec_thread).cond = ABT_COND_NULL;                \
        ABT_mutex_free(&(parsec_thread).mutex);              \
        (parsec_thread).mutex = ABT_MUTEX_NULL;              \
    }
#else
#define INIT_FUTURE_COND(parsec_thread) { }

#define CREATE_FUTURE_COND(parsec_thread) { }

#define FREE_FUTURE_COND(parsec_thread) { }

#endif /*DEMO_SC*/









#define PARSEC_THREAD_SET_CONCURRENCY( nb_threads ) {    \
    }

#define PARSEC_THREAD_CONTEXT_MALLOC( context, nb_threads ) do {         \
        (context)->parsec_threads = (parsec_thread_t*)malloc((nb_threads) * sizeof(parsec_thread_t)); \
        int t;                                                          \
        for ( t = 0; t < (nb_threads); ++t ) {                          \
            (context)->parsec_threads[t].context = context;              \
            (context)->parsec_threads[t].stream = ABT_XSTREAM_NULL;      \
            (context)->parsec_threads[t].sched  = ABT_SCHED_NULL;        \
            (context)->parsec_threads[t].pool   = ABT_POOL_NULL;         \
            (context)->parsec_threads[t].status = STREAM_READY;          \
            INIT_FUTURE_COND((context)->parsec_threads[t]);              \
        }                                                               \
    } while(0)

#define PARSEC_THREAD_STREAM_CREATE( parsec_thread, t ) do {              \
        int ret;                                                        \
        comm_sched_init( (parsec_thread).context, &(parsec_thread) );     \
        ret = ABT_xstream_create( (parsec_thread).sched, &(parsec_thread).stream ); \
        HANDLE_ABT_ERROR( ret, "ABT_xstream_create" );                  \
        if ( t > 0 ) {                                                  \
        ret = ABT_xstream_set_rank( (parsec_thread).stream, t );         \
        HANDLE_ABT_ERROR( ret, "ABT_xstream_set_rank" );                \
        }                                                               \
        ret = ABT_xstream_get_main_sched( (parsec_thread).stream, &(parsec_thread).sched ); \
        HANDLE_ABT_ERROR( ret, "ABT_xstream_get_main_sched" );		\
        ret = ABT_sched_get_pools( (parsec_thread).sched, 1, 0, &(parsec_thread).pool ); \
        HANDLE_ABT_ERROR( ret, "ABT_sched_get_pools" );			\
        (parsec_thread).status = STREAM_RUNNING;                           \
        CREATE_FUTURE_COND( parsec_thread );                             \
    } while(0)


#define PARSEC_THREAD_STREAM_MASTER_CREATE( parsec_thread, t ) do {       \
        int ret;                                                        \
        ret = ABT_xstream_self( &(parsec_thread).stream );               \
        HANDLE_ABT_ERROR( ret, "ABT_xstream_self" );                    \
        ret = ABT_xstream_set_rank( (parsec_thread).stream, t );         \
        HANDLE_ABT_ERROR( ret, "ABT_xstream_set_rank" );                \
        ret = ABT_xstream_get_main_sched( (parsec_thread).stream, &(parsec_thread).sched ); \
        HANDLE_ABT_ERROR( ret, "ABT_xstream_get_main_sched" );		\
        ret = ABT_sched_get_pools( (parsec_thread).sched, 1, 0, &(parsec_thread).pool ); \
        HANDLE_ABT_ERROR( ret, "ABT_sched_get_pools" );			\
        (parsec_thread).status = STREAM_RUNNING;                           \
        CREATE_FUTURE_COND( parsec_thread );                             \
    } while(0)

#define PARSEC_THREAD_CREATE( parsec_thread, thread_func, thread_args ) do { \
        ABT_thread_attr attr;                                           \
        ABT_thread_attr_create(&attr);                                  \
        ABT_thread_attr_set_stacksize(attr, STACKSIZE);                 \
        int ret = ABT_thread_create( (parsec_thread).pool,               \
                                     (void (*)(void *))thread_func,     \
                                     thread_args,                       \
                                     attr,                              \
                                     (parsec_thread).thread );           \
        if ( (parsec_thread).status == STREAM_READY )                    \
            (parsec_thread).status = STREAM_RUNNING;                     \
        HANDLE_ABT_ERROR( ret, "ABT_thread_create" );                   \
    } while(0)

#define PARSEC_THREAD_MASTER_CREATE( parsec_thread, t_func, t_args ) do { \
        ABT_thread_attr attr;                                           \
        ABT_thread_attr_create(&attr);                                  \
        ABT_thread_attr_set_stacksize(attr, STACKSIZE);                 \
        int ret = ABT_thread_create( (parsec_thread).pool,               \
                                     (void (*)(void *))t_func,          \
                                     t_args,                            \
                                     attr,                              \
                                     (parsec_thread).thread );           \
        HANDLE_ABT_ERROR( ret, "ABT_thread_create" );                   \
        if ( (parsec_thread).status == STREAM_READY )                    \
            (parsec_thread).status = STREAM_RUNNING;                     \
        ABT_thread_join( (parsec_thread).thread[0] );                    \
        ABT_thread_free( (parsec_thread).thread );                       \
    } while(0)

#define PARSEC_THREAD_CREATE_JOIN( t_func, t_args ) do {                 \
        ABT_xstream stream;                                             \
        ABT_sched sched;						\
        ABT_pool pool;							\
        ABT_thread thread;                                              \
        ABT_thread_attr attr;                                           \
        int ret = ABT_xstream_self( &stream );				\
        HANDLE_ABT_ERROR( ret, "ABT_xstream_self" );			\
        ret = ABT_thread_attr_create(&attr);                            \
        ret = ABT_thread_attr_set_stacksize(attr, STACKSIZE);           \
        ret = ABT_xstream_get_main_sched( stream, &sched );		\
        HANDLE_ABT_ERROR( ret, "ABT_xstream_get_main_sched" );		\
        ret = ABT_sched_get_pools( sched, 1, 0, &pool );		\
        HANDLE_ABT_ERROR( ret, "ABT_sched_get_pools" );			\
        ret = ABT_thread_create( pool,					\
                                 (void (*)(void *))t_func,              \
                                 t_args,                                \
                                 attr,                                  \
                                 &thread );                             \
        HANDLE_ABT_ERROR( ret, "ABT_thread_create" );                   \
        ABT_thread_join( thread );                                      \
        ABT_thread_free( &thread );                                     \
    } while(0)

#define PARSEC_THREAD_TASK( task_func, task_args ) do {                  \
        ABT_xstream stream;                                             \
        ABT_thread task;                                                \
        int ret = ABT_xstream_create( ABT_SCHED_NULL, &stream );        \
        HANDLE_ABT_ERROR( ret, "ABT_xstream_create" );                  \
        ret = ABT_task_create( stream,                                  \
                               task_func,                               \
                               task_args,                               \
                               &task );                                 \
        HANDLE_ABT_ERROR( ret, "ABT_task_create" );                     \
        ABT_task_free( &task );                                         \
        ABT_xstream_join( stream );                                     \
        ABT_xstream_free( &stream );                                    \
    } while(0)


#define PARSEC_THREAD_BINDING_WAIT( barrier ) {   \
    }

#define PARSEC_THREAD_THREAD_JOIN( parsec_thread, p_ret ) do {            \
        (void)p_ret;                                                    \
        int ret;                                                        \
        ABT_bool flag;                                                  \
        ABT_self_is_primary(&flag);                                     \
        if (ABT_FALSE == flag) {                                        \
            ret = ABT_thread_join(  (parsec_thread).thread[0] );        \
            HANDLE_ABT_ERROR( ret, "ABT_thread_join" );                 \
            ret = ABT_thread_free( (parsec_thread).thread );            \
        }                                                               \
    } while(0)

#define PARSEC_THREAD_STREAM_JOIN( parsec_thread ) do {                 \
        ABT_bool flag;                                                  \
        ABT_self_on_primary_xstream(&flag);                             \
        if (ABT_FALSE == flag) {                                        \
            ABT_xstream_join(  (parsec_thread).stream );                \
            ABT_xstream_free( &(parsec_thread).stream );                \
        }                                                               \
        (parsec_thread).stream = ABT_XSTREAM_NULL;                      \
        (parsec_thread).sched  = ABT_SCHED_NULL;                        \
        (parsec_thread).pool   = ABT_POOL_NULL;                         \
    } while(0)

#define PARSEC_THREAD_DELETE( context, nb_threads ) do {                \
        int t;                                                          \
        for( t = 0; t < nb_threads; ++t ) {                             \
            if ( (context)->parsec_threads[t].future != ABT_FUTURE_NULL ) { \
                ABT_future_free( &(context)->parsec_threads[t].future ); \
                (context)->parsec_threads[t].future = ABT_FUTURE_NULL;  \
            }                                                           \
        }                                                               \
        free( (context)->parsec_threads );                              \
        (context)->parsec_threads = NULL;                               \
    } while(0)

#define PARSEC_THREAD_INIT( parsec_thread ) do {                          \
        PARSEC_THREAD_STREAM_MASTER_CREATE( parsec_thread );              \
    } while(0)

#define PARSEC_THREAD_STANDALONE_INIT( parsec_thread, t ) do {            \
        PARSEC_THREAD_STREAM_CREATE( parsec_thread, t );                  \
    } while(0)


/* Argo itf v2 */

#if defined(DEMO_SC)

#  if (DEMO_SC == 2) /*Custom scheduler for the stream*/
#define PARSEC_THREAD_COMM_STREAM_INIT( parsec_context, parsec_thread ) do { \
        int ret;                                                        \
        (parsec_thread).rank = -1;                                       \
        comm_sched_init( parsec_context, &(parsec_thread) );              \
        ret = ABT_xstream_create( (parsec_thread).sched, &(parsec_thread).stream ); \
        HANDLE_ABT_ERROR( ret, "ABT_xstream_create" );                  \
        (parsec_thread).status = STREAM_RUNNING;                         \
        CREATE_FUTURE_COND( parsec_thread );                             \
    } while(0)
#  else /*Regular scheduler for the stream*/
#define PARSEC_THREAD_COMM_STREAM_INIT( parsec_context, parsec_thread ) do { \
        PARSEC_THREAD_STANDALONE_INIT( parsec_thread, -1 );                  \
    } while(0)
#  endif

#else

#  if defined(COMM_SHARED)
#define PARSEC_THREAD_COMM_STREAM_INIT( parsec_context, parsec_thread ) do { \
        PARSEC_THREAD_INIT( dep_thread );                                \
    } while(0)
#  else  /*defined(COMM_EXCLUSIVE)*/
#define PARSEC_THREAD_COMM_STREAM_INIT( parsec_context, parsec_thread ) do { \
        PARSEC_THREAD_STANDALONE_INIT( parsec_thread, -1 );                  \
    } while(0)
#  endif

#endif /*!DEMO_SC*/



/* End of Argo itf v2 */








typedef ABT_mutex parsec_thread_mutex_t;
typedef ABT_cond  parsec_thread_cond_t;
#define PARSEC_THREAD_MUTEX_INITIALIZER ABT_MUTEX_NULL
#define PARSEC_THREAD_MUTEX_NULL        ABT_MUTEX_NULL

static inline int PARSEC_THREAD_MUTEX_CREATE( parsec_thread_mutex_t* p_mutex, const void* attr ) {
    (void)attr;
    int ret;
    do {
        ret = ABT_mutex_create( p_mutex );
    } while(0);
    return ret;
}

#define PARSEC_THREAD_MUTEX_DESTROY( p_mutex )     \
    do { if (ABT_NULL != *(p_mutex)) ABT_mutex_free( p_mutex ); } while(0)

#define PARSEC_THREAD_MUTEX_LOCK( p_mutex )        \
    do { ABT_mutex_lock( *(p_mutex) ); } while(0)

#define PARSEC_THREAD_MUTEX_UNLOCK( p_mutex )      \
    do { ABT_mutex_unlock( *(p_mutex) ); } while(0)

static inline int PARSEC_THREAD_COND_CREATE( parsec_thread_cond_t* p_cond, const void* attr ) {
    (void)attr;
    int ret;
    do {
        ret = ABT_cond_create( p_cond );
    } while(0);
    return ret;
}

#define PARSEC_THREAD_COND_DESTROY( p_cond ) \
    do { ABT_cond_free( (p_cond) ); } while(0)

#define PARSEC_THREAD_COND_BROADCAST( p_cond )     \
    do { ABT_cond_broadcast( *(p_cond) ); } while(0)

#define PARSEC_THREAD_COND_WAIT( p_cond, p_mutex )       \
    do { ABT_cond_wait( *(p_cond), *(p_mutex) ); } while(0)

#define PARSEC_THREAD_COND_SIGNAL( p_cond )              \
    do { ABT_cond_signal( *(p_cond) ); } while(0)
