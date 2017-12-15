/*
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>

#include "parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/mca/mca_repository.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/profiling.h"
#include "parsec/datarepo.h"
#include "parsec/bindthread.h"
#include "parsec/execution_unit.h"
#include "parsec/vpmap.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/os-spec-timing.h"
#include "parsec/remote_dep.h"

#include "parsec/ayudame.h"
#include "parsec/constants.h"

#include "parsec/thread/thread.h"

static int sched_init(ABT_sched sched, ABT_sched_config config);
static void sched_run(ABT_sched sched);
static int sched_free(ABT_sched sched);

static ABT_bool ask_add_xstream(void *user_arg, void *abt_arg);
static ABT_bool act_add_xstream(void *user_arg, void *abt_arg);
static ABT_bool ask_stop_xstream(void *user_arg, void *abt_arg);
static ABT_bool act_stop_xstream(void *user_arg, void *abt_arg);

static inline void handle_error(const char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}

extern parsec_thread_t dep_thread;

void comm_sched_init(parsec_context_t* context, parsec_thread_t* comm_thread)
{
    int ret;
    ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE, &comm_thread->pool);
    HANDLE_ABT_ERROR( ret, "ABT_pool_create_basic" );

    ABT_sched_config config;

    ABT_sched_config_var cv_event_freq = {
        .idx = 0,
        .type = ABT_SCHED_CONFIG_INT
    };

    ABT_sched_config_var cv_idx = {
        .idx = 1,
        .type = ABT_SCHED_CONFIG_INT
    };

    ABT_sched_config_var context_idx = {
        .idx = 2,
        .type = ABT_SCHED_CONFIG_PTR
    };

    ABT_sched_def sched_def = {
        .type = ABT_SCHED_TYPE_ULT,
        .init = sched_init,
        .run = sched_run,
        .free = sched_free,
        .get_migr_pool = NULL
    };

    /* Create a scheduler */
    ret = ABT_sched_config_create(&config,
                                  cv_event_freq, 1,
                                  cv_idx, 42,
                                  context_idx, context,
                                  ABT_sched_config_var_end);
    HANDLE_ABT_ERROR( ret, "ABT_sched_config_create" );

    /* int nb_threads, i; */
    /* PARSEC_THREAD_GET_NUMBER(context, &nb_threads); */
    /* ABT_pool *pools = (ABT_pool*)malloc((1+nb_threads)*sizeof(ABT_pool)); */
    /* for ( i = 0; i < nb_threads; ++i ) { */
    /*     pools[i] = ABT_POOL_NULL; */
    /*     while ( context->parsec_threads[i].pool == ABT_POOL_NULL ) */
    /*         ABT_thread_yield(); */
    /*     pools[i] = context->parsec_threads[i].pool; */
    /* } */
    /* pools[nb_threads] = comm_thread->pool; */
    /* ret = ABT_sched_create(&sched_def, 1+nb_threads, pools, config, &comm_thread->sched); */

    ret = ABT_sched_create(&sched_def, 1, &comm_thread->pool, config, &comm_thread->sched);
    HANDLE_ABT_ERROR( ret, "ABT_sched_create" );

    ABT_sched_config_free(&config);
}

void comm_sched_free(parsec_thread_t *comm_thread) {
    int ret;
    if ( comm_thread->pool != ABT_POOL_NULL ) {
        ret = ABT_pool_free( &comm_thread->pool );
        HANDLE_ABT_ERROR( ret, "ABT_pool_free" );
        comm_thread->pool = ABT_POOL_NULL;
    }

    if ( comm_thread->sched != ABT_SCHED_NULL ) {
        ret = ABT_sched_free( &comm_thread->sched );
        HANDLE_ABT_ERROR( ret, "ABT_sched_free" );
        comm_thread->sched = ABT_SCHED_NULL;
    }
}


void comm_sched_register_callbacks(parsec_context_t* parsec_context)
{
    cb_data_t* data = (cb_data_t*)malloc(sizeof(cb_data_t));
    data->context = parsec_context;

    /* ABT_event_add_callback(ABT_EVENT_STOP_XSTREAM, */
    /*                        ask_stop_xstream, data, */
    /*                        act_stop_xstream, data, */
    /*                        &data->stop_cb_id); */
    /* ABT_event_add_callback(ABT_EVENT_ADD_XSTREAM, */
    /*                        ask_add_xstream, data, */
    /*                        act_add_xstream, data, */
    /*                        &data->add_cb_id); */
}

void comm_sched_unregister_callbacks(cb_data_t* data)
{
    /* Delete registered callbacks */
    /* ABT_event_del_callback(ABT_EVENT_STOP_XSTREAM, data->stop_cb_id); */
    /* ABT_event_del_callback(ABT_EVENT_ADD_XSTREAM, data->add_cb_id); */
    free(data);
}

static ABT_bool ask_add_xstream(void *user_arg, void *abt_arg)
{
    cb_data_t *data = (cb_data_t *)user_arg;
    parsec_context_t* context = data->context;
    int rank = (int)(intptr_t)abt_arg;
    int i, j;
    ABT_bool one_sleeping = ABT_FALSE;
    parsec_execution_unit_t* eu;
    parsec_vp_t* vp;

    if (rank == ABT_XSTREAM_ANY_RANK) {
        for(i=0; i<context->nb_vp && !one_sleeping; ++i) {
            vp = context->virtual_processes[i];
            for (j = 0; j < vp->nb_cores; ++j) {
                eu = vp->execution_units[j];
                if ( context->parsec_threads[ eu->th_id ].status == STREAM_SLEEPING ) {
                    one_sleeping = ABT_TRUE;
                    break;
                }
            }
        }
        return one_sleeping;
    }

    for(i=0; i<context->nb_vp && !one_sleeping; ++i) {
        vp = context->virtual_processes[i];
        for (j = 0; j < vp->nb_cores; ++j) {
            eu = vp->execution_units[j];
            if ( eu->core_id == rank )
                if ( context->parsec_threads[ eu->th_id ].status == STREAM_SLEEPING )
                    return ABT_TRUE;
        }
    }
    return ABT_FALSE;
}

static ABT_bool ask_stop_xstream(void *user_arg, void *abt_arg)
{
    (void)user_arg;
    ABT_bool is_primary;
    ABT_xstream tar_xstream = (ABT_xstream)abt_arg;

    /* The primary ES cannot be stopped */
    ABT_xstream_is_primary(tar_xstream, &is_primary);
    if (is_primary == ABT_TRUE) {
        return ABT_FALSE;
    }

    /*Protect the comm stream*/
    if (tar_xstream == dep_thread.stream)
        return ABT_FALSE;

    return ABT_TRUE;
}

static ABT_bool act_stop_xstream(void *user_arg, void *abt_arg)
{
    cb_data_t* data = (cb_data_t*)user_arg;
    parsec_context_t* context = data->context;
    ABT_xstream xstream = (ABT_xstream)abt_arg;
    int th_id;
    ABT_xstream_get_rank(xstream, &th_id);

    /*migrate the ULT*/
    if ( context->parsec_threads[ th_id ].status == STREAM_RUNNING ) {
        ABT_mutex_spinlock(context->parsec_threads[ th_id ].mutex);
        /* ABT_pool *pools; */
        /* int nb_pools; */
        /* ABT_sched_get_num_pools(); */


        context->parsec_threads[ th_id ].status = STREAM_STOPPING;
        printf("[P%d:Demo] th_id=%d on core X is now sleeping!\n",
               context->my_rank, th_id);
        context->parsec_threads[ th_id ].stream = ABT_XSTREAM_NULL;
        ABT_mutex_unlock(context->parsec_threads[ th_id ].mutex);
        return ABT_TRUE;
    }
    return ABT_FALSE; /*Will it try again? :D*/
}

static ABT_bool act_add_xstream(void *user_arg, void *abt_arg)
{
    cb_data_t* data = (cb_data_t*)user_arg;
    parsec_context_t* context = data->context;
    int tar_rank = (int)(intptr_t)abt_arg;
    ABT_bool result = ABT_TRUE;
    int rank, ret;
    int i, j;
    parsec_execution_unit_t* eu;
    parsec_vp_t* vp;

    /* Create a new ES */
    if (tar_rank == ABT_XSTREAM_ANY_RANK) {
        for(i=0; i<context->nb_vp; ++i) {
            vp = context->virtual_processes[i];
            for (j = 0; j < vp->nb_cores; ++j) {
                eu = vp->execution_units[j];
                if ( context->parsec_threads[ eu->th_id ].status == STREAM_SLEEPING ) {
                    rank = eu->th_id;
                    goto found_target;
                }
            }
        }
        return ABT_FALSE; /*didn't find any valid stream to start*/
    }
    else {
        if ( context->parsec_threads[ tar_rank ].status != STREAM_SLEEPING )
            return ABT_FALSE;
        rank = tar_rank;
    }

  found_target:
    /*We need a binding xstream create func*/
    ret = ABT_xstream_create(context->parsec_threads[rank].sched, &context->parsec_threads[rank].stream);
    if (ret != ABT_SUCCESS) {
        result = ABT_FALSE;
        printf("ES%d: failed to create\n", rank);
        goto fn_exit;
    }

    ABT_mutex_spinlock(context->parsec_threads[rank].mutex);
    context->parsec_threads[rank].status = STREAM_READY;
    ABT_cond_signal(context->parsec_threads[rank].cond);
    printf("[P%d:Demo] new stream %d\n", context->my_rank, rank);
    ABT_mutex_unlock(context->parsec_threads[rank].mutex);
    return ABT_SUCCESS;

  fn_exit:
    return result;
}

#if ( DEMO_SC == 2 )
void parsec_thread_check_status(void *arg)
{
    parsec_execution_unit_t *eu = (parsec_execution_unit_t*)arg;
    parsec_context_t* parsec_context = eu->virtual_process->parsec_context;
    int th_id = eu->th_id;

    if ( parsec_context->parsec_threads[th_id].status == STREAM_STOPPING ) {
#if 0
        /*scheduler is in charge now*/
        ABT_xstream me;
        ABT_xstream_self(&me);
        int rank;
        ABT_xstream_get_rank(me, &rank);
        ABT_mutex_lock(parsec_context->parsec_threads[ eu->th_id ].mutex);
        fprintf(stderr, "yielding from stream %d\n", rank);

        ABT_thread_yield();
        parsec_context->parsec_threads[th_id].status = STREAM_SLEEPING;

        ABT_xstream_self(&me);
        ABT_xstream_get_rank(me, &rank);
        fprintf(stderr, "sleeping in stream %d\n", rank);
        ABT_cond_wait(parsec_context->parsec_threads[ eu->th_id ].cond,
                      parsec_context->parsec_threads[ eu->th_id ].mutex);

        ABT_xstream_self(&me);
        ABT_xstream_get_rank(me, &rank);
        fprintf(stderr, "waking up in stream %d\n", rank);
        parsec_context->parsec_threads[th_id].status = STREAM_RUNNING;
        ABT_mutex_unlock(parsec_context->parsec_threads[ eu->th_id ].mutex);
    }
#endif
        ABT_thread me;
        int rank;
        ABT_thread_self(&me);
        ABT_xstream_get_rank(me, &rank);
        ABT_mutex_lock(parsec_context->parsec_threads[ eu->th_id ].mutex);
        ABT_thread_set_migratable(me, ABT_TRUE);
        parsec_context->parsec_threads[th_id].status = STREAM_GOING_TO_BED;
        ABT_thread_migrate_to_pool(me, parsec_context->monitoring_steering_threads[0].pool); /*I lost my quantum*/
        parsec_context->parsec_threads[th_id].status = STREAM_SLEEPING; /* got my quantum back from comm stream*/
        ABT_thread_self(&me);
        ABT_xstream_get_rank(me, &rank);
        printf("[P%d:%d] I'll be back, sent from my stream %d\n", parsec_context->my_rank, th_id, rank);
        /*by waiting, I yield to the scheduler, so it can execute the callback*/
        ABT_cond_wait(parsec_context->parsec_threads[ eu->th_id ].cond,
                      parsec_context->parsec_threads[ eu->th_id ].mutex);
        ABT_thread_self(&me);
        ABT_xstream_get_rank(me, &rank);
        /*by leaving the wait, an order to wake me has been issued*/
        parsec_context->parsec_threads[th_id].status = STREAM_LEAVING_BED;
        if (parsec_context->parsec_threads[th_id].stream != ABT_XSTREAM_NULL) {
            ABT_thread_migrate_to_pool(me, parsec_context->parsec_threads[ eu->th_id ].pool);
            parsec_context->parsec_threads[th_id].status = STREAM_RUNNING;
            printf("[P%d:%d] I'm back! sent from my stream %d\n", parsec_context->my_rank, th_id, rank);
        }
        else {
            parsec_context->parsec_threads[th_id].status = STREAM_RUNNING;
            printf("[P%d:%d] I'm back to finish! sent from my stream %d\n", parsec_context->my_rank, th_id, rank);
        }
        ABT_mutex_unlock(parsec_context->parsec_threads[ eu->th_id ].mutex);
    }

}
#endif



/******************************************************************************/
/* Scheduler data structure and functions                                     */
/******************************************************************************/
typedef struct {
    uint32_t event_freq;
    int idx;
    parsec_context_t* context;/*not sure I need it*/
} sched_data_t;

static int sched_init(ABT_sched sched, ABT_sched_config config)
{
    int ret = ABT_SUCCESS;

    sched_data_t *p_data = (sched_data_t *)calloc(1, sizeof(sched_data_t));

    ABT_sched_config_read(config, 3, &p_data->event_freq, &p_data->idx, &p_data->context);
    ret = ABT_sched_set_data(sched, (void *)p_data);

    return ret;
}

static void sched_run(ABT_sched sched)
{
    ABT_xstream me;
    ABT_xstream_self(&me);
    int rank;
    ABT_xstream_get_rank(me, &rank);
    ABT_pool pool;
    ABT_unit unit;

    ABT_sched_get_pools(sched, 1, 0, &pool);
    sched_data_t *p_data;
    ABT_sched_get_data(sched, (void**)&p_data);
    parsec_context_t *context = p_data->context;

    while (1) {
        /* Execute one work unit from the scheduler's pool */
        ABT_pool_pop(pool, &unit);
        if (unit != ABT_UNIT_NULL) {
            ABT_xstream_run_unit(unit, pool);
        }

        ABT_bool stop;
        int ret = ABT_sched_has_to_stop(sched, &stop);
        if (stop == ABT_TRUE) break;
        ABT_xstream_check_events(sched);
    }
}

static int sched_free(ABT_sched sched)
{
    sched_data_t *p_data;

    ABT_sched_get_data(sched, (void **)&p_data);
    free(p_data);

    return ABT_SUCCESS;
}

