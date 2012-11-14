/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "debug.h"
#include "scheduling.h"
#include "schedulers.h"
#include "dague_hwloc.h"
#include "dequeue.h"
#include "list.h"
#include "remote_dep.h"
#include "datarepo.h"
#include "maxheap.h"
#include "instrument.h"

#if defined(DAGUE_PROF_TRACE) && 0
#define TAKE_TIME(EU_PROFILE, KEY, ID)  dague_profiling_trace((EU_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(EU_PROFILE, KEY, ID) do {} while(0)
#endif

typedef struct {
	unsigned int try_count;
	unsigned int steal_count;
	//	int L1_cache_misses;
	//  int L2_cache_misses;
	//  int L3_cache_misses;
} neighbor_statistic;

int SYSTEM_NEIGHBOR = 0; // define this during init

typedef struct {
    dague_dequeue_t   *system_queue;
    dague_hbbuffer_t  *task_queue;
    int                nb_hierarch_queues;
	unsigned int       choice_count;
	neighbor_statistic *neighbor_stats;
    dague_hbbuffer_t **hierarch_queues;
} local_queues_scheduler_object_t;

#define LOCAL_QUEUES_OBJECT(eu_context) ((local_queues_scheduler_object_t*)(eu_context)->scheduler_object)

static int no_scheduler_is_active( dague_context_t *master );

static void push_in_queue_wrapper(void *store, dague_list_item_t *elt)
{
    dague_dequeue_chain_back( (dague_dequeue_t*)store, elt );
}

#ifdef HAVE_HWLOC
/** In case of hierarchical bounded buffer, define
 *  the wrappers to functions
 */
static void push_in_buffer_wrapper(void *store, dague_list_item_t *elt)
{
    /* Store is a hbbbuffer */
    dague_hbbuffer_push_all( (dague_hbbuffer_t*)store, elt );
}
#endif

/*********************************************************************/
/*************************** List of Trees ***************************/
/*********************************************************************/

#define dague_heap_priority_comparator (offsetof(dague_heap_t, priority))

// TREE
static int init_tree_queues(  dague_context_t *master )
{
    int t, p, nq = 1;
    int hwloc_levels;
    dague_execution_unit_t *eu;
    dague_vp_t * vp;
    uint32_t queue_size;
    local_queues_scheduler_object_t *sched_obj = NULL;

    if ( !no_scheduler_is_active(master) ) {
            return -1;
    }

    SYSTEM_NEIGHBOR = master->nb_vp * master->virtual_processes[0]->nb_cores; // defined for instrumentation

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = (local_queues_scheduler_object_t*)malloc(sizeof(local_queues_scheduler_object_t));
            eu->scheduler_object = sched_obj;

             if( eu->th_id == 0 ) {
                 sched_obj->system_queue = (dague_dequeue_t*)malloc(sizeof(dague_dequeue_t));
                 dague_dequeue_construct( sched_obj->system_queue );
             } else {
                 sched_obj->system_queue = LOCAL_QUEUES_OBJECT(vp->execution_units[0])->system_queue;
             }
        }

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

            sched_obj->nb_hierarch_queues = vp->nb_cores;
            sched_obj->hierarch_queues = (dague_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );
            queue_size = vp->nb_cores * 4;

            /* Each thread creates its own "local" queue, connected to the shared dequeue */
            sched_obj->task_queue = dague_hbbuffer_new( queue_size, 1, push_in_queue_wrapper,
                                                        (void*)sched_obj->system_queue);
            sched_obj->task_queue->assoc_core_num = p * vp->nb_cores + t; // stored for instrumentation
            sched_obj->hierarch_queues[0] = sched_obj->task_queue;
        }

        for(t = 0; t < vp->nb_cores; t++) {
            nq = 1;
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

#if defined(HAVE_HWLOC)
            hwloc_levels = dague_hwloc_nb_levels();
#else
            hwloc_levels = -1;
#endif

            /* Handle the case when HWLOC is present but cannot compute the hierarchy, 
             * as well as the casewhen HWLOC is not present
             */
            if( hwloc_levels == -1 ) {
                for( ; nq < sched_obj->nb_hierarch_queues; nq++ ) {
                    sched_obj->hierarch_queues[nq] =
                        LOCAL_QUEUES_OBJECT(vp->execution_units[(eu->th_id + nq) % vp->nb_cores])->task_queue;
                }
            } else {
#if defined(HAVE_HWLOC)
                /* Then, they know about all other queues, from the closest to the farthest */
                for(int level = 0; level <= hwloc_levels; level++) {
                    for(int id = (eu->th_id + 1) % vp->nb_cores; 
                        id != eu->th_id; 
                        id = (id + 1) %  vp->nb_cores) {
                        int d;
                        d = dague_hwloc_distance(eu->th_id, id);
                        if( d == 2*level || d == 2*level + 1 ) {
                            sched_obj->hierarch_queues[nq] = LOCAL_QUEUES_OBJECT(vp->execution_units[id])->task_queue;
                            DEBUG(("%d of %d: my %d preferred queue is the task queue of %d (%p)\n",
                                   eu->th_id, eu->virtual_process->vp_id, nq, id, sched_obj->hierarch_queues[nq]));
                            nq++;
                            if( nq == sched_obj->nb_hierarch_queues )
                                break;
                        }
                    }
                    if( nq == sched_obj->nb_hierarch_queues )
                        break;
                }
                assert( nq == sched_obj->nb_hierarch_queues );
#else
                /* Unreachable code */
#endif
            }
        }
    }

    return 0;
}

// TREE
static dague_execution_context_t * choose_job_tree_queues( dague_execution_unit_t *eu_context )
{
    dague_heap_t* heap = NULL;
    dague_heap_t* new_heap = NULL;
    dague_execution_context_t * exec_context = NULL;
    int i = 0;

    /*
     possible future improvement over using existing pop_best function:
     instead, i need to iterate manually over the buffer
     and choose a tree that has the highest value
     then take that task from that tree.
     */
    heap = (dague_heap_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->task_queue, dague_heap_priority_comparator);
    exec_context = heap_split_and_steal(&heap, &new_heap);
    if( NULL != heap )
        dague_hbbuffer_push_all(LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)heap);
    if (exec_context != NULL) {
	    PARSEC_INSTRUMENT(SCHED_STEAL, eu_context, exec_context, (void *)LOCAL_QUEUES_OBJECT(eu_context)->task_queue->assoc_core_num);
	    return exec_context;
    }

    // if we failed to find one in our queue
    for(i = 1; i <  LOCAL_QUEUES_OBJECT(eu_context)->nb_hierarch_queues; i++ ) {
        heap = (dague_heap_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i], dague_heap_priority_comparator);
        exec_context = heap_split_and_steal(&heap, &new_heap);
        if( NULL != heap ) {
            if (NULL != new_heap) {
                /* turn two-element doubly-linked list
                 (the default side effect of heap_split_and_steal)
                 into two singleton doubly-linked lists
                 */
                heap->list_item.list_next = (dague_list_item_t*)heap;
                heap->list_item.list_prev = (dague_list_item_t*)heap;
                new_heap->list_item.list_next = (dague_list_item_t*)new_heap;
                new_heap->list_item.list_prev = (dague_list_item_t*)new_heap;

                // put new heap back in neighboring queue
                dague_hbbuffer_push_all(LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i], (dague_list_item_t*)new_heap);
            }
            // put old heap in our queue -- it's a singleton either way by this point
            dague_hbbuffer_push_all(LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)heap);
        }
        if (exec_context != NULL) {
	        PARSEC_INSTRUMENT(SCHED_STEAL, eu_context, exec_context, (void *)LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]->assoc_core_num);
            return exec_context;
        }
    }

    // if nothing yet, then go to system queue
    heap = (dague_heap_t *)dague_dequeue_pop_front(LOCAL_QUEUES_OBJECT(eu_context)->system_queue);
    exec_context = heap_split_and_steal(&heap, &new_heap);
    if (heap != NULL)
        dague_hbbuffer_push_all(LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)heap);

    PARSEC_INSTRUMENT(SCHED_STEAL, eu_context, exec_context, SYSTEM_NEIGHBOR);
    return exec_context;
}

// TREE
/*
 * this function is basically thread-safe, in the sense that it does not affect
 * anything globally until it performs the dague_hbbuffer_push_all
 */
static int schedule_tree_queues( dague_execution_unit_t* eu_context,
                                 dague_execution_context_t* new_context )
{
    dague_execution_context_t * cur = new_context;
    dague_execution_context_t * next;
    dague_heap_t* heap = heap_create();
    dague_heap_t* first_h = heap;
    int matches = 0;
    int i, j;

    // do data_lookup
    if (cur->function->data_lookup(cur) != DAGUE_LOOKUP_DONE)
	    assert(0);

    while (1) {
        // check next element before insertion, which destroys next and prev
        next = (dague_execution_context_t*)cur->list_item.list_next;
        assert(next != NULL);

        heap_insert(heap, cur);

        if (next == cur /* looped */ || next == new_context /* looped */) {
	        break; // we're done
        }

        // compare data.... if we have at least one similar data item, then group
        if (next->function->data_lookup(next) != DAGUE_LOOKUP_DONE)
	        assert(0);
        matches = 0;
        for (i = 0; i < MAX_PARAM_COUNT; i++) {
	        for (j = 0; j < MAX_PARAM_COUNT; j++) {
		        if (cur->data[i].data != NULL && cur->data[i].data == next->data[j].data)
			        matches++;
	        }
        }

        cur = next;
        
        if (!matches) {
            // make new heap
            dague_heap_t * new_heap = heap_create();
            heap->list_item.list_next->list_prev = (dague_list_item_t*)new_heap;
            new_heap->list_item.list_prev = (dague_list_item_t*)heap;
            new_heap->list_item.list_next = (dague_list_item_t*)heap->list_item.list_next;
            heap->list_item.list_next = (dague_list_item_t*)new_heap;
            heap = new_heap;
        }
    }

    dague_hbbuffer_push_all( LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)first_h );

    return 0;
}

static void finalize_tree_queues( dague_context_t *master )
{
    int t, p;
    dague_execution_unit_t *eu;
    dague_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

            if( eu->th_id == 0 ) {
                dague_dequeue_destruct( sched_obj->system_queue );
                free( sched_obj->system_queue );
            }
            sched_obj->system_queue = NULL;

            dague_hbbuffer_destruct( sched_obj->task_queue );
            sched_obj->task_queue = NULL;

            free(sched_obj->hierarch_queues);
            sched_obj->hierarch_queues = NULL;

            free(eu->scheduler_object);
            eu->scheduler_object = NULL;
        }
    }
}


dague_scheduler_t sched_local_tree_queues = {
    .name = "Local Tree Queues",
    .init = init_tree_queues,
    .schedule_task = schedule_tree_queues,
    .select_task = choose_job_tree_queues,
    .display_stats = NULL,
    .finalize = finalize_tree_queues
};

static int no_scheduler_is_active( dague_context_t *master )
{
    int p, t;
    dague_vp_t *vp;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            if( vp->execution_units[t]->scheduler_object != NULL ) {
                return 0;
            }
        }
    }

    return 1;
}

/*********************************************************************/
/************************ Global Dequeue *****************************/
/*********************************************************************/

static int init_global_dequeue( dague_context_t *master )
{
    int p, t;
    dague_vp_t *vp;
    dague_dequeue_t *q;

    if( no_scheduler_is_active( master ) ) {

        for(p = 0; p < master->nb_vp; p++) {
            q = malloc(sizeof(dague_dequeue_t));
            dague_dequeue_construct( q );

            vp = master->virtual_processes[p];
            for(t = 0; t < vp->nb_cores; t++) {
                vp->execution_units[t]->scheduler_object = (void*)q;
            }

        }

        return 0;
    } else {
        return -1;
    }
}

static dague_execution_context_t *choose_job_global_dequeue( dague_execution_unit_t *eu_context )
{
    return (dague_execution_context_t*)dague_dequeue_try_pop_front( (dague_dequeue_t*)eu_context->scheduler_object );
}

static int schedule_global_dequeue( dague_execution_unit_t* eu_context,
                                    dague_execution_context_t* new_context )
{
    if( new_context->function->flags & DAGUE_HIGH_PRIORITY_TASK ) {
        dague_dequeue_chain_front( (dague_dequeue_t*)eu_context->scheduler_object, (dague_list_item_t*)new_context);
    } else {
        dague_dequeue_chain_back( (dague_dequeue_t*)eu_context->scheduler_object, (dague_list_item_t*)new_context);
    }
    return 0;
}

static void finalize_global_dequeue( dague_context_t *master )
{
    int p, t;
    dague_vp_t *vp;
    dague_execution_unit_t *eu;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            if( eu->th_id == 0 ) {
                dague_dequeue_destruct( (dague_dequeue_t*)eu->scheduler_object );
                free(eu->scheduler_object);
            }
            eu->scheduler_object = NULL;

        }
    }
}

dague_scheduler_t sched_global_dequeue = {
    .name = "Global Dequeues",
    .init = init_global_dequeue,
    .schedule_task = schedule_global_dequeue,
    .select_task = choose_job_global_dequeue,
    .display_stats = NULL,
    .finalize = finalize_global_dequeue
};


/*********************************************************************/
/****************** Local Queues (flat & hier) ***********************/
/****************** and Priority-Based Queues  ***********************/
/*********************************************************************/

static int init_local_flat_queues(  dague_context_t *master )
{
    int p, t, nq = 1;
    dague_execution_unit_t *eu;
    dague_vp_t *vp;
    uint32_t queue_size;
    local_queues_scheduler_object_t *sched_obj = NULL;
    int hwloc_levels;

    if( !no_scheduler_is_active( master ) ) {
        return -1;
    }

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        SYSTEM_NEIGHBOR = master->nb_vp * vp->nb_cores; // defined for instrumentation

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            queue_size = 4 * vp->nb_cores;

            sched_obj = (local_queues_scheduler_object_t*)malloc(sizeof(local_queues_scheduler_object_t));
            eu->scheduler_object = sched_obj;

            sched_obj->choice_count = 0;
            sched_obj->neighbor_stats = calloc(master->nb_vp * vp->nb_cores + 1, sizeof(neighbor_statistic)); // use calloc to set all counts to zero
    
            if( eu->th_id == 0 ) {
                sched_obj->system_queue = (dague_dequeue_t*)malloc(sizeof(dague_dequeue_t));
                dague_dequeue_construct( sched_obj->system_queue );
            } else {
                sched_obj->system_queue = LOCAL_QUEUES_OBJECT(vp->execution_units[0])->system_queue;
            }
        }

        DEBUG(("VP %d: creating queues for %d cores\n",
               p, vp->nb_cores));

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

            sched_obj->nb_hierarch_queues = vp->nb_cores;    
            sched_obj->hierarch_queues = (dague_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );

            /* Each thread creates its own "local" queue, connected to the shared dequeue */
            sched_obj->task_queue = dague_hbbuffer_new( queue_size, 1, push_in_queue_wrapper, 
                                                        (void*)sched_obj->system_queue);
            sched_obj->task_queue->assoc_core_num = p * vp->nb_cores + t; // stored for instrumentation
            sched_obj->hierarch_queues[0] = sched_obj->task_queue;
            DEBUG((" Core %d:%d: Task queue is %p (that's 0-preferred queue)\n",  p, t, sched_obj->task_queue));
        }

        for(t = 0; t < vp->nb_cores; t++) {
            nq = 1;
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);
 
#if defined(HAVE_HWLOC)
            hwloc_levels = dague_hwloc_nb_levels();
#else 
            hwloc_levels = -1;
#endif

            /* Handle the case when HWLOC is present but cannot compute the hierarchy, 
             * as well as the casewhen HWLOC is not present
             */
            if( hwloc_levels == -1 ) {
                for( ; nq < sched_obj->nb_hierarch_queues; nq++ ) {
                    sched_obj->hierarch_queues[nq] =
                        LOCAL_QUEUES_OBJECT(vp->execution_units[(eu->th_id + nq) % vp->nb_cores])->task_queue;
                }
            } else {
#if defined(HAVE_HWLOC)
                /* Then, they know about all other queues, from the closest to the farthest */
                for(int level = 0; level <= hwloc_levels; level++) {
                    for(int id = (eu->th_id + 1) % vp->nb_cores; 
                        id != eu->th_id; 
                        id = (id + 1) %  vp->nb_cores) {
                        int d;
                        d = dague_hwloc_distance(eu->th_id, id);
                        if( d == 2*level || d == 2*level + 1 ) {
                            sched_obj->hierarch_queues[nq] = LOCAL_QUEUES_OBJECT(vp->execution_units[id])->task_queue;
                            DEBUG(("%d of %d: my %d-preferred queue is the task queue of %d (%p)\n",
                                   eu->th_id, eu->virtual_process->vp_id, nq, id, sched_obj->hierarch_queues[nq]));
                            nq++;
                            if( nq == sched_obj->nb_hierarch_queues )
                                break;
                        }
                    }
                    if( nq == sched_obj->nb_hierarch_queues )
                        break;
                }
                assert( nq == sched_obj->nb_hierarch_queues );
#else
                /* Unreachable code */
#endif
            }
        }
    }   
    return 0;
}


static int init_local_hier_queues( dague_context_t *master )
{
#if !defined(HAVE_HWLOC)
    (void)master;
    ERROR(("hierarchical scheduler cannot be selected, you need to recompile DAGuE with hwloc, or select another scheduler.\n"));
    return -1;
#else
    int p, t;
    dague_execution_unit_t *eu;
    dague_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj = NULL;

    if( !no_scheduler_is_active( master ) ) {
        return -1;
    }

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];

            sched_obj = (local_queues_scheduler_object_t*)malloc(sizeof(local_queues_scheduler_object_t));
            eu->scheduler_object = sched_obj;

            if( eu->th_id == 0 ) {
                sched_obj->system_queue = (dague_dequeue_t*)malloc(sizeof(dague_dequeue_t));
                dague_list_construct( sched_obj->system_queue );
            } else {
                sched_obj->system_queue = LOCAL_QUEUES_OBJECT(vp->execution_units[0])->system_queue;
            }

            sched_obj->nb_hierarch_queues = vp->nb_cores;
            sched_obj->hierarch_queues = (dague_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );

            sched_obj->nb_hierarch_queues = dague_hwloc_nb_levels();
            sched_obj->hierarch_queues = (dague_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );

            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = dague_hwloc_master_id(level, eu->th_id);
                if( eu->th_id == m ) {
                    int nbcores = dague_hwloc_nb_cores(level, m);
                    int queue_size = 96 * (level+1) / nbcores;
                    if( queue_size < nbcores ) queue_size = nbcores;


                    /* The master(s) create the shared queues */
                    sched_obj->hierarch_queues[idx] =
                        dague_hbbuffer_new( queue_size, nbcores,
                                            level == 0 ? push_in_queue_wrapper : push_in_buffer_wrapper,
                                            level == 0 ? (void*)sched_obj->system_queue : (void*)sched_obj->hierarch_queues[idx+1]);
                    DEBUG3(("schedHQ %d: \tcreates hbbuffer of size %d (ideal %d) for level %d stored in %d: %p (parent: %p -- %s)\n",
                            eu->th_id, queue_size, nbcores,
                            level, idx, sched_obj->hierarch_queues[idx],
                            level == 0 ? (void*)sched_obj->system_queue : (void*)sched_obj->hierarch_queues[idx+1],
                            level == 0 ? "System queue" : "upper level hhbuffer"));
                }

            }
        }

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = (local_queues_scheduler_object_t*)eu->scheduler_object;
            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = dague_hwloc_master_id(level, eu->th_id);
                if( eu->th_id != m ) {
                    DEBUG3(("schedHQ %d: \ttakes the buffer of %d at level %d stored in %d: %p\n",
                            eu->th_id, m, level, idx, LOCAL_QUEUES_OBJECT(vp->execution_units[m])->hierarch_queues[idx]));
                    /* The slaves take their queue for this level from their master */
                    sched_obj->hierarch_queues[idx] = LOCAL_QUEUES_OBJECT(vp->execution_units[m])->hierarch_queues[idx];
                }

            }
            sched_obj->task_queue = sched_obj->hierarch_queues[0];
        }
    }

    return 0;
#endif
}

static dague_execution_context_t *choose_job_local_queues( dague_execution_unit_t *eu_context )
{
    dague_execution_context_t *exec_context = NULL;
    int i;

    /* LOCAL_QUEUES_OBJECT(eu_context)->choice_count++; */

    /* LOCAL_QUEUES_OBJECT(eu_context)->neighbor_stats[LOCAL_QUEUES_OBJECT(eu_context)->task_queue->assoc_core_num].try_count++; */
    /* exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->task_queue, */
    /*                                                                    dague_execution_context_priority_comparator); */

    /* if (LOCAL_QUEUES_OBJECT(eu_context)->task_queue->assoc_core_num == LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[0]->assoc_core_num) */
	/*     printf("this is just strange...\n"); */

    /* if( NULL != exec_context ) { */
	/*     printf("stole a first order\n"); */
    /*     LOCAL_QUEUES_OBJECT(eu_context)->neighbor_stats[LOCAL_QUEUES_OBJECT(eu_context)->task_queue->assoc_core_num].steal_count++; */
    /*     return exec_context; */
    /* } */

    for(i = 0; i <  LOCAL_QUEUES_OBJECT(eu_context)->nb_hierarch_queues; i++ ) {
        exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i],
                                                                           dague_execution_context_priority_comparator);
        LOCAL_QUEUES_OBJECT(eu_context)->neighbor_stats[LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]->assoc_core_num].try_count++;
        if( NULL != exec_context ) {
	        LOCAL_QUEUES_OBJECT(eu_context)->neighbor_stats[LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]->assoc_core_num].steal_count++; // increment steal count
            DEBUG3(("LQ\t: %d:%d found task %p in its %d-preferred hierarchical queue %p\n",
                    eu_context->virtual_process->vp_id, eu_context->th_id, exec_context, i, LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]));
            return exec_context;
        }
    }

    exec_context = (dague_execution_context_t *)dague_dequeue_try_pop_front(LOCAL_QUEUES_OBJECT(eu_context)->system_queue);
    LOCAL_QUEUES_OBJECT(eu_context)->neighbor_stats[SYSTEM_NEIGHBOR].try_count++;
    if( NULL != exec_context ) {
        LOCAL_QUEUES_OBJECT(eu_context)->neighbor_stats[SYSTEM_NEIGHBOR].steal_count++; // increment steal count
        DEBUG3(("LQ\t: %d:%d found task %p in its system queue %p\n",
                eu_context->virtual_process->vp_id, eu_context->th_id, exec_context, LOCAL_QUEUES_OBJECT(eu_context)->system_queue));
    }
    return exec_context;
}

static int schedule_local_queues( dague_execution_unit_t* eu_context,
                                  dague_execution_context_t* new_context )
{
    dague_hbbuffer_push_all( LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)new_context );
#if defined(DAGUE_PROF_TRACE)
    TAKE_TIME(eu_context->eu_profile, queue_add_begin, 0);
    TAKE_TIME(eu_context->eu_profile, queue_add_end, 0);
#endif
    return 0;
}

static int schedule_local_queues_by_priority( dague_execution_unit_t* eu_context,
                                              dague_execution_context_t* new_context )
{
    dague_hbbuffer_push_all_by_priority( LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)new_context);
    return 0;
}

static void finalize_local_hier_queues( dague_context_t *master )
{
#if !defined(HAVE_HWLOC)
    (void)master;
    return;
#else
    int p, t;
    dague_execution_unit_t *eu;
    dague_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);
    
            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = dague_hwloc_master_id(level, eu->th_id);
                if( eu->th_id == m ) {
                    dague_hbbuffer_destruct(sched_obj->hierarch_queues[idx]);
                    sched_obj->hierarch_queues[idx] = NULL;
                } else {
                    sched_obj->hierarch_queues[idx] = NULL;
                }
            }

            sched_obj->task_queue = NULL;
        
            free(eu->scheduler_object);
            eu->scheduler_object = NULL;
        }
    }
#endif
}

static void finalize_local_flat_queues( dague_context_t *master )
{
    int p, t;
    dague_execution_unit_t *eu;
    dague_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

            int u = 0;
            printf("    ");
            for (u = 0; u < master->nb_vp * vp->nb_cores; u++) {
	            printf("%4u ", sched_obj->neighbor_stats[u].steal_count);
            }
            printf("%4u\n", sched_obj->neighbor_stats[SYSTEM_NEIGHBOR].steal_count);

            /* int u = 0; */
            /* printf("    exec_unit %p, core #%u\n", eu, sched_obj->task_queue->assoc_core_num); */
            /* for (u = 0; u < sched_obj->nb_hierarch_queues; u++) { */
	        /*     printf("    steals/attempts from core %3d: %4u / %4u\n", u, sched_obj->neighbor_stats[u].steal_count, */
	        /*            sched_obj->neighbor_stats[u].try_count); */
            /* } */
            /* printf("    steals/attempts from SYSTEM QUEUE: %4u / %4u\n", sched_obj->neighbor_stats[SYSTEM_NEIGHBOR].steal_count,  */
            /*        sched_obj->neighbor_stats[SYSTEM_NEIGHBOR].try_count); */
            /* printf("    starvation:       %4u\n\n", sched_obj->neighbor_stats[SYSTEM_NEIGHBOR].try_count - sched_obj->neighbor_stats[SYSTEM_NEIGHBOR].steal_count); */

            free(sched_obj->neighbor_stats);
            sched_obj->neighbor_stats = NULL;

            if( eu->th_id == 0 ) {
                dague_dequeue_destruct( sched_obj->system_queue );
                free( sched_obj->system_queue );
            }
            sched_obj->system_queue = NULL;

            dague_hbbuffer_destruct( sched_obj->task_queue );
            sched_obj->task_queue = NULL;

            free(sched_obj->hierarch_queues);
            sched_obj->hierarch_queues = NULL;

            free(eu->scheduler_object);
            eu->scheduler_object = NULL;
        }
    }
}

dague_scheduler_t sched_local_flat_queues = {
    .name = "Local Flat Queues",
    .init = init_local_flat_queues,
    .schedule_task = schedule_local_queues,
    .select_task = choose_job_local_queues,
    .display_stats = NULL,
    .finalize = finalize_local_flat_queues
};

dague_scheduler_t sched_local_hier_queues = {
    .name = "Local Hierarchical Queues",
    .init = init_local_hier_queues,
    .schedule_task = schedule_local_queues,
    .select_task = choose_job_local_queues,
    .display_stats = NULL,
    .finalize = finalize_local_hier_queues
};

dague_scheduler_t sched_priority_based_queues = {
    .name = "Priority Based Queues",
    .init = init_local_flat_queues,
    .schedule_task = schedule_local_queues_by_priority,
    .select_task = choose_job_local_queues,
    .display_stats = NULL,
    .finalize = finalize_local_flat_queues
};

/*********************************************************************/
/********************* Absolute Priorities ***************************/
/*********************************************************************/

static dague_execution_context_t *choose_job_absolute_priorities( dague_execution_unit_t *eu_context )
{
    return (dague_execution_context_t*)dague_list_pop_front((dague_list_t*)eu_context->scheduler_object);
}

static int schedule_absolute_priorities( dague_execution_unit_t* eu_context,
                                         dague_execution_context_t* new_context )
{
#if defined(DAGUE_DEBUG_VERBOSE3)
    dague_list_item_t *it = (dague_list_item_t*)new_context;
    char tmp[MAX_TASK_STRLEN];
    do {
        DEBUG3(("AP:\t Pushing task %s\n", 
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, (dague_execution_context_t*)it)));
        it = (dague_list_item_t*)((dague_list_item_t*)it)->list_next;
    } while( it != (dague_list_item_t*)new_context );
#endif
    dague_list_chain_sorted((dague_list_t*)eu_context->scheduler_object,
                            (dague_list_item_t*)new_context,
                            dague_execution_context_priority_comparator);
    return 0;
}

static int init_absolute_priorities( dague_context_t *master )
{
    int p, t;
    dague_vp_t *vp;
    dague_execution_unit_t *eu;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            if( eu->th_id == 0 ) {
                eu->scheduler_object = (dague_list_t*)malloc(sizeof(dague_list_t));
                dague_list_construct( (dague_list_t*)eu->scheduler_object );
            } else {
                eu->scheduler_object = eu->virtual_process->execution_units[0]->scheduler_object;
            }
        }
    }
        
    return 0;
}

static void finalize_absolute_priorities( dague_context_t *master )
{
    int p, t;
    dague_vp_t *vp;
    dague_execution_unit_t *eu;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            if( eu->th_id == 0 ) {
                 dague_list_destruct( (dague_list_t*)eu->scheduler_object );
                free(eu->scheduler_object);
            }
            eu->scheduler_object = NULL;
        }
    }
}

dague_scheduler_t sched_absolute_priorities = {
    .name = "Absolute priorities",
    .init = init_absolute_priorities,
    .schedule_task = schedule_absolute_priorities,
    .select_task = choose_job_absolute_priorities,
    .display_stats = NULL,
    .finalize = finalize_absolute_priorities
};

dague_scheduler_t *dague_schedulers_array[NB_DAGUE_SCHEDULERS] =
{
	&sched_local_flat_queues,
	&sched_global_dequeue,
	&sched_local_hier_queues,
	&sched_absolute_priorities,
	&sched_priority_based_queues,
	&sched_local_tree_queues
};
