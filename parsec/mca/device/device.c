/*
 *
 * Copyright (c) 2013-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/device/device.h"
#include "parsec/utils/mca_param.h"
#include "parsec/mca/mca_repository.h"
#include "parsec/constants.h"
#include "parsec/utils/debug.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/argv.h"
#include "parsec/parsec_internal.h"

#include <stdlib.h>
#if defined(PARSEC_HAVE_ERRNO_H)
#include <errno.h>
#endif  /* PARSEC_HAVE_ERRNO_H */
#if defined(PARSEC_HAVE_DLFCN_H)
#include <dlfcn.h>
#endif  /* PARSEC_HAVE_DLFCN_H */
#include <sys/stat.h>
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include <math.h>
#if defined(__WINDOWS__)
#include <windows.h>
#endif  /* defined(__WINDOWS__) */
int parsec_device_output = 0;
static int parsec_device_verbose = 0;
uint32_t parsec_nb_devices = 0;
static uint32_t parsec_nb_max_devices = 0;
static uint32_t parsec_mca_device_are_freezed = 0;
parsec_atomic_lock_t parsec_devices_mutex = PARSEC_ATOMIC_UNLOCKED;
static parsec_device_module_t** parsec_devices = NULL;

static parsec_device_module_t* parsec_device_cpus = NULL;
static parsec_device_module_t* parsec_device_recursive = NULL;

static int num_modules_activated = 0;
static parsec_device_module_t **modules_activated = NULL;

static mca_base_component_t **device_components = NULL;

/**
 * Load balance skew we are willing to accept to favor RO data reuse
 * on GPU: a value of 20% means that we will schedule tasks on the preferred
 * GPU except if it is loaded 1.2 times as much as the best load balance option
 */
static int parsec_device_load_balance_skew = 20;
static float load_balance_skew;

/**
 * @brief Estimates how many nanoseconds this_task will run on devid
 *
 * @param this_task the task to run
 * @param dev the device that might run @p this_task
 * @return uint64_t an estimate of the number of nanoseconds @p this_task
 *   might run on the device @p dev
 */
static int64_t time_estimate(const parsec_task_t *this_task, parsec_device_module_t *dev)
{
    if( NULL != this_task->task_class->time_estimate ) {
        return this_task->task_class->time_estimate(this_task, dev);
    }
    /* No estimate given. we just return an arbitrary number based on the
     * double-precision floprate of the device: the weaker the device (w.r.t.
     * other available devices), the higher this number. */
    return dev->time_estimate_default;
}

/**
 * Find the best device to execute the kernel based on the compute
 * capability of the device.
 *
 * Returns:
 * PARSEC_SUCCESS - kernel must be executed by the device set in
 *                  this_task->selected_device (for convenience
 *                  this_task->selected_chore is also set)
 *                  this_task->load is set based on the selected device
 * PARSEC_ERROR   - no device could be selected
 */
int parsec_select_best_device( parsec_task_t* this_task ) {
    parsec_data_copy_t* data_copy = NULL;
    parsec_taskpool_t* tp = this_task->taskpool;
    parsec_device_module_t *dev = NULL, *rdata_dev = NULL;

    const parsec_task_class_t* tc = this_task->task_class;
    parsec_evaluate_function_t* eval;
    int rc;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task);
#endif
    unsigned int chore_id = 0, valid_types = 0;

    /* we did it before (this is a PARSEC_RETURN_AGAIN/ASYNC?) */
    if( this_task->selected_device ) {
        PARSEC_DEBUG_VERBOSE(30, parsec_device_output, "%s: Task %s a-priori selected_device %d:%s",
                             __func__, tmp, this_task->selected_device->device_index, this_task->selected_device->name);
        goto device_selected;
    }

    /* Run the evaluates for the incarnation types to determine if they can
     * execute this task */
    for(chore_id = 0; PARSEC_DEV_NONE != tc->incarnations[chore_id].type; chore_id++) {
        if( 0 == (this_task->chore_mask & (1<<chore_id)) ) continue;
        if( NULL == tc->incarnations[chore_id].hook ) continue; /* dyld hook not found during initialization */

        if( NULL != (eval = tc->incarnations[chore_id].evaluate) ) {
            rc = eval(this_task);
            if( PARSEC_HOOK_RETURN_DONE != rc ) {
                if( PARSEC_HOOK_RETURN_NEXT != rc ) {
                    PARSEC_DEBUG_VERBOSE(5, parsec_device_output, "Failed to evaluate %s[%d] chore %d",
                                         tmp, tc->incarnations[chore_id].type,
                                         chore_id);
                }
                /* Mark this chore as tested */
                this_task->chore_mask &= ~( 1<<chore_id );
                continue;
            }
        }
        valid_types |= tc->incarnations[chore_id].type; /* the eval accepted the type, but no device specified yet */
        /* Evaluate may have picked a device, abide by it */
        if( NULL != this_task->selected_device ) {
            assert( this_task->selected_device->type & valid_types );
            PARSEC_DEBUG_VERBOSE(30, parsec_device_output, "%s: Task %s evaluate set selected_device %d:%s",
                                 __func__, tmp, this_task->selected_device->device_index, this_task->selected_device->name);
            goto device_selected;
        }
    }

    if (!valid_types)
        goto no_valid_device;

    if (PARSEC_DEV_CPU == valid_types) { /* shortcut for CPU only tasks */
        this_task->selected_device = dev = parsec_mca_device_get(0);
        this_task->load = 0;
        for(chore_id = 0; tc->incarnations[chore_id].type != PARSEC_DEV_CPU; chore_id++);
        this_task->selected_chore = chore_id;
        PARSEC_DEBUG_VERBOSE(80, parsec_device_output, "%s: Task %s cpu-only task set selected_device %d:%s",
                             __func__, tmp, dev->device_index, dev->name);
        return PARSEC_SUCCESS;
    }

    /* For all devices with matching incarnation type for the chore_id, which one is the best?
     * We try to minimize the data movements, so favor devices that already
     * hold the data:
     *   Select the location of the first data that is used with ACCESS_WRITE (first loop)
     *   or prefer the location of one of the READ data (second loop).
     */
    for( int i = 0; i < this_task->task_class->nb_flows; i++ ) { /* look for an ACCESS_WRITE data */
        if( NULL == this_task->task_class->out[i]
         || !(PARSEC_FLOW_ACCESS_WRITE & this_task->task_class->out[i]->flow_flags) ) continue; /* not ACCESS_WRITE, skip */

        data_copy = this_task->data[this_task->task_class->out[i]->flow_index].data_in;
        if( NULL == data_copy ) continue; /* possible for NULL flows */

        /* If the WRITE data has a preferred device, select it. */
        dev = parsec_mca_device_get(data_copy->original->preferred_device);
        if( NULL != dev && (dev->type & valid_types) && (tp->devices_index_mask & (1<<dev->device_index))) {
            PARSEC_DEBUG_VERBOSE(30, parsec_device_output, "%s: Task %s set selected_device %d:%s because preferred by data %p AWRITE flow out[%d]",
                                 __func__, tmp, dev->device_index, dev->name, data_copy->original, i);
            this_task->selected_device = dev;
            goto device_selected;
        }
        /* WRITE data is already located on a GPU device, select it */
        dev = parsec_mca_device_get(data_copy->original->owner_device);
        if( NULL != dev && (dev->type & valid_types) && (tp->devices_index_mask & (1<<dev->device_index)) && PARSEC_DEV_IS_GPU(dev->type) ) {
            PARSEC_DEBUG_VERBOSE(30, parsec_device_output, "%s: Task %s set selected_device %d:%s because owner of data %p flow AWRITE flow out[%d]",
                                 __func__, tmp, dev->device_index, dev->name, data_copy->original, i);
            this_task->selected_device = dev;
            goto device_selected;
        }
    }
    /* If we reach here, we cannot yet decide which device to run on based on the task outputs */
    for( int i = 0; i < this_task->task_class->nb_flows; i++ ) { /* look for an ACCESS_READ data */
        if( NULL == this_task->task_class->in[i] ) continue; /* possible for WRITE (write-only) flows */

        data_copy = this_task->data[this_task->task_class->in[i]->flow_index].data_in;
        if( NULL == data_copy ) continue; /* possible for NULL flows */

        /* If the READ data has a preferred device, prefer it. */
        dev = parsec_mca_device_get(data_copy->original->preferred_device);
        if( NULL != dev && (dev->type & valid_types) && (tp->devices_index_mask & (1<<dev->device_index)) ) {
            PARSEC_DEBUG_VERBOSE(30, parsec_device_output, "%s: Task %s set selected_device %d:%s because preferred by data %p flow AREAD in[%d]",
                                 __func__, tmp, dev->device_index, dev->name, data_copy->original, i);
            this_task->selected_device = dev;
            goto device_selected;
        }
        /* READ data is already located on a GPU device, prefer it */
        dev = parsec_mca_device_get(data_copy->original->owner_device);
        if( NULL != dev && (dev->type & valid_types) && (tp->devices_index_mask & (1<<dev->device_index)) && PARSEC_DEV_IS_GPU(dev->type) /* no CPU or recursive, always true test would disable GPU execution */ ) {
            PARSEC_DEBUG_VERBOSE(30, parsec_device_output, "%s: Task %s favors device %d:%s because owner of data %p flow AREAD in[%d]",
                                 __func__, tmp, dev->device_index, dev->name, data_copy->original, i);
            rdata_dev = dev;
            break;
        }
    }

    assert( NULL == this_task->selected_device );
    { /* lets consider the time_estimates to select the best device */
        int best_index = -1;
        int64_t eta, best_eta = INT64_MAX; /* dev->device_load + time_estimate(this_task, dev); this commented out because we don't count cpu loads */

        /* If we have a preferred device (from READ flows), start with it, but still consider
         * other options to have some load balance */
        if( NULL != rdata_dev ) {
            best_index = rdata_dev->device_index;
            best_eta = rdata_dev->device_load + time_estimate(this_task, rdata_dev);
            /* we still prefer this device, until it is load_balance_skew as loaded as the
             * real best eta device, lets scale the best_eta accordingly. */
            best_eta *= load_balance_skew;
        }

        /* Consider how adding the current task would change load balancing
         * between devices */
        for( int dev_index = 0; dev_index < parsec_mca_device_enabled(); dev_index++ ) {
            /* Skip the device if it is disabled for the taskpool */
            if(!(tp->devices_index_mask & (1 << dev_index))) continue;
            dev = parsec_mca_device_get(dev_index);
            /* Skip the device if no incarnations for its type */
            if(!(dev->type & valid_types)) continue;
            /* Skip recursive devices: time estimates are computed on the associated CPU device */
            if(dev->type == PARSEC_DEV_RECURSIVE) continue;

            eta = dev->device_load + time_estimate(this_task, dev);
            if( best_eta > eta ) {
                if(best_index == -1)
                    PARSEC_DEBUG_VERBOSE(30, parsec_device_output, "%s: Task %s has eta %"PRIi64" on %d:%s (first pick)",
                                         __func__, tmp, eta, dev_index, dev->name);
                else
                    PARSEC_DEBUG_VERBOSE(30, parsec_device_output, "%s: Task %s has eta %"PRIi64" on %d:%s (better than eta %"PRIi64" on device index %d)",
                                         __func__, tmp, eta, dev_index, dev->name, best_eta, best_index);
                best_index = dev_index;
                best_eta = eta;
            }
        }
        if( -1 == best_index ) /* taskpool disabled all valid_types devices */
            goto no_valid_device;

        this_task->selected_device = parsec_mca_device_get(best_index);
        assert( this_task->selected_device->type != PARSEC_DEV_RECURSIVE );
    }

device_selected:
    dev = this_task->selected_device;
    assert( NULL != dev );
    assert( tp->devices_index_mask & (1 << dev->device_index) );
    for(chore_id = 0; tc->incarnations[chore_id].type != dev->type; chore_id++)
        assert(PARSEC_DEV_NONE != tc->incarnations[chore_id].type /* we have selected this device, so there *must* be an incarnation that matches */);
    this_task->selected_chore = chore_id;
    this_task->load = time_estimate(this_task, dev);

    PARSEC_DEBUG_VERBOSE(20, parsec_device_output, "%s: Task %s set selected_device %d:%s (eta %"PRIi64")",
                         __func__, tmp, dev->device_index, dev->name, dev->device_load+this_task->load);

#if defined(PARSEC_DEBUG_PARANOID)
    /* Sanity check: if at least one of the data copies is not parsec
     * managed, check that all the non-parsec-managed data copies
     * exist on the same device */
    for( int i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if (NULL == this_task->data[i].data_in) continue;
        if ((this_task->data[i].data_in->flags & PARSEC_DATA_FLAG_PARSEC_MANAGED) == 0 &&
            this_task->data[i].data_in->device_index != dev->device_index) {
            char task_str[MAX_TASK_STRLEN];
            parsec_fatal("*** User-Managed Copy Error: Task %s is selected to run on device %d:%s,\n"
                         "*** but flow %d is represented by a data copy not managed by PaRSEC,\n"
                         "*** and does not have a copy on that device\n",
                         parsec_task_snprintf(task_str, MAX_TASK_STRLEN, this_task), dev->device_index, dev->name, i);
        }
    }
#endif
    return PARSEC_SUCCESS;

no_valid_device: {
#if !defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task);
#endif
    parsec_warning("Task %s ran out of valid incarnations. No device selected.",
                   tmp);
    return PARSEC_ERROR;
  }
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_device_module_t, parsec_object_t,
                          NULL, NULL);

int parsec_mca_device_init(void)
{
    char** parsec_device_list = NULL;
    parsec_device_module_t **modules = NULL;
#if defined(PARSEC_PROF_TRACE)
    char modules_activated_str[1024] = "";
#endif  /* defined(PARSEC_PROF_TRACE) */
    int i, j, rc, priority;

    PARSEC_OBJ_CONSTRUCT(&parsec_per_device_infos, parsec_info_t);
    PARSEC_OBJ_CONSTRUCT(&parsec_per_stream_infos, parsec_info_t);

    (void)parsec_mca_param_reg_int_name("device", "show_capabilities",
                                        "Show the detailed devices capabilities",
                                        false, false, parsec_debug_verbose >= 4 || (parsec_debug_verbose >= 3 && parsec_debug_rank == 0), NULL);
    (void)parsec_mca_param_reg_int_name("device", "show_statistics",
                                        "Show the detailed devices statistics upon exit",
                                        false, false, 0, NULL);
    (void)parsec_mca_param_reg_int_name("device", "load_balance_skew",
                                        "Allow load balancing to skew by x%% to favor data reuse",
                                        false, false, parsec_device_load_balance_skew, NULL);
    if( 0 < (rc = parsec_mca_param_find("device", NULL, "load_balance_skew")) ) {
        parsec_mca_param_lookup_int(rc, &parsec_device_load_balance_skew);
    }
    load_balance_skew = 1.f/(parsec_device_load_balance_skew/100.f+1.f);
    if( 0 < (rc = parsec_mca_param_find("device", NULL, "verbose")) ) {
        parsec_mca_param_lookup_int(rc, &parsec_device_verbose);
    }
    if( 0 < parsec_device_verbose ) {
        parsec_device_output = parsec_output_open(NULL);
        parsec_output_set_verbosity(parsec_device_output, parsec_device_verbose);
    }
    parsec_device_list = mca_components_get_user_selection("device");

    device_components = mca_components_open_bytype("device");
    for(i = 0; NULL != device_components[i]; i++); /* nothing just counting */
    if( 0 == i ) {  /* no devices */
        parsec_debug_verbose(10, parsec_device_output, "No devices found on %s\n", parsec_hostname);
        return PARSEC_ERR_NOT_FOUND;
    }
    modules_activated = (parsec_device_module_t**)malloc(sizeof(parsec_device_module_t*) * i);
    modules_activated[0] = NULL;

    for(i = j = 0; NULL != device_components[i]; i++) {
        if( mca_components_belongs_to_user_list(parsec_device_list, device_components[i]->mca_component_name) ) {
            if (device_components[i]->mca_query_component != NULL) {
                rc = device_components[i]->mca_query_component((mca_base_module_t**)&modules, &priority);
                if( MCA_SUCCESS != rc ) {
                    parsec_debug_verbose(10, parsec_device_output, "query function for component %s return no module",
                                         device_components[i]->mca_component_name);
                    device_components[i]->mca_close_component();
                    device_components[i] = NULL;
                    continue;
                }
                parsec_debug_verbose(10, parsec_device_output, "query function for component %s[%d] returns priority %d",
                                     device_components[i]->mca_component_name, i, priority);
                if( (NULL == modules) || (NULL == modules[0]) ) {
                    parsec_debug_verbose(10, parsec_device_output, "query function for component %s returns no modules. Remove.",
                                         device_components[i]->mca_component_name);
                    device_components[i]->mca_close_component();
                    device_components[i] = NULL;
                    continue;
                }
                if( i != j ) {  /* compress the list of components */
                    device_components[j] = device_components[i];
                    device_components[i] = NULL;
                    j++;
                }
                modules_activated[num_modules_activated++] = modules[0];

#if defined(PARSEC_PROF_TRACE)
                strncat(modules_activated_str, device_components[i]->mca_component_name, 1023);
                strncat(modules_activated_str, ",", 1023);
#endif
            }
        }
    }
    mca_components_free_user_list(parsec_device_list);
    parsec_debug_verbose(5, parsec_device_output, "Found %d components, activated %d", i, num_modules_activated);

#if defined(PARSEC_PROF_TRACE)
    /* replace trailing comma with \0 */
    if ( strlen(modules_activated_str) > 1) {
        if( modules_activated_str[ strlen(modules_activated_str) - 1 ] == ',' ) {
            modules_activated_str[strlen(modules_activated_str) - 1] = '\0';
        }
    }
    parsec_profiling_add_information("DEVICE_MODULES", modules_activated_str);
#endif

    return PARSEC_SUCCESS;
}

void parsec_compute_best_unit( uint64_t length, float* updated_value, char** best_unit )
{
    float measure = (float)length;

    *best_unit = "B";
    if( measure > 1024.0f ) { /* 1KB */
        *best_unit = "KB";
        measure = measure / 1024.0f;
        if( measure > 1024.0f ) { /* 1MB */
            *best_unit = "MB";
            measure = measure / 1024.0f;
            if( measure > 1024.0f ) {
                *best_unit = "GB";
                measure = measure / 1024.0f;
            }
        }
    }
    *updated_value = measure;
    return;
}

void parsec_devices_save_statistics(uint64_t **pstats) {
    if(NULL == *pstats) {
        *pstats = (uint64_t*)calloc(sizeof(uint64_t), parsec_nb_devices * 6 /* see below for the number of arrays */);
    }
    else {
        memset(*pstats, 0, parsec_nb_devices * sizeof(uint64_t) * 6);
    }
    uint64_t *stats = *pstats;
    uint64_t *executed_tasks = stats;
    uint64_t *transfer_in    = stats +   parsec_nb_devices;
    uint64_t *transfer_out   = stats + 2*parsec_nb_devices;
    uint64_t *req_in         = stats + 3*parsec_nb_devices;
    uint64_t *req_out        = stats + 4*parsec_nb_devices;
    uint64_t *transfer_d2d   = stats + 5*parsec_nb_devices;

    for(uint32_t i = 0; i < parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_devices[i];
        if(NULL == device) continue;
        assert( i == device->device_index );
        executed_tasks[i] = device->executed_tasks;
        transfer_in[i]    = device->data_in_from_device[0]; /* cpu-core device */
        transfer_out[i]   = device->data_out_to_host;
        req_in[i]         = device->required_data_in;
        req_out[i]        = device->required_data_out;
        for(unsigned int j = 1; j < device->data_in_array_size; j++) /* d2d */
            transfer_d2d[i] += device->data_in_from_device[j];
        /* don't compute global stats yet, do it during print */
    }
}

void parsec_devices_free_statistics(uint64_t **pstats) {
    assert(NULL != pstats && NULL != *pstats);
    free(*pstats);
    *pstats = NULL;
}

void parsec_devices_print_statistics(parsec_context_t *parsec_context, uint64_t *start_stats) {
    uint64_t *end_stats = NULL;
    uint64_t total_tasks = 0, total_data_in = 0, total_data_out = 0;
    uint64_t total_required_in = 0, total_required_out = 0, total_d2d = 0;
    float gtotal = 0.0;
    float best_data_in, best_data_out, best_d2d;
    float best_required_in, best_required_out;
    char *data_in_unit, *data_out_unit, *d2d_unit;
    char *required_in_unit, *required_out_unit;
    parsec_device_module_t *device;
    uint32_t i;

    /* initialize the arrays */
    parsec_devices_save_statistics(&end_stats);
    if(NULL != start_stats) {
        for(i = 0; i < parsec_nb_devices * 6; i++) {
            assert(end_stats[i] >= start_stats[i]);
            end_stats[i] -= start_stats[i];
        }
    }
    uint64_t *executed_tasks    = end_stats;
    uint64_t *transferred_in    = end_stats +   parsec_nb_devices;
    uint64_t *transferred_out   = end_stats + 2*parsec_nb_devices;
    uint64_t *required_in       = end_stats + 3*parsec_nb_devices;
    uint64_t *required_out      = end_stats + 4*parsec_nb_devices;
    uint64_t *transferred_d2d   = end_stats + 5*parsec_nb_devices;

    /* Compute total statistics */
    for(i = 0; i < parsec_nb_devices; i++) {
        if( NULL == (device = parsec_devices[i]) ) continue;
        assert( i == device->device_index );
        total_tasks        += executed_tasks[i];
        total_data_in      += transferred_in[i] + transferred_d2d[i];
        total_data_out     += transferred_out[i];
        total_required_in  += required_in[i];
        total_required_out += required_out[i];
        total_d2d          += transferred_d2d[i];
    }

    /* Print statistics */
    gtotal = (float)total_tasks;
    double percent_in, percent_out, percent_d2d;

    printf("+----------------------------------------------------------------------------------------------------------------------------+\n");
    printf("|         |                    |                       Data In                              |         Data Out               |\n");
    printf("|Rank %3d |  # KERNEL |    %%   |  Required  |   Transfered H2D(%%)   |   Transfered D2D(%%)   |  Required  |   Transfered(%%)   |\n",
           (NULL == parsec_context ? parsec_debug_rank : parsec_context->my_rank));
    printf("|---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|\n");
    for( i = 0; i < parsec_nb_devices; i++ ) {
        if( NULL == (device = parsec_devices[i]) ) continue;

        parsec_compute_best_unit( required_in[i],     &best_required_in,  &required_in_unit  );
        parsec_compute_best_unit( required_out[i],    &best_required_out, &required_out_unit );
        parsec_compute_best_unit( transferred_in[i],  &best_data_in,      &data_in_unit      );
        parsec_compute_best_unit( transferred_out[i], &best_data_out,     &data_out_unit     );
        parsec_compute_best_unit( transferred_d2d[i], &best_d2d,          &d2d_unit          );

        percent_in  = (0 == required_in[i])? nan(""): (((double)transferred_in[i])  / (double)required_in[i] ) * 100.0;
        percent_d2d = (0 == required_in[i])? nan(""): (((double)transferred_d2d[i])  / (double)required_in[i] ) * 100.0;
        percent_out = (0 == required_out[i])? nan(""): (((double)transferred_out[i])  / (double)required_out[i] ) * 100.0;

        printf("|  Dev %2d |%10"PRIu64" | %6.2f | %8.2f%2s |   %8.2f%2s(%5.2f)   |   %8.2f%2s(%5.2f)   | %8.2f%2s | %8.2f%2s(%5.2f) | %s\n",
               device->device_index, executed_tasks[i], (executed_tasks[i]/gtotal)*100.00,
               best_required_in,  required_in_unit,  best_data_in,  data_in_unit, percent_in,
               best_d2d, d2d_unit, percent_d2d,
               best_required_out, required_out_unit, best_data_out, data_out_unit, percent_out,
               device->name );
    }

    printf("|---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|\n");

    parsec_compute_best_unit( total_required_in,  &best_required_in,  &required_in_unit  );
    parsec_compute_best_unit( total_required_out, &best_required_out, &required_out_unit );
    parsec_compute_best_unit( total_data_in,      &best_data_in,      &data_in_unit      );
    parsec_compute_best_unit( total_data_out,     &best_data_out,     &data_out_unit     );
    parsec_compute_best_unit( total_d2d,          &best_d2d,          &d2d_unit          );

    percent_in  = (0 == total_required_in)? nan(""): (((double)total_data_in)  / (double)total_required_in) * 100.0;
    percent_d2d = (0 == total_required_in)? nan(""): (((double)total_d2d)  / (double)total_required_in) * 100.0;
    percent_out = (0 == total_required_out)? nan(""): (((double)total_data_out)  / (double)total_required_out) * 100.0;

    printf("|All Devs |%10"PRIu64" | %6.2f | %8.2f%2s |   %8.2f%2s(%5.2f)   |   %8.2f%2s(%5.2f)   | %8.2f%2s | %8.2f%2s(%5.2f) |\n",
           total_tasks, (total_tasks/gtotal)*100.00,
           best_required_in,  required_in_unit,  best_data_in,  data_in_unit, percent_in,
           best_d2d, d2d_unit, percent_d2d,
           best_required_out, required_out_unit, best_data_out, data_out_unit, percent_out);
    printf("+----------------------------------------------------------------------------------------------------------------------------+\n");

    parsec_devices_free_statistics(&end_stats);
}

void parsec_mca_device_reset_statistics(parsec_context_t *parsec_context) {
    parsec_device_module_t *device;

    (void)parsec_context;
    for(uint32_t i = 0; i < parsec_nb_devices; i++) {
        if( NULL == (device = parsec_devices[i]) ) continue;
        assert( i == device->device_index );
        device->executed_tasks       = 0;
        memset(device->data_in_from_device, 0, sizeof(uint64_t)*device->data_in_array_size);
        device->data_out_to_host     = 0;
        device->required_data_in     = 0;
        device->required_data_out    = 0;
    }
}

void parsec_mca_device_dump_and_reset_statistics(parsec_context_t* parsec_context)
{
    parsec_devices_print_statistics(parsec_context, NULL);

    uint64_t d2dtmp; float best_d2d; char *d2d_unit;
    uint32_t i;
    parsec_device_module_t *device;
    printf("\n"
           "Full transfer matrix:\n"
           "dst\\src ");
    for(i = 0; i < parsec_nb_devices; i++) {
        if(NULL ==  parsec_devices[i]) continue;
        printf("%10d ", i);
    }
    printf("\n");
    // 0 is stored in the other devices, because they push to 0, 0 doesn't pull data.
    printf(" %3d        -     ", 0);
    for(i = 1; i < parsec_nb_devices; i++) {
        if( NULL == (device = parsec_devices[i]) ) continue;
        assert( i == device->device_index );
        parsec_compute_best_unit(device->data_out_to_host, &best_d2d, &d2d_unit);
        printf(" %8.2f%2s", best_d2d, d2d_unit);
    }
    printf("\n");
    // The other devices pull data, and they have counted locally how much
    for(i = 1; i < parsec_nb_devices; i++) {
        if( NULL == (device = parsec_devices[i]) ) continue;
        assert( i == device->device_index );
        printf(" %3d   ", i);
        for(unsigned int j = 0; j < parsec_nb_devices; j++) {
            if( device->data_in_array_size ) {
                d2dtmp = device->data_in_from_device[j];
            } else {
                d2dtmp = 0;
            }
            parsec_compute_best_unit( d2dtmp, &best_d2d, &d2d_unit);
            if(i!=j) printf(" %8.2f%2s", best_d2d, d2d_unit);
            else printf("     -     ");
        }
        printf("\n");
    }

    /**
     * Reset the statistics for next turn if there is one.
     */
    parsec_mca_device_reset_statistics(parsec_context);
}

int parsec_mca_device_fini(void)
{
    int show_stats_index, show_stats = 0;

    /* If no statistics are required */
    show_stats_index = parsec_mca_param_find("device", NULL, "show_statistics");
    if( 0 < show_stats_index )
        parsec_mca_param_lookup_int(show_stats_index, &show_stats);
    if( show_stats ) {
        parsec_mca_device_dump_and_reset_statistics(NULL);
    }

    parsec_device_module_t *module;
    mca_base_component_t *component;
    for(int i = 0; i < num_modules_activated; i++ ) {
        module = modules_activated[i];
        component = (mca_base_component_t*)module->component;
        component->mca_close_component();
        modules_activated[i] = NULL;
    }
    num_modules_activated = 0;
    free(modules_activated); modules_activated = NULL;

    if( NULL != parsec_device_recursive ) {  /* Release recursive device */
        parsec_mca_device_remove(parsec_device_recursive);
        free(parsec_device_recursive); parsec_device_recursive = NULL;
    }
    if( NULL != parsec_device_cpus ) {  /* Release the main CPU device */
        parsec_mca_device_remove(parsec_device_cpus);
        free(parsec_device_cpus); parsec_device_cpus = NULL;
    }

    free(parsec_devices); parsec_devices = NULL;

    if( 0 < parsec_device_verbose ) {
        parsec_output_close(parsec_device_output);
        parsec_device_output = parsec_debug_output;
    }

    PARSEC_OBJ_DESTRUCT(&parsec_per_device_infos);
    PARSEC_OBJ_DESTRUCT(&parsec_per_stream_infos);

    return PARSEC_SUCCESS;
}

void*
parsec_device_find_function(const char* function_name,
                            const char* libname,
                            const char* paths[])
{
    char library_name[FILENAME_MAX];
    const char **target;
    void *fn = NULL;

    for( target = paths; (NULL != target) && (NULL != *target); target++ ) {
        struct stat status;
        if( 0 != stat(*target, &status) ) {
            parsec_debug_verbose(80, parsec_device_output,
                                 "Could not stat the %s path (%s)", *target, strerror(errno));
            continue;
        }
        if( S_ISDIR(status.st_mode) ) {
            if( NULL == libname )
                continue;
            snprintf(library_name,  FILENAME_MAX, "%s/%s", *target, libname);
        } else {
            snprintf(library_name,  FILENAME_MAX, "%s", *target);
        }
#if defined(__WINDOWS__)
        wchar_t wlibrary_name[FILENAME_MAX];
        MultiByteToWideChar(CP_ACP, MB_COMPOSITE, library_name, strlen(library_name),
                            wlibrary_name, FILENAME_MAX);
        HMODULE dlh = LoadLibraryW(wlibrary_name);
        if(NULL == dlh) {
            parsec_debug_verbose(80, parsec_device_output,
                                 "Could not find %s dynamic library (%s)", library_name, GetLastError());
            continue;
        }
        fn = GetProcAddress(dlh, function_name);
        FreeLibrary(dlh);
#elif defined(PARSEC_HAVE_DLFCN_H)
        void* dlh = dlopen(library_name, RTLD_NOW | RTLD_NODELETE );
        if(NULL == dlh) {
            parsec_debug_verbose(80, parsec_device_output,
                                 "Could not find %s dynamic library (%s)", library_name, dlerror());
            continue;
        }
        fn = dlsym(dlh, function_name);
        dlclose(dlh);
#else
#error Lacking dynamic loading capabilities.
#endif
        if( NULL != fn ) {
            parsec_debug_verbose(4, parsec_device_output,
                                 "Function %s found in shared library %s",
                                 function_name, library_name);
            break;  /* we got one, stop here */
        }
    }
    /* Couldn't load from named dynamic libs, try linked/static */
    if(NULL == fn) {
        parsec_output_verbose(80, parsec_device_output,
                              "No dynamic function %s found, trying from compile time linked in\n",
                              function_name);
#if defined(__WINDOWS__)
        HMODULE dlh = GetModuleHandleA(NULL);
        if(NULL != dlh) {
            fn = GetProcAddress(dlh, function_name);
            FreeLibrary(dlh);
        }
#elif defined(PARSEC_HAVE_DLFCN_H)
        void* dlh = dlopen(NULL, RTLD_NOW | RTLD_NODELETE);
        if(NULL != dlh) {
            fn = dlsym(dlh, function_name);
            dlclose(dlh);
        }
#else
#error Lacking dynamic loading capabilities.
#endif
            if(NULL != fn) {
                parsec_debug_verbose(4, parsec_device_output,
                                     "Function %s found in the application symbols",
                                     function_name);
            }
    }
    if(NULL == fn) {
        parsec_debug_verbose(10, parsec_device_output,
                             "No function %s found", function_name);
    }
    return fn;
}

int parsec_mca_device_registration_complete(parsec_context_t* context)
{
    int64_t total_gflops_fp16 = 0, total_gflops_fp32 = 0, total_gflops_fp64 = 0, total_gflops_tf32 = 0, c;
    (void)context;

    if(parsec_mca_device_are_freezed)
        return PARSEC_ERR_NOT_SUPPORTED;
    parsec_mca_device_are_freezed = 1;

    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        parsec_device_module_t* device = parsec_devices[i];
        if( NULL == device ) continue;
        if( PARSEC_DEV_RECURSIVE == device->type ) continue;
        if( PARSEC_DEV_CPU == device->type ) {
            c = 0;
            for(int p = 0; p < context->nb_vp; p++)
                c += context->virtual_processes[p]->nb_cores;
        }
        else {
            c = 1;
        }
        total_gflops_fp16 += c * device->gflops_fp16;
        total_gflops_tf32 += c * device->gflops_tf32;
        total_gflops_fp32 += c * device->gflops_fp32;
        total_gflops_fp64 += c * device->gflops_fp64;
    }

    /* Compute the weight of each device including the cores */
    parsec_debug_verbose(6, parsec_device_output, "Global Theoretical performance:        double %-8"PRId64" single %-8"PRId64" tensor %-8"PRId64" half %-8"PRId64, total_gflops_fp64, total_gflops_fp32, total_gflops_tf32, total_gflops_fp16);
    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        parsec_device_module_t* device = parsec_devices[i];
        if( NULL == device ) continue;
        if( PARSEC_DEV_RECURSIVE == device->type ) continue;
        device->time_estimate_default = total_gflops_fp64/(double)device->gflops_fp64;
        parsec_debug_verbose(6, parsec_device_output, "  Dev[%d] default-time-estimate %-4"PRId64" <- double %-8"PRId64" single %-8"PRId64" tensor %-8"PRId64" half %-8"PRId64" %s",
                             i, device->time_estimate_default, device->gflops_fp64, device->gflops_fp32, device->gflops_tf32, device->gflops_fp16, device->gflops_guess? "GUESSED": "");
        if(NULL != device->all_devices_attached) {
            device->all_devices_attached(device);
        }
    }

    return PARSEC_SUCCESS;
}

int parsec_mca_device_registration_completed(parsec_context_t* context)
{
    (void)context;
    return parsec_mca_device_are_freezed;
}

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static int cpu_weights(parsec_device_module_t* device, int nstreams)
{
    /* This is default value when it cannot be computed */
    /* Crude estimate that holds for Nehalem era Xeon processors */
    float freq = 2.5f;
    float fp_ipc = 8.f;
    float dp_ipc = 4.f;
    char cpu_model[256]="Unkown";
    char *cpu_flags = NULL;

#if defined(__linux__)
    FILE* procinfo = fopen("/proc/cpuinfo", "r");
    if( NULL == procinfo ) {
        parsec_warning("CPU Features cannot be autodetected on this machine: %s", strerror(errno));
        goto notfound;
    }
    cpu_flags = calloc(4096, sizeof(char));
    char str[4096];
    while( NULL != fgets(str, 4096, procinfo) ) {
        /* Intel/AMD */
        sscanf(str, "model name : %255[^\n]%*c", cpu_model);
        if( 0 != sscanf(str, "cpu MHz : %f", &freq) )
            freq *= 1e-3;
        if( 0 != sscanf(str, "flags : %4095[^\n]%*c", cpu_flags) )
            break; /* done reading for an x86 type CPU */
        /* IBM: Power */
        sscanf(str, "cpu : %255[^\n]%*c", cpu_model);
        if( 0 != sscanf(str, "clock : %fMHz", &freq) ) {
            freq *= 1e-3;
            break; /* done reading for a Power type CPU */
        }
    }
    fclose(procinfo);
#elif defined(__APPLE__)
    size_t len = sizeof(cpu_model);
    int rc = sysctlbyname("machdep.cpu.brand_string", cpu_model, &len, NULL, 0);
    if( rc ) {
        parsec_warning("CPU Features cannot be autodetected on this machine (Detected OSX): %s", strerror(errno));
        goto notfound;
    }
    len = 0;
    rc = sysctlbyname("machdep.cpu.features", NULL, &len, NULL, 0);
    cpu_flags = malloc(len);
    rc = sysctlbyname("machdep.cpu.features", cpu_flags, &len, NULL, 0);
    if( rc ) {
        parsec_warning("CPU Features cannot be autodetected on this machine (Detected OSX): %s", strerror(errno));
        goto notfound;
    }
#else
    goto notfound;
#endif
    /* prefer base frequency from model name when available (avoids power
     * saving modes and dynamic frequency scaling issues) */
    sscanf(cpu_model, "%*[^@] @ %fGHz", &freq);

    fp_ipc = 8;
    dp_ipc = 4;
#if defined(__x86_64__) || defined(__i386__)
#if defined(PARSEC_HAVE_BUILTIN_CPU)
    __builtin_cpu_init();
#if defined(__AVX__)
    if(__builtin_cpu_supports("avx")) {
        fp_ipc = 16;
        dp_ipc = 8;
    }
#endif  /* defined(__AVX__) */
#if defined(__AVX2__)
    if(__builtin_cpu_supports("avx2")) {
        fp_ipc = 32;
        dp_ipc = 16;
    }
#endif  /* defined(__AVX2__) */
#if defined(__AVX512F__)
    if(__builtin_cpu_supports("avx512f")) {
        fp_ipc = 64;
        dp_ipc = 32;
    }
#endif  /* defined(__AVX512F__) */
#else
    if( strstr(cpu_flags, " avx512f") ) {
        fp_ipc = 64;
        dp_ipc = 32;
    }
    else if( strstr(cpu_flags, " avx2") ) {
        fp_ipc = 32;
        dp_ipc = 16;
    }
    else if( strstr(cpu_flags, " avx") ) {
        fp_ipc = 16;
        dp_ipc = 8;
    }
#endif
#endif  /* defined(__x86_64__) || defined(__i386__) */
    free(cpu_flags);

    {
      int show_caps = 0;
      int show_caps_index = parsec_mca_param_find("device", NULL, "show_capabilities");
      if(0 < show_caps_index) {
          parsec_mca_param_lookup_int(show_caps_index, &show_caps);
      }
      if( show_caps ) {
          parsec_inform("CPU Device: %s\n"
                        "\tParsec Streams     : %d\n"
                        "\tFrequency (GHz)    : %.2f\n"
                        "\tPeak Tflop/s       : fp64: %-8.3f fp32: %-8.3f",
                        cpu_model,
                        nstreams,
                        freq, nstreams*freq*dp_ipc*1e-3, nstreams*freq*fp_ipc*1e-3);
       }
    }
 notfound:

    /* compute capacity is per-core, not per-device, so as to account for the
     * prevalent model where we use sequential, single threaded tasks on CPU devices.
     * Advanced users can use the time_estimate property to override if using
     * multi-core parallel tasks. */
    device->gflops_fp16 = fp_ipc * freq; /* No processor have half precision for now */
    device->gflops_tf32 = fp_ipc * freq; /* No processor support tensor operations for now */
    device->gflops_fp32 = fp_ipc * freq;
    device->gflops_fp64 = dp_ipc * freq;

    return PARSEC_SUCCESS;
}

static int
device_taskpool_register_static(parsec_device_module_t* device, parsec_taskpool_t* tp)
{
    int32_t rc = PARSEC_ERR_NOT_FOUND;
    uint32_t i, j;

    /**
     * Detect if a particular chore has a dynamic load dependency and if yes
     * load the corresponding module and find the function.
     */
    assert(tp->devices_index_mask & (1 << device->device_index));

    for( i = 0; i < tp->nb_task_classes; i++ ) {
        const parsec_task_class_t* tc = tp->task_classes_array[i];
        if(NULL == tp->task_classes_array[i])
            continue;
        __parsec_chore_t* chores = (__parsec_chore_t*)tc->incarnations;
        for( j = 0; NULL != chores[j].hook; j++ ) {
            if( chores[j].type != device->type )
                continue;
            if(  NULL != chores[j].dyld_fn ) {
                continue;  /* the function has been set for another device of the same type */
            }
            if ( NULL == chores[j].dyld ) {
                assert( NULL == chores[j].dyld_fn );  /* No dynamic support required for this kernel */
                rc = PARSEC_SUCCESS;
            } else {
                void* devf = parsec_device_find_function(chores[j].dyld, NULL, NULL);
                if( NULL != devf ) {
                    chores[j].dyld_fn = devf;
                    rc = PARSEC_SUCCESS;
                }
            }
        }
    }
    if( PARSEC_SUCCESS != rc ) {
        tp->devices_index_mask &= ~(1 << device->device_index);  /* discard this type */
        parsec_debug_verbose(10, parsec_device_output,
                             "Device %d:%s disabled for taskpool %s (%p)", device->device_index, device->name, tp->taskpool_name, tp);
    }

    return rc;
}

int parsec_mca_device_attach(parsec_context_t* context)
{
    parsec_device_base_component_t *component;
    parsec_device_module_t *module;
    int nb_total_comp_threads = 0, rc;

    for(int p = 0; p < context->nb_vp; p++) {
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;
    }

 #if defined(PARSEC_HAVE_DEV_CPU_SUPPORT)
    /* Add the predefined devices: one device for the CPUs */
    {
        parsec_device_cpus = (parsec_device_module_t*)calloc(1, sizeof(parsec_device_module_t));
        parsec_device_cpus->name = "cpu-cores";
        parsec_device_cpus->type = PARSEC_DEV_CPU;
        parsec_device_cpus->data_in_array_size = 0;
        parsec_device_cpus->data_in_from_device = NULL;
        cpu_weights(parsec_device_cpus, nb_total_comp_threads);
        parsec_device_cpus->taskpool_register = device_taskpool_register_static;
        parsec_mca_device_add(context, parsec_device_cpus);
    }
 #endif  /* defined(PARSEC_HAVE_DEV_CPU_SUPPORT) */

#if defined(PARSEC_HAVE_DEV_RECURSIVE_SUPPORT)
    /* and one for the recursive kernels */
    {
        parsec_device_recursive = (parsec_device_module_t*)calloc(1, sizeof(parsec_device_module_t));
        parsec_device_recursive->name = "cpu-recursive";
        parsec_device_recursive->type = PARSEC_DEV_RECURSIVE;
        parsec_device_recursive->data_in_array_size = 0;
        parsec_device_recursive->data_in_from_device = NULL;
        parsec_device_recursive->gflops_fp16 = parsec_device_cpus->gflops_fp16;
        parsec_device_recursive->gflops_tf32 = parsec_device_cpus->gflops_tf32;
        parsec_device_recursive->gflops_fp32 = parsec_device_cpus->gflops_fp32;
        parsec_device_recursive->gflops_fp64 = parsec_device_cpus->gflops_fp64;
        parsec_device_recursive->taskpool_register = device_taskpool_register_static;
        parsec_mca_device_add(context, parsec_device_recursive);
    }
#endif  /* defined(PARSEC_HAVE_DEV_RECURSIVE_SUPPORT) */

    for( int i = 0; NULL != (component = (parsec_device_base_component_t*)device_components[i]); i++ ) {
        for( int j = 0; NULL != (module = component->modules[j]); j++ ) {
            if (NULL == module->attach) {
                parsec_debug_verbose(10, parsec_device_output, "A device module MUST contain an attach function. Disqualifying component module %s/%s",
                                     component->base_version.mca_component_name, module->name);
                continue;
            }
            rc = module->attach(module, context);
            if( 0 > rc ) {
                parsec_debug_verbose(10, parsec_device_output, "Attach failed for device component module %s/%s on context %p.",
                                     component->base_version.mca_component_name, module->name, context);
                continue;
            }
            module->data_in_array_size = 0;
            module->data_in_from_device = NULL;
            parsec_debug_verbose(5, parsec_device_output, "Activated device component module %s/%s on context %p.",
                                 component->base_version.mca_component_name, module->name, context);
        }
    }
    /* Now that all the devices have been correctly attached, we can prepare the memory for
     * the transfer statistics.
     */
    for( int i = 0; i < (int)parsec_nb_devices; i++ ) {
        module = parsec_devices[i];
        module->data_in_array_size = parsec_nb_devices;
        module->data_in_from_device = (uint64_t*)calloc(module->data_in_array_size, sizeof(uint64_t));
    }
    return PARSEC_SUCCESS;
}

int parsec_mca_device_add(parsec_context_t* context, parsec_device_module_t* device)
{
    if( parsec_mca_device_are_freezed ) {
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    if( NULL != device->context ) {
        /* This device already belong to a PaRSEC context */
        return PARSEC_ERR_BAD_PARAM;
    }
    parsec_atomic_lock(&parsec_devices_mutex);  /* CRITICAL SECTION: BEGIN */
    if( (parsec_nb_devices+1) > parsec_nb_max_devices ) {
        if( NULL == parsec_devices ) /* first time */
            parsec_nb_max_devices = 4;
        else
            parsec_nb_max_devices *= 2; /* every other time */
        parsec_devices = realloc(parsec_devices, parsec_nb_max_devices * sizeof(parsec_device_module_t*));
    }
    parsec_devices[parsec_nb_devices] = device;
    device->device_index = parsec_nb_devices;
    parsec_nb_devices++;
    device->context = context;
    parsec_atomic_unlock(&parsec_devices_mutex);  /* CRITICAL SECTION: END */
    PARSEC_OBJ_CONSTRUCT(&device->infos, parsec_info_object_array_t);
    parsec_info_object_array_init(&device->infos, &parsec_per_device_infos, device);
    return device->device_index;
}

parsec_device_module_t* parsec_mca_device_get(uint32_t device_index)
{
    if( device_index >= parsec_nb_devices )
        return NULL;
    return parsec_devices[device_index];
}

int parsec_mca_device_is_gpu(uint32_t devindex) {
    parsec_device_module_t *dev = parsec_mca_device_get(devindex);
    if(NULL == dev) return false;
    return PARSEC_DEV_RECURSIVE < dev->type;
}

int parsec_mca_device_remove(parsec_device_module_t* device)
{
    int rc = PARSEC_SUCCESS;

    PARSEC_OBJ_DESTRUCT(&device->infos);
    parsec_atomic_lock(&parsec_devices_mutex);  /* CRITICAL SECTION: BEGIN */
    if( NULL == device->context ) {
        rc = PARSEC_ERR_BAD_PARAM;
        goto unlock_and_return_rc;
    }
    if(device != parsec_devices[device->device_index]) {
        rc = PARSEC_ERR_NOT_FOUND;
        goto unlock_and_return_rc;
    }
    parsec_devices[device->device_index] = NULL;
    device->context = NULL;
    device->device_index = -1;
    if(NULL != device->data_in_from_device) {
        free(device->data_in_from_device);
        device->data_in_from_device = NULL;
        device->data_in_array_size = 0;
    }
  unlock_and_return_rc:
    parsec_atomic_unlock(&parsec_devices_mutex);  /* CRITICAL SECTION: END */
    return rc;
}


void parsec_mca_device_taskpool_restrict(parsec_taskpool_t *tp,
                                         uint8_t            device_type)
{
    parsec_device_module_t *device;
    uint32_t i;

    for (i = 0; i < parsec_nb_devices; i++) {
        device = parsec_mca_device_get(i);
        if ((NULL == device) || (device->type & device_type))
            continue;

        /* Force unregistration for this type of device. This is not correct, as some of
         * the memory related to the taskpool might still be registered with the
         * devices we drop support. */
        tp->devices_index_mask &= ~(1 << device->device_index);
    }
    return;
}

int parsec_advise_data_on_device(parsec_data_t *data, int device, int advice)
{
    parsec_device_module_t *dev = parsec_mca_device_get(device);

    if( NULL == dev )
        return PARSEC_ERR_NOT_FOUND;
    if( NULL == dev->data_advise )
        return PARSEC_SUCCESS;
    return dev->data_advise(dev, data, advice);
}

void parsec_devices_reset_load(parsec_context_t *context)
{
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *dev = parsec_mca_device_get(i);
        dev->device_load = 0;
    }
    (void)context;
}

int parsec_devices_release_memory(void)
{
    parsec_device_module_t *dev;
    for(int i = 1; i < (int)parsec_nb_devices; i++) {
        dev = parsec_mca_device_get(i);
        if((NULL != dev) && (NULL != dev->memory_release)) {
            dev->memory_release(dev);
        }
    }
    return PARSEC_SUCCESS;
}
