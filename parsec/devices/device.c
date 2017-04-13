/*
 *
 * Copyright (c) 2013-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec/devices/device.h"
#include "parsec/utils/mca_param.h"
#include "parsec/constants.h"
#include "parsec/debug.h"
#include "parsec/execution_unit.h"
#include "parsec/utils/argv.h"

#include <stdlib.h>

#include <parsec/devices/cuda/dev_cuda.h>

uint32_t parsec_nb_devices = 0;
static uint32_t parsec_nb_max_devices = 0;
static uint32_t parsec_devices_are_freezed = 0;
parsec_atomic_lock_t parsec_devices_mutex = { PARSEC_ATOMIC_UNLOCKED };
static parsec_device_t** parsec_devices = NULL;
static char* parsec_device_list_str = NULL, **parsec_device_list = NULL;

static parsec_device_t* parsec_device_cpus = NULL;
static parsec_device_t* parsec_device_recursive = NULL;

/**
 * Temporary solution: Use the following two arrays to handle the weight and
 * the load on different devices. These arrays are not available before the
 * call to parsec_devices_freeze(). This is just a first step, a smarter approach
 * should take this spot.
 */
float *parsec_device_load = NULL;
float *parsec_device_sweight = NULL;
float *parsec_device_dweight = NULL;

int parsec_devices_init(parsec_context_t* parsec_context)
{
    (void)parsec_mca_param_reg_string_name("device", NULL,
                                           "Comma delimited list of devices to be enabled (or all)",
                                           false, false, "all", &parsec_device_list_str);
    (void)parsec_mca_param_reg_int_name("device", "show_capabilities",
                                        "Show the detailed devices capabilities",
                                        false, false, 0, NULL);
    (void)parsec_mca_param_reg_int_name("device", "show_statistics",
                                        "Show the detailed devices statistics upon exit",
                                        false, false, 0, NULL);
    parsec_device_list = parsec_argv_split(parsec_device_list_str, ',');

    (void)parsec_context;
    return PARSEC_ERR_NOT_IMPLEMENTED;
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

int parsec_devices_fini(parsec_context_t* parsec_context)
{
    parsec_device_t *device;
    int show_stats_index, show_stats = 0;

    /* If no statistics are required */
    show_stats_index = parsec_mca_param_find("device", NULL, "show_statistics");
    if( 0 < show_stats_index )
        parsec_mca_param_lookup_int(show_stats_index, &show_stats);
    (void)parsec_context;
    if( show_stats ) {
        int *device_counter, total = 0;
        uint64_t total_data_in = 0,     total_data_out = 0;
        uint64_t total_required_in = 0, total_required_out = 0;
        uint64_t *transferred_in, *transferred_out;
        uint64_t *required_in,    *required_out;
        float gtotal = 0.0;
        float best_data_in, best_data_out;
        float best_required_in, best_required_out;
        char *data_in_unit, *data_out_unit;
        char *required_in_unit, *required_out_unit;
        char percent1[64], percent2[64];
        uint32_t i;

        /* GPU counter for GEMM / each */
        device_counter  = (int*)     calloc(parsec_nb_devices, sizeof(int)     );
        transferred_in  = (uint64_t*)calloc(parsec_nb_devices, sizeof(uint64_t));
        transferred_out = (uint64_t*)calloc(parsec_nb_devices, sizeof(uint64_t));
        required_in     = (uint64_t*)calloc(parsec_nb_devices, sizeof(uint64_t));
        required_out    = (uint64_t*)calloc(parsec_nb_devices, sizeof(uint64_t));

        /**
         * Save the statistics locally.
         */
        for(i = 0; i < parsec_nb_devices; i++) {
            if( NULL == (device = parsec_devices[i]) ) continue;
            assert( i == device->device_index );
            /* Save the statistics */
            device_counter[device->device_index]  += device->executed_tasks;
            transferred_in[device->device_index]  += device->transferred_data_in;
            transferred_out[device->device_index] += device->transferred_data_out;
            required_in[device->device_index]     += device->required_data_in;
            required_out[device->device_index]    += device->required_data_out;
        }

        /* Print statistics */
        for( i = 0; i < parsec_nb_devices; i++ ) {
            total              += device_counter[i];
            total_data_in      += transferred_in[i];
            total_data_out     += transferred_out[i];
            total_required_in  += required_in[i];
            total_required_out += required_out[i];
        }

        if( 0 == total_data_in )  total_data_in  = 1;
        if( 0 == total_data_out ) total_data_out = 1;
        gtotal = (float)total;

        printf("--------------------------------------------------------------------------------------------------\n");
        printf("|         |                    |         Data In                |         Data Out               |\n");
        printf("|Rank %3d |  # KERNEL |    %%   |  Required  |   Transfered(%%)   |  Required  |   Transfered(%%)   |\n",
               parsec_context->my_rank);
        printf("|---------|-----------|--------|------------|-------------------|------------|-------------------|\n");
        for( i = 0; i < parsec_nb_devices; i++ ) {
            if( NULL == (device = parsec_devices[i]) ) continue;

            parsec_compute_best_unit( required_in[i],     &best_required_in,  &required_in_unit  );
            parsec_compute_best_unit( required_out[i],    &best_required_out, &required_out_unit );
            parsec_compute_best_unit( transferred_in[i],  &best_data_in,      &data_in_unit      );
            parsec_compute_best_unit( transferred_out[i], &best_data_out,     &data_out_unit     );

            printf("|  Dev %2d |%10d | %6.2f | %8.2f%2s | %8.2f%2s(%5.2f) | %8.2f%2s | %8.2f%2s(%5.2f) | %s\n",
                   device->device_index, device_counter[i], (device_counter[i]/gtotal)*100.00,
                   best_required_in,  required_in_unit,  best_data_in,  data_in_unit,
                   (((double)transferred_in[i])  / (double)required_in[i] ) * 100.0,
                   best_required_out, required_out_unit, best_data_out, data_out_unit,
                   (((double)transferred_out[i]) / (double)required_out[i]) * 100.0, device->name );
        }

        printf("|---------|-----------|--------|------------|-------------------|------------|-------------------|\n");

        parsec_compute_best_unit( total_required_in,  &best_required_in,  &required_in_unit  );
        parsec_compute_best_unit( total_required_out, &best_required_out, &required_out_unit );
        parsec_compute_best_unit( total_data_in,      &best_data_in,      &data_in_unit      );
        parsec_compute_best_unit( total_data_out,     &best_data_out,     &data_out_unit     );

        if( 0 == total_required_in ) {
            snprintf(percent1, 64, "nan");
        } else {
            snprintf(percent1, 64, "%5.2f",  ((double)total_data_in  / (double)total_required_in ) * 100.0);
        }
        if( 0 == total_required_out ) {
            snprintf(percent2, 64, "nan");
        } else {
            snprintf(percent2, 64, "%5.2f", ((double)total_data_out / (double)total_required_out) * 100.0);
        }
        printf("|All Devs |%10d | %5.2f | %8.2f%2s | %8.2f%2s(%s) | %8.2f%2s | %8.2f%2s(%s) |\n",
               total, (total/gtotal)*100.00,
               best_required_in,  required_in_unit,  best_data_in,  data_in_unit,
               percent1,
               best_required_out, required_out_unit, best_data_out, data_out_unit,
               percent2);
        printf("-------------------------------------------------------------------------------------------------\n");

        free(device_counter);
        free(transferred_in);
        free(transferred_out);
        free(required_in);
        free(required_out);
    }

    /* Free the local memory */
    if(NULL != parsec_device_load) free(parsec_device_load);
    parsec_device_load = NULL;
    if(NULL != parsec_device_sweight) free(parsec_device_sweight);
    parsec_device_sweight = NULL;
    if(NULL != parsec_device_dweight) free(parsec_device_dweight);
    parsec_device_dweight = NULL;

#if defined(PARSEC_HAVE_CUDA)
    (void)parsec_gpu_fini();
#endif  /* defined(PARSEC_HAVE_CUDA) */

    if( NULL != parsec_device_recursive ) {  /* Release recursive device */
        parsec_devices_remove(parsec_device_recursive);
        free(parsec_device_recursive); parsec_device_recursive = NULL;
    }
    if( NULL != parsec_device_cpus ) {  /* Release the main CPU device */
        parsec_devices_remove(parsec_device_cpus);
        free(parsec_device_cpus); parsec_device_cpus = NULL;
    }

    free(parsec_devices); parsec_devices = NULL;
    free(parsec_device_list_str); parsec_device_list_str = NULL;
    if( NULL != parsec_device_list ) {
        parsec_argv_free(parsec_device_list); parsec_device_list = NULL;
    }
    return PARSEC_SUCCESS;
}

int parsec_devices_freeze(parsec_context_t* context)
{
    float total_sperf = 0.0, total_dperf = 0.0;
    (void)context;

    if(parsec_devices_are_freezed)
        return -1;

    if(NULL != parsec_device_load) free(parsec_device_load);
    parsec_device_load = (float*)calloc(parsec_nb_devices, sizeof(float));
    if(NULL != parsec_device_sweight) free(parsec_device_sweight);
    parsec_device_sweight = (float*)calloc(parsec_nb_devices, sizeof(float));
    if(NULL != parsec_device_dweight) free(parsec_device_dweight);
    parsec_device_dweight = (float*)calloc(parsec_nb_devices, sizeof(float));
    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        parsec_device_t* device = parsec_devices[i];
        if( NULL == device ) continue;
        parsec_device_sweight[i] = device->device_sweight;
        total_sperf += device->device_sweight;
        parsec_device_dweight[i] = device->device_dweight;
        total_dperf += device->device_dweight;
    }

    /* Compute the weight of each device including the cores */
    parsec_debug_verbose(4, parsec_debug_output, "Global Theoretical performance: single %2.4f double %2.4f", total_sperf, total_dperf);
    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        parsec_debug_verbose(4, parsec_debug_output, "  Dev[%d]             ->ratio single %2.4e double %2.4e",
                             i, parsec_device_sweight[i], parsec_device_dweight[i]);

        parsec_device_sweight[i] = (total_sperf / parsec_device_sweight[i]);
        parsec_device_dweight[i] = (total_dperf / parsec_device_dweight[i]);
        /* after the weighting */
        parsec_debug_verbose(4, parsec_debug_output, "  Dev[%d]             ->ratio single %2.4e double %2.4e",
                             i, parsec_device_sweight[i], parsec_device_dweight[i]);
    }

    parsec_devices_are_freezed = 1;
    return 0;
}

int parsec_devices_freezed(parsec_context_t* context)
{
    (void)context;
    return parsec_devices_are_freezed;
}

int parsec_devices_select(parsec_context_t* context)
{
    int nb_total_comp_threads = 0;

    for(int p = 0; p < context->nb_vp; p++) {
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;
    }

    /* By now let's add one device for the CPUs */
    {
        parsec_device_cpus = (parsec_device_t*)calloc(1, sizeof(parsec_device_t));
        parsec_device_cpus->name = "default";
        parsec_device_cpus->type = PARSEC_DEV_CPU;
        parsec_devices_add(context, parsec_device_cpus);
        /* TODO: This is plain WRONG, but should work by now */
        parsec_device_cpus->device_sweight = nb_total_comp_threads * 8 * (float)2.27;
        parsec_device_cpus->device_dweight = nb_total_comp_threads * 4 * 2.27;
    }

    /* By now let's add one device for the recursive kernels */
    {
        parsec_device_recursive = (parsec_device_t*)calloc(1, sizeof(parsec_device_t));
        parsec_device_recursive->name = "recursive";
        parsec_device_recursive->type = PARSEC_DEV_RECURSIVE;
        parsec_devices_add(context, parsec_device_recursive);
        /* TODO: This is plain WRONG, but should work by now */
        parsec_device_recursive->device_sweight = nb_total_comp_threads * 8 * (float)2.27;
        parsec_device_recursive->device_dweight = nb_total_comp_threads * 4 * 2.27;
    }
#if defined(PARSEC_HAVE_CUDA)
    return parsec_gpu_init(context);
#else
    return PARSEC_SUCCESS;
#endif  /* defined(PARSEC_HAVE_CUDA) */
}

int parsec_devices_add(parsec_context_t* context, parsec_device_t* device)
{
    if( parsec_devices_are_freezed ) {
        return PARSEC_ERROR;
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
        parsec_devices = realloc(parsec_devices, parsec_nb_max_devices * sizeof(parsec_device_t*));
    }
    parsec_devices[parsec_nb_devices] = device;
    device->device_index = parsec_nb_devices;
    parsec_nb_devices++;
    device->context = context;
    parsec_atomic_unlock(&parsec_devices_mutex);  /* CRITICAL SECTION: END */
    return device->device_index;
}

parsec_device_t* parsec_devices_get(uint32_t device_index)
{
    if( device_index >= parsec_nb_devices )
        return NULL;
    return parsec_devices[device_index];
}

int parsec_devices_remove(parsec_device_t* device)
{
    int rc = PARSEC_SUCCESS;

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
  unlock_and_return_rc:
    parsec_atomic_unlock(&parsec_devices_mutex);  /* CRITICAL SECTION: BEGIN */
    return rc;
}


void parsec_devices_handle_restrict(parsec_handle_t *handle,
                                    uint8_t         devices_type)
{
    parsec_device_t *device;
    uint32_t i;

    for (i = 0; i < parsec_nb_devices; i++) {
	if (!(handle->devices_mask & (1 << i)))
	    continue;

	device = parsec_devices_get(i);
	if ((NULL == device) || (device->type & devices_type))
	    continue;

        /* Disable this type of device */
        handle->devices_mask &= ~(1 << i);
    }
    return;
}

