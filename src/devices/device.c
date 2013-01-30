/*
 *
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include <dague/devices/device.h>
#include <dague/utils/mca_param.h>
#include <dague/constants.h>
#include "debug.h"
#include "execution_unit.h"

#include <stdlib.h>

#include <dague/devices/cuda/dev_cuda.h>

uint32_t dague_nb_devices = 0;
static uint32_t dague_nb_max_devices = 0;
static uint32_t dague_devices_are_freezed = 0;
uint32_t dague_devices_mutex = 0;  /* unlocked */
static dague_device_t** dague_devices = NULL;

/**
 * Temporary solution: Use the following two arrays to handle the weight and
 * the load on different devices. These arrays are not available before the
 * call to dague_devices_freeze(). This is just a first step, a smarter approach
 * should take this spot.
 */
float *dague_device_load = NULL;
float *dague_device_sweight = NULL;
float *dague_device_dweight = NULL;

int dague_devices_init(dague_context_t* dague_context)
{
    (void)dague_mca_param_reg_int_name("device", "show_capabilities",
                                       "Show the detailed devices capabilities",
                                       false, false, 0, NULL);
    (void)dague_mca_param_reg_string_name("device", NULL,
                                          "Comma delimited list of devices to be enabled",
                                          false, false, "none", NULL);
    (void)dague_mca_param_reg_int_name("device", "show_statistics",
                                       "Show the detailed devices statistics upon exit",
                                       false, false, 0, NULL);
    (void)dague_context;
    return DAGUE_ERR_NOT_IMPLEMENTED;
}

static void
dague_compute_best_unit( uint64_t length, float* updated_value, char** best_unit )
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

int dague_devices_fini(dague_context_t* dague_context)
{
    dague_device_t *device;
    int show_stats_index, show_stats = 0;

    /* If no statistics are required */
    show_stats_index = dague_mca_param_find("device", NULL, "show_statistics");
    if( 0 < show_stats_index )
        dague_mca_param_lookup_int(show_stats_index, &show_stats);
    (void)dague_context;
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
        uint32_t i;

        /* GPU counter for GEMM / each */
        device_counter  = (int*)     calloc(dague_nb_devices, sizeof(int)     );
        transferred_in  = (uint64_t*)calloc(dague_nb_devices, sizeof(uint64_t));
        transferred_out = (uint64_t*)calloc(dague_nb_devices, sizeof(uint64_t));
        required_in     = (uint64_t*)calloc(dague_nb_devices, sizeof(uint64_t));
        required_out    = (uint64_t*)calloc(dague_nb_devices, sizeof(uint64_t));

        /**
         * Save the statistics locally.
         */
        for(i = 0; i < dague_nb_devices; i++) {
            if( NULL == (device = dague_devices[i]) ) continue;
            assert( i == device->device_index );
            /* Save the statistics */
            device_counter[device->device_index]  += device->executed_tasks;
            transferred_in[device->device_index]  += device->transferred_data_in;
            transferred_out[device->device_index] += device->transferred_data_out;
            required_in[device->device_index]     += device->required_data_in;
            required_out[device->device_index]    += device->required_data_out;
        }

        /* Print statistics */
        for( i = 0; i < dague_nb_devices; i++ ) {
            total              += device_counter[i];
            total_data_in      += transferred_in[i];
            total_data_out     += transferred_out[i];
            total_required_in  += required_in[i];
            total_required_out += required_out[i];
        }

        if( 0 == total_data_in )  total_data_in  = 1;
        if( 0 == total_data_out ) total_data_out = 1;
        gtotal = (float)total;

        printf("-------------------------------------------------------------------------------------------------\n");
        printf("|         |                   |         Data In                |         Data Out               |\n");
        printf("|Rank %3d |  # KERNEL |   %%   |  Required  |   Transfered(%%)   |  Required  |   Transfered(%%)   |\n",
               dague_context->my_rank);
        printf("|---------|-----------|-------|------------|-------------------|------------|-------------------|\n");
        for( i = 0; i < dague_nb_devices; i++ ) {
            if( NULL == (device = dague_devices[i]) ) continue;

            dague_compute_best_unit( required_in[i],     &best_required_in,  &required_in_unit  );
            dague_compute_best_unit( required_out[i],    &best_required_out, &required_out_unit );
            dague_compute_best_unit( transferred_in[i],  &best_data_in,      &data_in_unit      );
            dague_compute_best_unit( transferred_out[i], &best_data_out,     &data_out_unit     );

            printf("|  Dev %2d |%10d | %5.2f | %8.2f%2s | %8.2f%2s(%5.2f) | %8.2f%2s | %8.2f%2s(%5.2f) | %s\n",
                   device->device_index, device_counter[i], (device_counter[i]/gtotal)*100.00,
                   best_required_in,  required_in_unit,  best_data_in,  data_in_unit,
                   (((double)transferred_in[i])  / (double)required_in[i] ) * 100.0,
                   best_required_out, required_out_unit, best_data_out, data_out_unit,
                   (((double)transferred_out[i]) / (double)required_out[i]) * 100.0, device->name );
        }

        printf("|---------|-----------|-------|------------|-------------------|------------|-------------------|\n");

        dague_compute_best_unit( total_required_in,  &best_required_in,  &required_in_unit  );
        dague_compute_best_unit( total_required_out, &best_required_out, &required_out_unit );
        dague_compute_best_unit( total_data_in,      &best_data_in,      &data_in_unit      );
        dague_compute_best_unit( total_data_out,     &best_data_out,     &data_out_unit     );

        printf("|All Devs |%10d | %5.2f | %8.2f%2s | %8.2f%2s(%5.2f) | %8.2f%2s | %8.2f%2s(%5.2f) |\n",
               total, (total/gtotal)*100.00,
               best_required_in,  required_in_unit,  best_data_in,  data_in_unit,
               ((double)total_data_in  / (double)total_required_in ) * 100.0,
               best_required_out, required_out_unit, best_data_out, data_out_unit,
               ((double)total_data_out / (double)total_required_out) * 100.0);
        printf("-------------------------------------------------------------------------------------------------\n");

        free(device_counter);
        free(transferred_in);
        free(transferred_out);
        free(required_in);
        free(required_out);
    }
    /* Free the local memory */
    if(NULL != dague_device_load) free(dague_device_load);
    dague_device_load = NULL;
    if(NULL != dague_device_sweight) free(dague_device_sweight);
    dague_device_sweight = NULL;
    if(NULL != dague_device_dweight) free(dague_device_dweight);
    dague_device_dweight = NULL;

#if defined(HAVE_CUDA)
    return dague_gpu_fini();
#else
    return 0;
#endif  /* defined(HAVE_CUDA) */
}

int dague_devices_freeze(dague_context_t* context)
{
    float total_sperf = 0.0, total_dperf = 0.0;
    (void)context;

    if(dague_devices_are_freezed)
        return -1;

    if(NULL != dague_device_load) free(dague_device_load);
    dague_device_load = (float*)calloc(dague_nb_devices, sizeof(float));
    if(NULL != dague_device_sweight) free(dague_device_sweight);
    dague_device_sweight = (float*)calloc(dague_nb_devices, sizeof(float));
    if(NULL != dague_device_dweight) free(dague_device_dweight);
    dague_device_dweight = (float*)calloc(dague_nb_devices, sizeof(float));
    for( uint32_t i = 0; i < dague_nb_devices; i++ ) {
        dague_device_t* device = dague_devices[i];
        if( NULL == device ) continue;
        dague_device_sweight[i] = device->device_sweight;
        total_sperf += device->device_sweight;
        dague_device_dweight[i] = device->device_dweight;
        total_dperf += device->device_dweight;
    }

    /* Compute the weight of each device including the cores */
    DEBUG(("Global Theoritical performance: single %2.4f double %2.4f\n", total_sperf, total_dperf));
    for( uint32_t i = 0; i < dague_nb_devices; i++ ) {
        DEBUG(("Dev[%d]             ->ratio single %2.4e double %2.4e\n",
               i, dague_device_sweight[i], dague_device_dweight[i]));

        dague_device_sweight[i] = (total_sperf / dague_device_sweight[i]);
        dague_device_dweight[i] = (total_dperf / dague_device_dweight[i]);
        /* after the weighting */
        DEBUG(("Dev[%d]             ->ratio single %2.4e double %2.4e\n",
               i, dague_device_sweight[i], dague_device_dweight[i]));
    }

    dague_devices_are_freezed = 1;
    return 0;
}

int dague_devices_freezed(dague_context_t* context)
{
    (void)context;
    return dague_devices_are_freezed;
}

int dague_devices_select(dague_context_t* context)
{
    (void)context;
#if defined(HAVE_CUDA)
    return dague_gpu_init(context);
#else
    return DAGUE_SUCCESS;
#endif  /* defined(HAVE_CUDA) */
}

int dague_devices_add(dague_context_t* context, dague_device_t* device)
{
    if( dague_devices_are_freezed ) {
        return DAGUE_ERROR;
    }
    if( NULL != device->context ) {
        /* This device already belong to a DAGuE context */
        return DAGUE_ERR_BAD_PARAM;
    }
    dague_atomic_lock(&dague_devices_mutex);  /* CRITICAL SECTION: BEGIN */
    if( (dague_nb_devices+1) > dague_nb_max_devices ) {
        if( NULL == dague_devices ) /* first time */
            dague_nb_max_devices = 4;
        else
            dague_nb_max_devices *= 2; /* every other time */
        dague_devices = realloc(dague_devices, dague_nb_max_devices * sizeof(dague_device_t*));
    }
    dague_devices[dague_nb_devices] = device;
    device->device_index = dague_nb_devices;
    dague_nb_devices++;
    device->context = context;
    dague_atomic_unlock(&dague_devices_mutex);  /* CRITICAL SECTION: END */
    return device->device_index;
}

dague_device_t* dague_devices_get(uint32_t device_index)
{
    if( device_index >= dague_nb_devices )
        return NULL;
    return dague_devices[device_index];
}

int dague_device_remove(dague_device_t* device)
{
    int rc = DAGUE_SUCCESS;

    dague_atomic_lock(&dague_devices_mutex);  /* CRITICAL SECTION: BEGIN */
    if( NULL == device->context ) {
        rc = DAGUE_ERR_BAD_PARAM;
        goto unlock_and_return_rc;
    }
    if(device != dague_devices[device->device_index]) {
        rc = DAGUE_ERR_NOT_FOUND;
        goto unlock_and_return_rc;
    }
    dague_devices[device->device_index] = NULL;
    device->context = NULL;
    device->device_index = -1;
  unlock_and_return_rc:
    dague_atomic_unlock(&dague_devices_mutex);  /* CRITICAL SECTION: BEGIN */
    return rc;
}
