/*
 *
 * Copyright (c) 2013-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/devices/device.h"
#include "parsec/utils/mca_param.h"
#include "parsec/constants.h"
#include "parsec/utils/debug.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/argv.h"
#include "parsec/data_internal.h"
#include <stdlib.h>

#include <parsec/devices/cuda/dev_cuda.h>
#include <parsec/devices/openmp/dev_omp.h>

uint32_t parsec_nb_devices = 0;
static uint32_t parsec_nb_max_devices = 0;
static uint32_t parsec_devices_are_freezed = 0;
parsec_atomic_lock_t parsec_devices_mutex = { PARSEC_ATOMIC_UNLOCKED };
static parsec_device_t** parsec_devices = NULL;
static char* parsec_device_list_str = NULL, **parsec_device_list = NULL;

static parsec_device_t* parsec_device_cpus = NULL;
static parsec_device_t* parsec_device_recursive = NULL;

/**
 * Temporary solution: Use the following two arrays to taskpool the weight and
 * the load on different devices. These arrays are not available before the
 * call to parsec_devices_freeze(). This is just a first step, a smarter approach
 * should take this spot.
 */
float *parsec_device_load = NULL;
float *parsec_device_hweight = NULL;
float *parsec_device_sweight = NULL;
float *parsec_device_dweight = NULL;
float *parsec_device_tweight = NULL;

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
    if(NULL != parsec_device_hweight) free(parsec_device_hweight);
    parsec_device_hweight = NULL;
    if(NULL != parsec_device_sweight) free(parsec_device_sweight);
    parsec_device_sweight = NULL;
    if(NULL != parsec_device_dweight) free(parsec_device_dweight);
    parsec_device_dweight = NULL;
    if(NULL != parsec_device_tweight) free(parsec_device_tweight);
    parsec_device_tweight = NULL;

#if defined(PARSEC_HAVE_CUDA)
    (void)parsec_cuda_fini();
#endif  /* defined(PARSEC_HAVE_CUDA) */

#if defined(PARSEC_HAVE_OPENMP)

    (void)parsec_omp_fini();
#endif /* defined(PARSEC_HAVE_OPENMP) */

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
    float total_hperf = 0.0, total_sperf = 0.0, total_dperf = 0.0, total_tperf = 0.0;
    (void)context;

    if(parsec_devices_are_freezed)
        return -1;

    if(NULL != parsec_device_load) free(parsec_device_load);
    parsec_device_load = (float*)calloc(parsec_nb_devices, sizeof(float));
    if(NULL != parsec_device_hweight) free(parsec_device_hweight);
    parsec_device_hweight = (float*)calloc(parsec_nb_devices, sizeof(float));
    if(NULL != parsec_device_sweight) free(parsec_device_sweight);
    parsec_device_sweight = (float*)calloc(parsec_nb_devices, sizeof(float));
    if(NULL != parsec_device_dweight) free(parsec_device_dweight);
    parsec_device_dweight = (float*)calloc(parsec_nb_devices, sizeof(float));
    if(NULL != parsec_device_tweight) free(parsec_device_tweight);
    parsec_device_tweight = (float*)calloc(parsec_nb_devices, sizeof(float));
    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        parsec_device_t* device = parsec_devices[i];
        if( NULL == device ) continue;
        parsec_device_hweight[i] = device->device_hweight;
        parsec_device_sweight[i] = device->device_sweight;
        parsec_device_dweight[i] = device->device_dweight;
        parsec_device_tweight[i] = device->device_tweight;
        if( PARSEC_DEV_RECURSIVE == device->type ) continue;
        total_hperf += device->device_hweight;
        total_tperf += device->device_tweight;
        total_sperf += device->device_sweight;
        total_dperf += device->device_dweight;
    }

    /* Compute the weight of each device including the cores */
    parsec_debug_verbose(4, parsec_debug_output, "Global Theoretical performance: double %2.4f single %2.4f tensor %2.4f half %2.4f", total_dperf, total_sperf, total_tperf, total_hperf);
    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        parsec_debug_verbose(4, parsec_debug_output, "  Dev[%d]             ->flops double %2.4f single %2.4f tensor %2.4f half %2.4f",
                             i, parsec_device_dweight[i], parsec_device_sweight[i], parsec_device_tweight[i], parsec_device_hweight[i]);

        parsec_device_hweight[i] = (total_hperf / parsec_device_hweight[i]);
        parsec_device_tweight[i] = (total_tperf / parsec_device_tweight[i]);
        parsec_device_sweight[i] = (total_sperf / parsec_device_sweight[i]);
        parsec_device_dweight[i] = (total_dperf / parsec_device_dweight[i]);
        /* after the weighting */
        parsec_debug_verbose(4, parsec_debug_output, "  Dev[%d]             ->ratio double %2.4e single %2.4e tensor %2.4e half %2.4e",
                             i, parsec_device_dweight[i], parsec_device_sweight[i], parsec_device_tweight[i], parsec_device_hweight[i]);
    }

    parsec_devices_are_freezed = 1;
    return 0;
}

int parsec_devices_freezed(parsec_context_t* context)
{
    (void)context;
    return parsec_devices_are_freezed;
}

int parsec_devices_best_load( parsec_task_t* this_task, double ratio )
{
    int i, dev_index = -1, data_index = 0;
    parsec_taskpool_t* tp = this_task->taskpool;

    /* Step one: Find the first data in WRITE mode stored on a GPU */
    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        if( (NULL != this_task->task_class->out[i]) &&
            (this_task->task_class->out[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            data_index = this_task->task_class->out[i]->flow_index;
            dev_index  = this_task->data[data_index].data_in->original->owner_device;
            if (dev_index > 1) {
                break;
            }
        }
    }
    assert(dev_index >= 0);

    /* 0 is CPU, and 1 is recursive device */
    if( dev_index <= 1 ) {  /* This is the first time we see this data for a GPU.
                             * Let's decide which GPU will work on it. */
        int best_index = 0;  /* default value: first CPU device */
        float weight, best_weight = parsec_device_load[0] + ratio * parsec_device_sweight[0];

        /* Start at 2, to skip the recursive body */
        for( dev_index = 2; dev_index < parsec_devices_enabled(); dev_index++ ) {
            /* Skip the device if it is not configured */
            if(!(tp->devices_mask & (1 << dev_index))) continue;
            weight = parsec_device_load[dev_index] + ratio * parsec_device_sweight[dev_index];
            if( best_weight > weight ) {
                best_index = dev_index;
                best_weight = weight;
            }
        }
        parsec_device_load[best_index] += ratio * parsec_device_sweight[best_index];
        assert( best_index != 1 );
        dev_index = best_index;
    }

    return dev_index;
}

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif
#if defined(PARSEC_HAVE_ERRNO_H)
#include <errno.h>
#endif

static int cpu_weights(parsec_device_t* device, int nstreams) {
    /* This is default value when it cannot be computed */
    /* Crude estimate that holds for Nehalem era Xeon processors */
    float freq = 2.5f;
    float fp_ipc = 8.f;
    float dp_ipc = 4.f;
    char cpu_model[256]="Unkown";
    char cpu_flags[256]="";

#if defined(__linux__)
    FILE* procinfo = fopen("/proc/cpuinfo", "r");
    if( NULL == procinfo ) {
        parsec_warning("CPU Features cannot be autodetected on this machine: %s", strerror(errno));
        goto notfound;
    }
    char str[256];
    while( NULL != fgets(str, 256, procinfo) ) {
        /* Intel/AMD */
        sscanf(str, "model name : %256[^\n]%*c", cpu_model);
        if( 0 != sscanf(str, "cpu MHz : %f", &freq) )
            freq *= 1e-3;
        if( 0 != sscanf(str, "flags : %256[^\n]%*c", cpu_flags) )
            break; /* done reading for an x86 type CPU */
        /* IBM: Power */
        sscanf(str, "cpu : %256[^\n]%*c", cpu_model);
        if( 0 != sscanf(str, "clock : %fMHz", &freq) ) {
            freq *= 1e-3;
            break; /* done reading for a Power type CPU */
        }
    }
    fclose(procinfo);
#elif defined(__APPLE__)
    size_t len = 256;
    int rc = sysctlbyname("machdep.cpu.brand_string", cpu_model, &len, NULL, 0);
    if( rc ) {
        parsec_warning("CPU Features cannot be autodetected on this machine (Detected OSX): %s", strerror(errno));
        goto notfound;
    }
    rc = sysctlbyname("machdep.cpu.features", cpu_flags, &len, NULL, 0);
    if( rc ) {
        parsec_warning("CPU Features cannot be autodetected on this machine (Detected OSX): %s", strerror(errno));
        goto notfound;
    }
#endif
    /* prefer base frequency from model name when available (avoids power
     * saving modes and dynamic frequency scaling issues) */
    sscanf(cpu_model, "%*[^@] @ %fGHz", &freq);

#if defined(PARSEC_HAVE_BUILTIN_CPU)
    __builtin_cpu_init();
#if defined(PARSEC_HAVE_BUILTIN_CPU512)
    if(__builtin_cpu_supports("avx512f")) {
        fp_ipc = 64;
        dp_ipc = 32;
    } else
#endif /* PARSEC_HAVE_BUILTIN_CPU512; */
         if(__builtin_cpu_supports("avx2")) {
        fp_ipc = 32;
        dp_ipc = 16;
    }
    else if(__builtin_cpu_supports("avx")) {
        fp_ipc = 16;
        dp_ipc = 8;
    }
    else {
        fp_ipc = 8;
        dp_ipc = 4;
    }
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
    else {
        fp_ipc = 8;
        dp_ipc = 4;
    }
#endif


    {
      int show_caps = 0;
      int show_caps_index = parsec_mca_param_find("device", NULL, "show_capabilities");
      if(0 < show_caps_index) {
          parsec_mca_param_lookup_int(show_caps_index, &show_caps);
      }
      if( show_caps ) {
          parsec_inform("CPU Device: %s\n"
                        "\tParsec Streams     : %d\n"
                        "\tclockRate (GHz)    : %2.2f\n"
                        "\tpeak Gflops        : double %2.4f, single %2.4f",
                        cpu_model,
                        nstreams,
                        freq, nstreams*freq*dp_ipc, nstreams*freq*fp_ipc);
       }
    }
 notfound:

    device->device_hweight = nstreams * fp_ipc * freq; /* No processor have half precision for now */
    device->device_tweight = nstreams * fp_ipc * freq; /* No processor support tensor operations for now */
    device->device_sweight = nstreams * fp_ipc * freq;
    device->device_dweight = nstreams * dp_ipc * freq;

    return PARSEC_SUCCESS;
}

int parsec_devices_select(parsec_context_t* context)
{
    int rc = PARSEC_SUCCESS;
    int nb_total_comp_threads = 0;

    for(int p = 0; p < context->nb_vp; p++) {
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;
    }

    /* By now let's add one device for the CPUs */
    {
        parsec_device_cpus = (parsec_device_t*)calloc(1, sizeof(parsec_device_t));
        parsec_device_cpus->name = "default";
        parsec_device_cpus->type = PARSEC_DEV_CPU;
        cpu_weights(parsec_device_cpus, nb_total_comp_threads);
        parsec_devices_add(context, parsec_device_cpus);
   }

    /* By now let's add one device for the recursive kernels */
    {
        parsec_device_recursive = (parsec_device_t*)calloc(1, sizeof(parsec_device_t));
        parsec_device_recursive->name = "recursive";
        parsec_device_recursive->type = PARSEC_DEV_RECURSIVE;
        parsec_device_recursive->device_hweight = parsec_device_cpus->device_hweight;
        parsec_device_recursive->device_tweight = parsec_device_cpus->device_tweight;
        parsec_device_recursive->device_sweight = parsec_device_cpus->device_sweight;
        parsec_device_recursive->device_dweight = parsec_device_cpus->device_dweight;
        parsec_devices_add(context, parsec_device_recursive);
    }

#if defined(PARSEC_HAVE_OPENMP)
    rc = parsec_omp_init(context);
    if( PARSEC_SUCCESS != rc ) return rc;
#endif /* define(PARSEC_HAVE_OPENMP) */

#if defined(PARSEC_HAVE_CUDA)
    rc = parsec_cuda_init(context);
    if( PARSEC_SUCCESS != rc ) return rc;
#endif  /* defined(PARSEC_HAVE_CUDA) */
    return PARSEC_SUCCESS;
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


void parsec_devices_taskpool_restrict(parsec_taskpool_t *tp,
                                      uint8_t            devices_type)
{
    parsec_device_t *device;
    uint32_t i;

    for (i = 0; i < parsec_nb_devices; i++) {
	if (!(tp->devices_mask & (1 << i)))
	    continue;

	device = parsec_devices_get(i);
	if ((NULL == device) || (device->type & devices_type))
	    continue;

        /* Disable this type of device */
        tp->devices_mask &= ~(1 << i);
    }
    return;
}

