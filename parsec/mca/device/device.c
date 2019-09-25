/*
 *
 * Copyright (c) 2013-2019 The University of Tennessee and The University
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
#include <dlfcn.h>
#include <sys/stat.h>
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

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
 * Temporary solution: Use the following two arrays to taskpool the weight and
 * the load on different devices. These arrays are not available before the
 * call to parsec_mca_device_freeze(). This is just a first step, a smarter approach
 * should take this spot.
 */
float *parsec_device_load = NULL;
float *parsec_device_hweight = NULL;
float *parsec_device_sweight = NULL;
float *parsec_device_dweight = NULL;
float *parsec_device_tweight = NULL;

int parsec_mca_device_init(void)
{
    char* parsec_device_list_str = NULL, **parsec_device_list = NULL;
    parsec_device_module_t **modules = NULL;
#if defined(PARSEC_PROF_TRACE)
    char modules_activated_str[1024] = "";
#endif  /* defined(PARSEC_PROF_TRACE) */
    int i, j, rc, priority;

    if( 0 < (rc = parsec_mca_param_find("device", NULL, NULL)) ) {
        parsec_output(0, "Use of MCA parameter device has been deprecated. Please migrate your MCA param file to use mca_device instead\n");
        parsec_mca_param_lookup_string(rc, &parsec_device_list_str);
    }
    (void)parsec_mca_param_reg_string_name("mca", "device",
                                           "Comma delimited list of devices to be enabled (or all)",
                                           false, false,
                                           NULL == parsec_device_list_str ? "all" : parsec_device_list_str, &parsec_device_list_str);

    (void)parsec_mca_param_reg_int_name("device", "show_capabilities",
                                        "Show the detailed devices capabilities",
                                        false, false, 0, NULL);
    (void)parsec_mca_param_reg_int_name("device", "show_statistics",
                                        "Show the detailed devices statistics upon exit",
                                        false, false, 0, NULL);
    (void)parsec_mca_param_reg_int_name("device", "verbose",
                                        "The level of verbosity of all operations related to devices",
                                        false, false, 0, &parsec_device_verbose);
    if( 0 < parsec_device_verbose ) {
        parsec_device_output = parsec_output_open(NULL);
        parsec_output_set_verbosity(parsec_device_output, parsec_device_verbose);
    }
    parsec_device_list = mca_components_get_user_selection("device");

    device_components = mca_components_open_bytype("device");
    for(i = 0; NULL != device_components[i]; i++); /* nothing just counting */
    if( 0 == i ) {  /* no devices */
        parsec_debug_verbose(10, parsec_debug_output, "No devices found on %s\n", parsec_hostname);
        return PARSEC_ERR_NOT_FOUND;
    }
    modules_activated = (parsec_device_module_t**)malloc(sizeof(parsec_device_module_t*) * i);
    modules_activated[0] = NULL;

    for(i = j = 0; NULL != device_components[i]; i++) {
        if( mca_components_belongs_to_user_list(parsec_device_list, device_components[i]->mca_component_name) ) {
            if (device_components[i]->mca_query_component != NULL) {
                rc = device_components[i]->mca_query_component((mca_base_module_t**)&modules, &priority);
                if( MCA_SUCCESS != rc ) {
                    parsec_debug_verbose(10, parsec_debug_output, "query function for component %s return no module",
                                         device_components[i]->mca_component_name);
                    device_components[i]->mca_close_component();
                    device_components[i] = NULL;
                    continue;
                }
                parsec_debug_verbose(10, parsec_debug_output, "query function for component %s[%d] returns priority %d",
                                     device_components[i]->mca_component_name, i, priority);
                if( NULL == modules ) {
                    parsec_debug_verbose(10, parsec_debug_output, "query function for component %s returns no modules. Remove.",
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
    parsec_debug_verbose(5, parsec_debug_output, "Found %d components, activated %d", i, num_modules_activated);

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

void parsec_mca_device_dump_and_reset_statistics(parsec_context_t* parsec_context)
{
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
    parsec_device_module_t *device;
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
        /* Update the context-level statistics */
        total              += device->executed_tasks;
        total_data_in      += device->transferred_data_in;
        total_data_out     += device->transferred_data_out;
        total_required_in  += device->required_data_in;
        total_required_out += device->required_data_out;

        device->executed_tasks       = 0;
        device->transferred_data_in  = 0;
        device->transferred_data_out = 0;
        device->required_data_in     = 0;
        device->required_data_out    = 0;
    }

    /* Print statistics */
    if( 0 == total_data_in )  total_data_in  = 1;
    if( 0 == total_data_out ) total_data_out = 1;
    gtotal = (float)total;

    printf("--------------------------------------------------------------------------------------------------\n");
    printf("|         |                    |         Data In                |         Data Out               |\n");
    printf("|Rank %3d |  # KERNEL |    %%   |  Required  |   Transfered(%%)   |  Required  |   Transfered(%%)   |\n",
           (NULL == parsec_context ? -1 : parsec_context->my_rank));
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
    return PARSEC_SUCCESS;
}

void*
parsec_device_find_function(const char* function_name,
                            const char* libname,
                            const char* paths[])
{
    char library_name[FILENAME_MAX];
    const char **target;
    void *dlh, *fn = NULL;

    for( target = paths; (NULL != target) && (NULL != *target); target++ ) {
        struct stat status;
        if( 0 != stat(*target, &status) ) {
            parsec_debug_verbose(10, parsec_device_output,
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
        dlh = dlopen(library_name, RTLD_NOW | RTLD_NODELETE );
        if(NULL == dlh) {
            parsec_debug_verbose(10, parsec_device_output,
                                 "Could not find %s dynamic library (%s)", library_name, dlerror());
            continue;
        }
        fn = dlsym(dlh, function_name);
        dlclose(dlh);
        if( NULL != fn ) {
            parsec_debug_verbose(4, parsec_device_output,
                                 "Function %s found in shared library %s",
                                 function_name, library_name);
            break;  /* we got one, stop here */
        }
    }
    /* Couldn't load from named dynamic libs, try linked/static */
    if(NULL == fn) {
        parsec_output_verbose(10, parsec_device_output,
                              "No dynamic function %s found, trying from compile time linked in\n",
                              function_name);
        dlh = dlopen(NULL, RTLD_NOW | RTLD_NODELETE);
        if(NULL != dlh) {
            fn = dlsym(dlh, function_name);
            if(NULL != fn) {
                parsec_debug_verbose(4, parsec_device_output,
                                     "Function %s found in the application symbols",
                                     function_name);
            }
            dlclose(dlh);
        }
    }
    if(NULL == fn) {
        parsec_debug_verbose(10, parsec_device_output,
                             "No function %s found", function_name);
    }
    return fn;
}

int parsec_mca_device_freeze(parsec_context_t* context)
{
    float total_hperf = 0.0, total_sperf = 0.0, total_dperf = 0.0, total_tperf = 0.0;
    (void)context;

    if(parsec_mca_device_are_freezed)
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
        parsec_device_module_t* device = parsec_devices[i];
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
    parsec_debug_verbose(4, parsec_device_output, "Global Theoretical performance: double %2.4f single %2.4f tensor %2.4f half %2.4f", total_dperf, total_sperf, total_tperf, total_hperf);
    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        parsec_debug_verbose(4, parsec_device_output, "  Dev[%d]             ->flops double %2.4f single %2.4f tensor %2.4f half %2.4f",
                             i, parsec_device_dweight[i], parsec_device_sweight[i], parsec_device_tweight[i], parsec_device_hweight[i]);

        parsec_device_hweight[i] = (total_hperf / parsec_device_hweight[i]);
        parsec_device_tweight[i] = (total_tperf / parsec_device_tweight[i]);
        parsec_device_sweight[i] = (total_sperf / parsec_device_sweight[i]);
        parsec_device_dweight[i] = (total_dperf / parsec_device_dweight[i]);
        /* after the weighting */
        parsec_debug_verbose(4, parsec_device_output, "  Dev[%d]             ->ratio double %2.4e single %2.4e tensor %2.4e half %2.4e",
                             i, parsec_device_dweight[i], parsec_device_sweight[i], parsec_device_tweight[i], parsec_device_hweight[i]);
    }

    parsec_mca_device_are_freezed = 1;
    return 0;
}

int parsec_mca_device_freezed(parsec_context_t* context)
{
    (void)context;
    return parsec_mca_device_are_freezed;
}

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static int cpu_weights(parsec_device_module_t* device, int nstreams) {
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
                             "Device %d (%s) disabled for taskpool %p", device->device_index, device->name, tp);
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

    /* By now let's add one device for the CPUs */
    {
        parsec_device_cpus = (parsec_device_module_t*)calloc(1, sizeof(parsec_device_module_t));
        parsec_device_cpus->name = "default";
        parsec_device_cpus->type = PARSEC_DEV_CPU;
        cpu_weights(parsec_device_cpus, nb_total_comp_threads);
        parsec_device_cpus->taskpool_register = device_taskpool_register_static;
        parsec_mca_device_add(context, parsec_device_cpus);
   }

    /* By now let's add one device for the recursive kernels */
    {
        parsec_device_recursive = (parsec_device_module_t*)calloc(1, sizeof(parsec_device_module_t));
        parsec_device_recursive->name = "recursive";
        parsec_device_recursive->type = PARSEC_DEV_RECURSIVE;
        parsec_device_recursive->device_hweight = parsec_device_cpus->device_hweight;
        parsec_device_recursive->device_tweight = parsec_device_cpus->device_tweight;
        parsec_device_recursive->device_sweight = parsec_device_cpus->device_sweight;
        parsec_device_recursive->device_dweight = parsec_device_cpus->device_dweight;
        parsec_device_recursive->taskpool_register = device_taskpool_register_static;
        parsec_mca_device_add(context, parsec_device_recursive);
    }

    for( int i = 0; NULL != (component = (parsec_device_base_component_t*)device_components[i]); i++ ) {
        for( int j = 0; NULL != (module = component->modules[j]); j++ ) {
            if (NULL == module->attach) {
                parsec_debug_verbose(10, parsec_debug_output, "A device module MUST contain an attach function. Disqualifying %s:%s module",
                                     component->base_version.mca_component_name, module->name);
                continue;
            }
            rc = module->attach(module, context);
            if( 0 > rc ) {
                parsec_debug_verbose(10, parsec_debug_output, "Attach failed for device %s:%s on context %p.",
                                     component->base_version.mca_component_name, module->name, context);
                continue;
            }
            parsec_debug_verbose(5, parsec_debug_output, "Activated DEVICE module %s:%s on context %p.",
                                 component->base_version.mca_component_name, module->name, context);
        }
    }
    return PARSEC_SUCCESS;
}

int parsec_mca_device_add(parsec_context_t* context, parsec_device_module_t* device)
{
    if( parsec_mca_device_are_freezed ) {
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
        parsec_devices = realloc(parsec_devices, parsec_nb_max_devices * sizeof(parsec_device_module_t*));
    }
    parsec_devices[parsec_nb_devices] = device;
    device->device_index = parsec_nb_devices;
    parsec_nb_devices++;
    device->context = context;
    parsec_atomic_unlock(&parsec_devices_mutex);  /* CRITICAL SECTION: END */
    return device->device_index;
}

parsec_device_module_t* parsec_mca_device_get(uint32_t device_index)
{
    if( device_index >= parsec_nb_devices )
        return NULL;
    return parsec_devices[device_index];
}

int parsec_mca_device_remove(parsec_device_module_t* device)
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
         * the memory related to the taskpoool might still be registered with the
         * devices we drop support. */
        tp->devices_index_mask &= ~(1 << device->device_index);
    }
    return;
}

int parsec_advise_data_on_device(parsec_data_t *data, int device, int advice)
{
    parsec_device_t *dev;
    dev = parsec_devices_get(device);
    if( NULL == dev )
        return PARSEC_ERR_NOT_FOUND;
    if( NULL == dev->device_data_advise )
        return PARSEC_SUCCESS;
    return dev->device_data_advise(dev, data, advice);
}

void parsec_devices_reset_load(parsec_context_t *context)
{
    int i;
    (void)context;
    if( NULL == parsec_device_load )
        return;
    for(i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_load[i] = 0;
    }
}

int parsec_devices_release_memory(void)
{
    int i;
    parsec_device_t *d;
    for(i = 1; i < (int)parsec_nb_devices; i++) {
        d = parsec_devices_get(i);
        if(NULL != d && NULL != d->device_memory_release) {
            d->device_memory_release(d);
        }
    }
    return PARSEC_SUCCESS;
}
