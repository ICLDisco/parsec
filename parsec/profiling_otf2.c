/*
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"

#include "parsec/profiling.h"
#include "parsec/data_distribution.h"
#include "parsec/utils/debug.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/os-spec-timing.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/mca_param.h"
#include "parsec/sys/tls.h"

#include <sys/stat.h>
#include <otf2/otf2.h>
#if MPI_VERSION < 3
#define OTF2_MPI_UINT64_T MPI_UNSIGNED_LONG
#define OTF2_MPI_INT64_T  MPI_LONG
#endif
#include <otf2/OTF2_MPI_Collectives.h>

#include <otf2/OTF2_Pthread_Locks.h>

#include <mpi.h>
static MPI_Comm parsec_otf2_profiling_comm;
static int parsec_profiling_mpi_on = 0;
static int process_id = 0;

/* where to start region IDs */
#define REGION_ID_OFFSET 2

#define STREAM_NAME_MAX 128

/* max id of profiling streams supported, runtime configurable
 * needed for file naming scheme: rank*max_stream_id+stream_id */
static int max_stream_id = 1000;

typedef struct {
    int                id;
    uint64_t           nb_evt;
    char               name[STREAM_NAME_MAX];
} parsec_profiling_stream_data_t;

struct parsec_profiling_stream_s {
    parsec_list_item_t super;
    parsec_list_t      informations;
    OTF2_EvtWriter    *evt_writer;
    parsec_profiling_stream_data_t data;
};

typedef struct {
    char *name;
    int   type;
    int   id;
} parsec_profiling_attribute_t;

typedef struct {
    int    otf2_region_id;
    char  *name;
    char  *alternative_name;
    char  *description;
    size_t info_length;
    int    otf2_nb_attributes;
    parsec_profiling_attribute_t *attr_info;
} parsec_profiling_region_t;

typedef struct {
    parsec_list_item_t super;
    char *key;
    char *value;
} parsec_profiling_info_t;

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_profiling_info_t);
PARSEC_OBJ_CLASS_INSTANCE(parsec_profiling_info_t, parsec_list_item_t,
                   NULL, NULL);

int parsec_profile_enabled = 0;

static PARSEC_TLS_DECLARE(tls_profiling);
static parsec_list_t threads;
static int __profile_initialized = 0;  /* not initialized */
static int __already_called = 0;
static parsec_time_t parsec_start_time;
static int          start_called = 0;
#define MAX_PROFILING_ERROR_STRING_LEN 1024
static char  parsec_profiling_last_error[MAX_PROFILING_ERROR_STRING_LEN+1] = { '\0', };
static int   parsec_profiling_raise_error = 0;
static parsec_list_t global_informations;

static parsec_profiling_region_t *regions = NULL;
static int nbregions                      = 0;
static int next_region                    = 0;
static int next_attr_id                   = 0;

static int32_t  thread_profiling_id = 0;

typedef struct {
    char *type_name;
    OTF2_Type type_desc;
} otf2_convertor_t;
static otf2_convertor_t otf2_convertor[] = {
    { "uint8_t", OTF2_TYPE_UINT8 },
    { "uint16_t", OTF2_TYPE_UINT16 },
    { "uint32_t", OTF2_TYPE_UINT32 },
    { "uint64_t", OTF2_TYPE_UINT64 },
    { "int8_t", OTF2_TYPE_INT8 },
    { "int16_t", OTF2_TYPE_INT16 },
    { "int32_t", OTF2_TYPE_INT32 },
    { "int", OTF2_TYPE_INT32 },
    { "int64_t", OTF2_TYPE_INT64 },
    { "float", OTF2_TYPE_FLOAT },
    { "double", OTF2_TYPE_DOUBLE },
};
static int nb_native_otf2_types = (int)sizeof(otf2_convertor)/sizeof(otf2_convertor_t);

static void set_last_error(const char *format, ...)
{
    va_list ap;
    int rc;
    va_start(ap, format);
    rc = vsnprintf(parsec_profiling_last_error, MAX_PROFILING_ERROR_STRING_LEN, format, ap);
    va_end(ap);
    parsec_warning("ParSEC profiling OTF2 -- Last error set to %s", parsec_profiling_last_error);
    parsec_profiling_raise_error = 1;
    (void)rc;
}

static OTF2_FlushType pre_flush(void*            userData,
                                OTF2_FileType    fileType,
                                OTF2_LocationRef location,
                                void*            callerData,
                                bool             final )
{
    (void)userData;
    (void)fileType;
    (void)location;
    (void)callerData;
    (void)final;
    return OTF2_FLUSH;
}

static OTF2_TimeStamp post_flush(void*            userData,
                                 OTF2_FileType    fileType,
                                 OTF2_LocationRef location )
{
    (void)userData;
    (void)fileType;
    (void)location;
    return (OTF2_TimeStamp)parsec_profiling_get_time();
}

static OTF2_FlushCallbacks flush_callbacks = {
    .otf2_pre_flush  = pre_flush,
    .otf2_post_flush = post_flush
};


static OTF2_Archive* otf2_archive = NULL;
static OTF2_GlobalDefWriter* global_def_writer = NULL;
static int32_t next_strid = 1;
static int emptystrid = 0;

static int next_otf2_global_strid(void)
{
    return parsec_atomic_fetch_inc_int32(&next_strid);
}

char *parsec_profiling_strerror(void)
{
    return parsec_profiling_last_error;
}

void parsec_profiling_add_information( const char *key, const char *value )
{
    parsec_profiling_info_t *new_info;
    char *c;

    if( !__profile_initialized ) return;

    new_info = PARSEC_OBJ_NEW(parsec_profiling_info_t);
    /* OTF2 format: keys must be in a namespace separated by ::,
     *              keys are in [a-zA-Z0-9_]+ */
    asprintf(&new_info->key, "PARSEC::%s", key);
    for(c = new_info->key; *c!='\0'; c++) {
        if( ! ( (*c>='a' && *c <= 'z') ||
                (*c>='A' && *c <= 'Z') ||
                (*c>='0' && *c <= '9') ||
                (*c == ':') ||
                (*c == '_') ) ) {
            *c = '_';
        }
    }
    /* OTF2 cannot handle empty property values so make them explicit */
    new_info->value = (0 == strlen(value)) ? strdup("<empty>") : strdup(value);
    parsec_list_push_back(&global_informations, &new_info->super);
}

void parsec_profiling_stream_add_information(parsec_profiling_stream_t* stream,
                                             const char *key, const char *value )
{
    char *info;
    asprintf(&info, "%s [Thread %d]", key, stream->data.id);
    parsec_profiling_add_information(info, value);
    free(info);
}

void parsec_profiling_otf2_set_comm(void *_pcomm)
{
    MPI_Comm *pcomm = (MPI_Comm*)_pcomm;
    (void)MPI_Initialized( &parsec_profiling_mpi_on );
    if( parsec_profiling_mpi_on ) {
        MPI_Comm_dup(*pcomm, &parsec_otf2_profiling_comm);
        MPI_Comm_rank(parsec_otf2_profiling_comm, &process_id);
    }
}

int parsec_profiling_init( int process_id )
{
    if( __profile_initialized ) return PARSEC_ERR_NOT_SUPPORTED;

    (void)process_id; /* OTF2 renames the processes according to their rank */

    PARSEC_TLS_KEY_CREATE(tls_profiling);

    PARSEC_OBJ_CONSTRUCT( &threads, parsec_list_t );
    PARSEC_OBJ_CONSTRUCT(&global_informations, parsec_list_t);

    parsec_mca_param_reg_int_name("profile", "max_streams", "Maximum number of profiling streams per process",
                                  false, false, max_stream_id, &max_stream_id);

    /* As we called the _start function automatically, the timing will be
     * based on this moment. By forcing back the __already_called to 0, we
     * allow the caller to decide when to rebase the timing in case there
     * is a need.
     */
    __already_called = 0;
    parsec_profile_enabled = 1;  /* turn on the profiling */

    /* add the hostname, for the sake of explicit profiling */
    char buf[HOST_NAME_MAX];
    if (0 == gethostname(buf, HOST_NAME_MAX))
        parsec_profiling_add_information("hostname", buf);
    else
        parsec_profiling_add_information("hostname", "");

    /* the current working directory may also be helpful */
    char * newcwd = NULL;
    int bufsize = HOST_NAME_MAX;
    errno = 0;
    char * cwd = getcwd(buf, bufsize);
    while (cwd == NULL && errno == ERANGE) {
        bufsize *= 2;
        cwd = realloc(cwd, bufsize);
        if (cwd == NULL)            /* failed  - just give up */
            break;
        errno = 0;
        newcwd = getcwd(cwd, bufsize);
        if (newcwd == NULL) {
            free(cwd);
            cwd = NULL;
        }
    }
    if (cwd != NULL) {
        parsec_profiling_add_information("cwd", cwd);
        if (cwd != buf)
            free(cwd);
    } else
        parsec_profiling_add_information("cwd", "");

    nbregions = 128;
    regions = (parsec_profiling_region_t*)malloc(sizeof(parsec_profiling_region_t) * nbregions);
    /* start regions with 2 to allow for positive and negative values, -1 is an invalid region */
    next_region = REGION_ID_OFFSET;

    if( !parsec_profiling_mpi_on ) {
        /* Nobody has called parsec_profiling_otf2_set_comm yet,
         * so we use MPI_COMM_WORLD by default */
        MPI_Comm comm = MPI_COMM_WORLD;
        parsec_profiling_otf2_set_comm( &comm );
    }

    __profile_initialized = 1; //* confirmed */
    return 0;
}

parsec_profiling_stream_t* parsec_profiling_stream_init( size_t length, const char *format, ...)
{
    parsec_profiling_stream_t* res;

    (void)length;

    if( !__profile_initialized ) return NULL;

    res = (parsec_profiling_stream_t*)calloc(sizeof(parsec_profiling_stream_t), 1);
    PARSEC_OBJ_CONSTRUCT(res, parsec_list_item_t);
    PARSEC_OBJ_CONSTRUCT(&res->informations, parsec_list_t);

    res->data.id = parsec_atomic_fetch_inc_int32(&thread_profiling_id);
    res->data.nb_evt = 0;
    res->evt_writer = NULL;

    if (res->data.id >= max_stream_id) {
        parsec_warning("More than %d profiling streams allocated, trace may become inconsistent!\n",
                       max_stream_id);
    }

    va_list ap;
    va_start(ap, format);
    vsnprintf(res->data.name, STREAM_NAME_MAX, format, ap);
    va_end(ap);


    if (start_called) {
        /* adjust for the communicator rank */
        res->data.id += process_id*max_stream_id;
        res->evt_writer = OTF2_Archive_GetEvtWriter( otf2_archive, res->data.id );
        if( NULL == res->evt_writer ) {
            parsec_warning("PaRSEC Profiling -- OTF2: could not allocate event writer for location %d\n", res->data.id);
        }
    }

    parsec_list_push_back( &threads, (parsec_list_item_t*)res );

    return res;
}

parsec_profiling_stream_t *parsec_profiling_set_default_thread( parsec_profiling_stream_t *new )
{
    parsec_profiling_stream_t *old;
    old = PARSEC_TLS_GET_SPECIFIC(tls_profiling);
    PARSEC_TLS_SET_SPECIFIC(tls_profiling, new);
    return old;
}

int parsec_profiling_dbp_start( const char *_basefile, const char *hr_info )
{
    char *archive_path, *archive_name, *c, *basefile;
    struct stat sb;
    OTF2_ErrorCode rc;
    char hostname[256];
    char *xmlbuffer;
    int buflen;

    if( !__profile_initialized ) return -1;

    basefile = strdup(_basefile);
    archive_name = NULL;
    for(c = basefile; *c != '\0'; c++) {
        if( *c == '/' ) {
            archive_name = c+1;
        }
    }
    if( NULL == archive_name ) {
        /* No '/' in basefile */
        archive_path = strdup(".");
        archive_name = basefile;
    } else {
        archive_path = basefile;
        *(archive_name-1) = '\0'; /* Cut basefile at last '/' */
        archive_name = strdup(archive_name); /* Get an independent copy of the archive_name */
    }
    basefile = NULL; /* It's either in archive_name or archive_path, so it's going to be freed later anyway */

    if (stat(archive_path, &sb) != 0 || !S_ISDIR(sb.st_mode)) {
        set_last_error("PaRSEC Profiling System: error -- '%s': directory not found", archive_path);
        free(archive_path);
        free(archive_name);
        return PARSEC_ERR_NOT_FOUND;
    }

    /* Reset the error system */
    snprintf(parsec_profiling_last_error, MAX_PROFILING_ERROR_STRING_LEN, "PaRSEC Profiling System: success");
    parsec_profiling_raise_error = 0;

    /* It's fine to re-reset the event date: we're back with a zero-length event set */
    start_called = 0;

    otf2_archive = OTF2_Archive_Open( archive_path,
                                      archive_name,
                                      OTF2_FILEMODE_WRITE,
                                      1024 * 1024 /* event chunk size */,
                                      4 * 1024 * 1024 /* def chunk size */,
                                      OTF2_SUBSTRATE_POSIX,
                                      OTF2_COMPRESSION_NONE );
    free(archive_path);
    free(archive_name);

    if( NULL == otf2_archive ) {
        set_last_error("PaRSEC Profiling System: OTF2 Error while creating archive");
        /* archive was not created, do not close it */
        return PARSEC_ERROR;
    }

    rc = OTF2_Pthread_Archive_SetLockingCallbacks(otf2_archive, NULL);
    if( OTF2_SUCCESS != rc ) {
        set_last_error("PaRSEC Profiling System: OTF2 error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        /* OTF2 seg faults if closing the archive at this time */
        otf2_archive = NULL;
        return -1;
    }

    rc = OTF2_Archive_SetFlushCallbacks( otf2_archive, &flush_callbacks, NULL );
    if( OTF2_SUCCESS != rc ) {
        set_last_error("PaRSEC Profiling System: OTF2 error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        /* OTF2 seg faults if closing the archive at this time */
        otf2_archive = NULL;
        return PARSEC_ERROR;
    }
    rc = OTF2_MPI_Archive_SetCollectiveCallbacks( otf2_archive,
                                                  parsec_otf2_profiling_comm,
                                                  MPI_COMM_NULL );
    if( OTF2_SUCCESS != rc ) {
        set_last_error("PaRSEC Profiling System: OTF2 error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        /* OTF2 seg faults if closing the archive at this time */
        otf2_archive = NULL;
        return PARSEC_ERROR;
    }
    rc = OTF2_Archive_OpenEvtFiles( otf2_archive );
    if( OTF2_SUCCESS != rc ) {
        set_last_error("PaRSEC Profiling System: OTF2 error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        OTF2_Archive_Close(otf2_archive);
        return PARSEC_ERROR;
    }

    if( process_id == 0 ) {
        global_def_writer = OTF2_Archive_GetGlobalDefWriter( otf2_archive );
        OTF2_GlobalDefWriter_WriteString(global_def_writer, emptystrid, "");
    }

    gethostname(hostname, 256);
    if( (rc = OTF2_Archive_SetMachineName(otf2_archive, hostname)) != OTF2_SUCCESS ) {
        set_last_error("PaRSEC Profiling System: error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        return PARSEC_ERROR;
    }

    if( (rc = OTF2_Archive_SetDescription(otf2_archive, hr_info)) != OTF2_SUCCESS ) {
        set_last_error("PaRSEC Profiling System: error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        return PARSEC_ERROR;
    }

    if( (rc = OTF2_Archive_SetCreator(otf2_archive, "PaRSEC Profiling System")) != OTF2_SUCCESS ) {
        set_last_error("PaRSEC Profiling System: error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        return PARSEC_ERROR;
    }

    if( parsec_hwloc_export_topology(&buflen, &xmlbuffer) != -1 &&
        buflen > 0 ) {
        parsec_profiling_add_information("HWLOC-XML", xmlbuffer);
        parsec_hwloc_free_xml_buffer(xmlbuffer);
    }

    return 0;
}

void parsec_profiling_start(void)
{
    parsec_list_item_t *r;
    parsec_profiling_stream_t *tp;

    if(start_called)
        return;

    if( NULL == otf2_archive )
        return;

    parsec_list_lock( &threads );
    for( r = PARSEC_LIST_ITERATOR_FIRST(&threads);
         r != PARSEC_LIST_ITERATOR_END(&threads);
         r = PARSEC_LIST_ITERATOR_NEXT(r) ) {
        tp = (parsec_profiling_stream_t*)r;
        /* adjust the id for the communicator rank */
        tp->data.id += process_id*max_stream_id;
        tp->evt_writer = OTF2_Archive_GetEvtWriter( otf2_archive, tp->data.id );
        if( NULL == tp->evt_writer ) {
            parsec_warning("PaRSEC Profiling -- OTF2: could not allocate event writer for location %d\n", tp->data.id);
        }
    }
    parsec_list_unlock( &threads );

    if( parsec_profiling_mpi_on ) {
        MPI_Barrier(parsec_otf2_profiling_comm);
    }

    start_called = 1;
    parsec_start_time = take_time();
}


int parsec_profiling_reset( void )
{
    return 0;
}

int parsec_profiling_add_dictionary_keyword( const char* key_name, const char* attributes,
                                             size_t info_length,
                                             const char* orig_convertor_code,
                                             int* key_start, int* key_end )
{
    int region;
    int rc;
    char *c;
    char *name, *type;
    int t;
    int strid;
    char *convertor_code = NULL;

    if( !__profile_initialized ) return 0;

    (void)attributes;

    if( next_region + 1 >= nbregions ) {
        nbregions += 128;
        regions = realloc(regions, sizeof(parsec_profiling_region_t) * nbregions);
    }
    region = next_region;
    next_region++;

    regions[region].otf2_region_id = region;
    regions[region].name = strdup(key_name);
    regions[region].alternative_name = strdup(key_name);
    regions[region].description = strdup("");
    regions[region].info_length = info_length;
    regions[region].attr_info = NULL;
    regions[region].otf2_nb_attributes = 0;

    if( NULL == orig_convertor_code ) {
        /* skip converter code parsing */
        goto malformed_convertor_code;
    }

    convertor_code = strdup(orig_convertor_code);
    c = convertor_code;
    regions[region].otf2_nb_attributes = 1;
    for(c = convertor_code; *c != '\0'; c++)
        if( *c == ';' )
            regions[region].otf2_nb_attributes++;

    while( *c != '\0') {
        while( *c != '{' ) {
            c++;
        }
        c++;
        for(t = 0; t < nb_native_otf2_types; t++) {
            if( strcmp(c, otf2_convertor[t].type_name) == 0 ) {
                regions[region].otf2_nb_attributes++;
                break;
            }
        }
        if(t == nb_native_otf2_types) {
            parsec_warning("parsec_profiling: in description of informations for dictionary entry '%s', unspecified type '%s' used in convertor code\n"
                           "  All informations for this key are going to be ignored.\n",
                           key_name, c, convertor_code);
            regions[region].otf2_nb_attributes = 0;
            goto malformed_convertor_code;
        }
        while( *c != ';' ) {
            if( *c == '\0' )
                break;
            c++;
        }
    }

    regions[region].attr_info = (parsec_profiling_attribute_t*)malloc(sizeof(parsec_profiling_attribute_t)
                                                                      * regions[region].otf2_nb_attributes);
    c = convertor_code;
    name = c;
    regions[region].otf2_nb_attributes = 0;
    while( *c != '\0') {
        while( *c != '{' ) {
            if( *c == ';' || *c == '\0' ) {
                parsec_warning("parsec_profiling: in description of informations for dictionary entry '%s', an invalid convertor code is used (at character %d of '%s')\n"
                               "  All informations for this key are going to be ignored.\n",
                               key_name, (int)((uintptr_t)c - (uintptr_t)convertor_code), convertor_code);
                regions[region].otf2_nb_attributes = 0;
                goto malformed_convertor_code;
            }
            c++;
        }
        *c++ = '\0'; /* Overwrite '{' into a '\0' so name is terminated */
        type = c;
        while( *c != '}' ) {
            if( *c == ';' || *c == '\0' ) {
                parsec_warning("parsec_profiling: in description of informations for dictionary entry '%s', an invalid convertor code is used (at character %d of '%s')\n"
                               "  All informations for this key are going to be ignored.\n",
                               key_name, (int)((uintptr_t)c - (uintptr_t)convertor_code), orig_convertor_code);
                regions[region].otf2_nb_attributes = 0;
                goto malformed_convertor_code;
            }
            c++;
        }
        *c++ = '\0'; /* Overwrite '}' into a '\0' so type is terminated */

        OTF2_Type otf2_type = 0;
        for(t = 0; t < nb_native_otf2_types; t++) {
            if( strcmp(type, otf2_convertor[t].type_name) == 0 ) {
                otf2_type = otf2_convertor[t].type_desc;
                break;
            }
        }

        bool attr_found = false;
        if (t < nb_native_otf2_types) {
            /**
             * check if this attribute has been defined already for another region
             */
            for (int r = REGION_ID_OFFSET; r < next_region; ++r) {
                for (int i = 0; i < regions[r].otf2_nb_attributes; ++i) {
                    if (0 == strcmp(regions[r].attr_info[i].name, name)) {
                        if (otf2_type != regions[r].attr_info[i].type) {
                            parsec_warning("parsec_profiling: found different types for attribute %s in dictionary entries %s and %s\n"
                                           "Your trace might not be visualized correctly in some tools",
                                           name, regions[region].name, regions[r].name);
                        } else {
                            regions[region].attr_info[regions[region].otf2_nb_attributes].type = otf2_type;
                            regions[region].attr_info[regions[region].otf2_nb_attributes].name = strdup(regions[r].name);
                            regions[region].attr_info[regions[region].otf2_nb_attributes].id   = regions[r].attr_info[i].id;
                            attr_found = true;
                        }
                        break;
                    }
                }
                if (attr_found) break;
            }

            if (!attr_found) {

                regions[region].attr_info[regions[region].otf2_nb_attributes].type = otf2_type;
                regions[region].attr_info[regions[region].otf2_nb_attributes].name = strdup(name);
                regions[region].attr_info[regions[region].otf2_nb_attributes].id   = next_attr_id++;

                if( NULL != global_def_writer ) {
                    strid = next_otf2_global_strid();
                    rc = OTF2_GlobalDefWriter_WriteString(global_def_writer,
                                                          strid,
                                                          name);
                    if(rc != OTF2_SUCCESS) {
                        parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
                    }
                    rc = OTF2_GlobalDefWriter_WriteAttribute(global_def_writer,
                                                             regions[region].attr_info[regions[region].otf2_nb_attributes].id,
                                                             strid,
                                                             emptystrid,
                                                             otf2_convertor[t].type_desc);
                    if(rc != OTF2_SUCCESS) {
                        parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
                    }
                }
            }
        } else {
            if( strncmp(type, "char[", 5) == 0 ) {
                /* We don't support fixed-size strings yet, so we just remember to skip the bytes */
                int nb = atoi(&type[5]);
                regions[region].attr_info[regions[region].otf2_nb_attributes].type = -nb;
            } else {
                parsec_warning("PaRSEC Profiling System: OTF2 Error -- Unrecognized type '%s' -- type size must be specified e.g. int32_t", type);
                regions[region].attr_info[regions[region].otf2_nb_attributes].type = 0;
            }
        }
        regions[region].otf2_nb_attributes++;
        if(*c == '\0')
            break;
        if(*c == ';') {
            c++;
            name=c;
            continue;
        }
        parsec_warning("parsec_profiling: in description of informations for dictionary entry '%s', an invalid convertor code is used (at character %d of '%s')\n"
                               "  All informations for this key are going to be ignored.\n",
                               key_name, (int)((uintptr_t)c - (uintptr_t)convertor_code), orig_convertor_code);
                regions[region].otf2_nb_attributes = 0;
                goto malformed_convertor_code;
    }

malformed_convertor_code:
    free(convertor_code);
    *key_start = region;
    *key_end   = -region;

    return 0;
}


int parsec_profiling_dictionary_flush( void )
{
    return 0;
}

int parsec_profiling_ts_trace_flags_info_fn(int key, uint64_t event_id, uint32_t taskpool_id,
                                            parsec_profiling_info_fn_t *info_fn, const void *info_data, uint16_t flags )
{
    parsec_profiling_stream_t* ctx;

    if( !start_called ) {
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    ctx = PARSEC_TLS_GET_SPECIFIC(tls_profiling);
    if( NULL != ctx )
        return parsec_profiling_trace_flags_info_fn(ctx, key, event_id, taskpool_id, info_fn, info_data, flags);

    set_last_error("Profiling system: error: called parsec_profiling_ts_trace_flags_info_fn"
                   " from a thread that did not call parsec_profiling_stream_init\n");
    return PARSEC_ERR_NOT_SUPPORTED;
}

int
parsec_profiling_trace_flags(parsec_profiling_stream_t* context, int key,
                            uint64_t event_id, uint32_t taskpool_id,
                            const void *info, uint16_t flags)
{
    return parsec_profiling_trace_flags_info_fn(context, key, event_id, taskpool_id, memcpy, info, flags);
}

int
parsec_profiling_trace_flags_info_fn(parsec_profiling_stream_t* context, int key,
                                     uint64_t event_id, uint32_t taskpool_id,
                                     parsec_profiling_info_fn_t *info_fn, const void *info_data, uint16_t flags)
{
    parsec_time_t now;
    int region;
    int rc = OTF2_SUCCESS;
    OTF2_AttributeList *attribute_list = NULL;
    uint64_t timestamp;

    (void)taskpool_id;
    (void)event_id;
    (void)flags;

    if( !start_called ) {
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    if( NULL == context->evt_writer ) {
        return PARSEC_ERR_BAD_PARAM;
    }

    if (key > -REGION_ID_OFFSET && key < REGION_ID_OFFSET) {
        /* -1 is an invalid key, silently ignore it */
        return PARSEC_ERR_BAD_PARAM;
    }

    now = take_time();
    timestamp = diff_time(parsec_start_time, now);

    region = key < 0 ? -key : key;

    if( (NULL != info_fn) && (NULL != info_data) ) {
        size_t info_length = regions[region].info_length;
        char *info = alloca(info_length);
        info_fn(info, info_data, info_length);
        attribute_list = OTF2_AttributeList_New();
        const char *ptr = info;
        for(int t = 0; t < regions[region].otf2_nb_attributes; t++) {
            if( regions[region].attr_info[t].type > 0 ) {
                switch( regions[region].attr_info[t].type ) {
                case  OTF2_TYPE_UINT8:
                    rc = OTF2_AttributeList_AddUint8(attribute_list, regions[region].attr_info[t].id, *(uint8_t*)ptr);
                    ptr += sizeof(uint8_t);
                    break;
                case OTF2_TYPE_UINT16:
                    rc = OTF2_AttributeList_AddUint16(attribute_list, regions[region].attr_info[t].id, *(uint16_t*)ptr);
                    ptr += sizeof(uint16_t);
                    break;
                case OTF2_TYPE_UINT32:
                    rc = OTF2_AttributeList_AddUint32(attribute_list, regions[region].attr_info[t].id, *(uint32_t*)ptr);
                    ptr += sizeof(uint32_t);
                    break;
                case OTF2_TYPE_UINT64:
                    rc = OTF2_AttributeList_AddUint64(attribute_list, regions[region].attr_info[t].id, *(uint64_t*)ptr);
                    ptr += sizeof(uint64_t);
                    break;
                case OTF2_TYPE_INT8:
                    rc = OTF2_AttributeList_AddInt8(attribute_list, regions[region].attr_info[t].id, *(int8_t*)ptr);
                    ptr += sizeof(int8_t);
                    break;
                case OTF2_TYPE_INT16:
                    rc = OTF2_AttributeList_AddInt16(attribute_list, regions[region].attr_info[t].id, *(int16_t*)ptr);
                    ptr += sizeof(int16_t);
                    break;
                case OTF2_TYPE_INT32:
                    rc = OTF2_AttributeList_AddInt32(attribute_list, regions[region].attr_info[t].id, *(int32_t*)ptr);
                    ptr += sizeof(int32_t);
                    break;
                case OTF2_TYPE_INT64:
                    rc = OTF2_AttributeList_AddInt64(attribute_list, regions[region].attr_info[t].id, *(int64_t*)ptr);
                    ptr += sizeof(int64_t);
                    break;
                case OTF2_TYPE_FLOAT:
                    rc = OTF2_AttributeList_AddFloat(attribute_list, regions[region].attr_info[t].id, *(float*)ptr);
                    ptr += sizeof(float);
                    break;
                case OTF2_TYPE_DOUBLE:
                    rc = OTF2_AttributeList_AddDouble(attribute_list, regions[region].attr_info[t].id, *(double*)ptr);
                    ptr += sizeof(double);
                    break;
                default:
                    parsec_warning("PaRSEC Profiling System: internal error, type %d unkown", regions[region].attr_info[t].type);
                    break;
                }
            } else {
                ptr += -regions[region].attr_info[t].type; /* Skip negative types: they are used to say how many bytes to skip */
            }
        }
    }

    region -= REGION_ID_OFFSET;
    if( key > 0 )
        rc = OTF2_EvtWriter_Enter( context->evt_writer,
                                   attribute_list,
                                   timestamp,
                                   region );
    else
        rc = OTF2_EvtWriter_Leave( context->evt_writer,
                                   attribute_list,
                                   timestamp,
                                  region );
    if(rc != OTF2_SUCCESS) {
        parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
    } else {
        context->data.nb_evt++;
    }
    if( NULL != attribute_list ) {
        rc = OTF2_AttributeList_Delete(attribute_list);
         if(rc != OTF2_SUCCESS) {
             parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
         }
    }

    return 0;
}

int parsec_profiling_dbp_dump( void )
{
    uint64_t epoch, gepoch;
    uint64_t *perlocation = NULL;
    uint64_t nb_local_threads = 0;
    parsec_list_item_t *r;
    int rc;
    int strid;
    char string[64];
    int comm_size = 1;

    if( NULL == otf2_archive )
        return PARSEC_ERR_NOT_SUPPORTED;

    if( !__profile_initialized ) return 0;

    epoch = parsec_profiling_get_time();

    int num_locations = thread_profiling_id;
    int total_num_locations = num_locations;
    parsec_profiling_stream_data_t *stream_data;

    if( parsec_profiling_mpi_on ) {
        MPI_Reduce(&num_locations, &total_num_locations, 1, MPI_INT, MPI_SUM, 0, parsec_otf2_profiling_comm);
    }
    stream_data = malloc(total_num_locations*sizeof(*stream_data));

    parsec_list_lock( &threads );
    for( r = PARSEC_LIST_ITERATOR_FIRST(&threads);
         r != PARSEC_LIST_ITERATOR_END(&threads);
         r = PARSEC_LIST_ITERATOR_NEXT(r) ) {
        parsec_profiling_stream_t *tp = (parsec_profiling_stream_t*)r;
        rc = OTF2_Archive_CloseEvtWriter( otf2_archive, tp->evt_writer );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }

        stream_data[nb_local_threads] = tp->data;

        nb_local_threads++;

        /* create a def file for this location */
        OTF2_DefWriter *def_writer = OTF2_Archive_GetDefWriter( otf2_archive, tp->data.id );
        if(NULL == def_writer ) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- could not open def_writer for location %d", tp->data.id);
        }
        rc = OTF2_Archive_CloseDefWriter( otf2_archive, def_writer );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }
    }
    assert((uint64_t)thread_profiling_id == nb_local_threads);
    parsec_list_unlock( &threads );


    if( parsec_profiling_mpi_on ) {
        MPI_Comm_size(parsec_otf2_profiling_comm, &comm_size);
    }

    perlocation = (uint64_t*)malloc( sizeof(uint64_t) * comm_size);

    if( parsec_profiling_mpi_on ) {
        MPI_Reduce( &epoch,
                    &gepoch,
                    1, OTF2_MPI_UINT64_T, MPI_MAX,
                    0, parsec_otf2_profiling_comm );
        MPI_Gather( &nb_local_threads, 1,
                    OTF2_MPI_UINT64_T,
                    perlocation, 1, OTF2_MPI_UINT64_T,
                    0, parsec_otf2_profiling_comm );

        int *recvcounts = NULL;
        int *displs = NULL;

        if( process_id == 0 ) {
            recvcounts = malloc(sizeof(int)*comm_size);
            displs = malloc(sizeof(int)*comm_size);
        }

        if (0 == process_id) {
            int displ = 0;
            for (int i = 0; i < comm_size; ++i) {
                displs[i] = displ;
                displ += perlocation[i]*sizeof(parsec_profiling_stream_data_t);
                recvcounts[i] = perlocation[i]*sizeof(parsec_profiling_stream_data_t);
            }
        }
        MPI_Gatherv(process_id == 0 ? MPI_IN_PLACE : stream_data,
                    nb_local_threads*sizeof(parsec_profiling_stream_data_t),
                    MPI_BYTE, stream_data, recvcounts, displs, MPI_BYTE, 0,
                    parsec_otf2_profiling_comm);
        free(recvcounts);
        free(displs);
    } else {
        gepoch = epoch;
        perlocation[0] = nb_local_threads;
    }


    if ( 0 == process_id ) {
        r = PARSEC_LIST_ITERATOR_FIRST(&global_informations);
        while( r != PARSEC_LIST_ITERATOR_END(&global_informations) ) {
            parsec_profiling_info_t *pi = (parsec_profiling_info_t*)r;
            if((rc = OTF2_Archive_SetProperty(otf2_archive, pi->key, pi->value, 1)) != OTF2_SUCCESS) {
                set_last_error("PaRSEC Profiling System: error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
            }
            free(pi->key);
            free(pi->value);
            r = PARSEC_LIST_ITERATOR_NEXT(r);
            PARSEC_OBJ_RELEASE(pi);
        }

        rc = OTF2_GlobalDefWriter_WriteClockProperties( global_def_writer,
                                                        1000000000,
                                                        0, gepoch + 1);
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }

        /* Define the dictionary */
        for(int r = REGION_ID_OFFSET; r < next_region; r++) {
            int nameid = next_otf2_global_strid();
            rc = OTF2_GlobalDefWriter_WriteString( global_def_writer, nameid, regions[r].name );
            if(rc != OTF2_SUCCESS) {
                parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
            }
            int altnameid = next_otf2_global_strid();
            rc = OTF2_GlobalDefWriter_WriteString( global_def_writer, altnameid, regions[r].alternative_name );
            if(rc != OTF2_SUCCESS) {
                parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
            }
            int descid = next_otf2_global_strid();
            rc = OTF2_GlobalDefWriter_WriteString( global_def_writer, descid, regions[r].description );
            if(rc != OTF2_SUCCESS) {
                parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
            }

            rc = OTF2_GlobalDefWriter_WriteRegion( global_def_writer,
                                                   regions[r].otf2_region_id - REGION_ID_OFFSET /* id */,
                                                   nameid /* region name  */,
                                                   altnameid /* alternative name */,
                                                   descid /* description */,
                                                   OTF2_REGION_ROLE_FUNCTION,
                                                   OTF2_PARADIGM_NONE,
                                                   OTF2_REGION_FLAG_NONE,
                                                   emptystrid /* source file */,
                                                   0 /* begin lno */,
                                                   0 /* end lno */ );
            if(rc != OTF2_SUCCESS) {
                parsec_warning("PaRSEC Profiling System: OTF2 Error in write region -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
            }
        }

        /* Write the system tree into the global definitions */
        strid = next_otf2_global_strid();
        char hostname[256];
        gethostname(hostname, 256);
        rc = OTF2_GlobalDefWriter_WriteString( global_def_writer, strid, hostname );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }

        int strid2 = next_otf2_global_strid();
        rc = OTF2_GlobalDefWriter_WriteString( global_def_writer, strid2, "Node" );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }
        rc = OTF2_GlobalDefWriter_WriteSystemTreeNode( global_def_writer,
                                                       0 /* Node id */,
                                                       strid /* name */,
                                                       strid2 /* class */,
                                                       OTF2_UNDEFINED_SYSTEM_TREE_NODE /* parent */ );

        int stream_id = 0;

        strid = next_otf2_global_strid();
        rc = OTF2_GlobalDefWriter_WriteString( global_def_writer, strid, "Binding" );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }
        int bound_to_strid = strid;
        for(int r = 0; r < comm_size; r++) {
            char pname[64];
            strid = next_otf2_global_strid();
            snprintf(pname, 64, "MPI Rank %d", r);
            rc = OTF2_GlobalDefWriter_WriteString( global_def_writer, strid, pname );
            if(rc != OTF2_SUCCESS) {
                parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
            }
            rc = OTF2_GlobalDefWriter_WriteLocationGroup( global_def_writer,
                                                          r,
                                                          strid,
                                                          OTF2_LOCATION_GROUP_TYPE_PROCESS,
                                                          0 /* system tree */ );

            for (uint64_t i = 0; i < perlocation[r]; ++i) {

                /* store the "Bound ..." part in a property */
                char *bound_ptr = strstr(stream_data[stream_id].name, "Bound");
                if (NULL != bound_ptr) {
                    /* cut out the "Bound on" part and only store the hex number */
                    *(bound_ptr-1) = '\0';
                    bound_ptr = strstr(bound_ptr, "0x");
                }

                strid = next_otf2_global_strid();
                rc = OTF2_GlobalDefWriter_WriteString(global_def_writer, strid, stream_data[stream_id].name);
                if(rc != OTF2_SUCCESS) {
                    parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
                }

                OTF2_LocationType loctype = OTF2_LOCATION_TYPE_CPU_THREAD;
                if (NULL != strstr(stream_data[stream_id].name, "GPU")) {
                    loctype = OTF2_LOCATION_TYPE_GPU;
                }

                rc = OTF2_GlobalDefWriter_WriteLocation(global_def_writer, stream_data[stream_id].id, strid,
                                                        loctype, stream_data[stream_id].nb_evt, r);
                if(rc != OTF2_SUCCESS) {
                    parsec_warning("PaRSEC Profiling System: OTF2 Error in write region -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
                }

                if (NULL != bound_ptr) {

                    strid = next_otf2_global_strid();
                    rc = OTF2_GlobalDefWriter_WriteString(global_def_writer, strid, bound_ptr);
                    if(rc != OTF2_SUCCESS) {
                        parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
                    }

                    OTF2_AttributeValue val;
                    val.stringRef = strid;
                    rc = OTF2_GlobalDefWriter_WriteLocationProperty(global_def_writer, stream_data[stream_id].id, bound_to_strid, OTF2_TYPE_STRING, val);
                    if(rc != OTF2_SUCCESS) {
                        parsec_warning("PaRSEC Profiling System: OTF2 Error in write region -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
                    }
                }

                ++stream_id;
            }
        }

        /* Now we need to define the MPI communicator
         *  - First, what is the universe */
        int id = 0;
        for(int r = 0; r < comm_size; r++) {
            perlocation[id++] = r*max_stream_id;
        }
        strid = next_otf2_global_strid();
        snprintf(string, 64, "MPI");
        rc = OTF2_GlobalDefWriter_WriteString( global_def_writer, strid, string );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }
        rc = OTF2_GlobalDefWriter_WriteGroup( global_def_writer,
                                              0 /* Group id */,
                                              strid /* name */,
                                              OTF2_GROUP_TYPE_COMM_LOCATIONS,
                                              OTF2_PARADIGM_MPI,
                                              OTF2_GROUP_FLAG_NONE,
                                              comm_size,
                                              perlocation );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }
        /*  - Second, what is the group that contains parsec_otf2_profiling_comm */
        rc = OTF2_GlobalDefWriter_WriteGroup( global_def_writer,
                                              1 /* Group id */,
                                              emptystrid /* name */,
                                              OTF2_GROUP_TYPE_COMM_GROUP,
                                              OTF2_PARADIGM_MPI,
                                              OTF2_GROUP_FLAG_NONE,
                                              comm_size,
                                              perlocation );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }
        /*  - And third, define the communicator above that group */
        strid = next_otf2_global_strid();
        snprintf(string, 64, "MPI_COMM_WORLD");
        rc = OTF2_GlobalDefWriter_WriteString( global_def_writer, strid, string );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }
        rc = OTF2_GlobalDefWriter_WriteComm( global_def_writer,
                                             0 /* Communicator id */,
                                             strid /* name */,
                                             1 /* group */,
                                             OTF2_UNDEFINED_COMM /* parent */ );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }

        /* Finally done: close the global definition writer */
        rc = OTF2_Archive_CloseGlobalDefWriter( otf2_archive, global_def_writer );
        if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
        }
    }
    free(perlocation);
    free(stream_data);

    MPI_Barrier(parsec_otf2_profiling_comm); /* All the ranks must wait here that the rank 0 has written everything */

    rc = OTF2_Archive_Close( otf2_archive );
    if(rc != OTF2_SUCCESS) {
            parsec_warning("PaRSEC Profiling System: OTF2 Error -- %s (%s)", OTF2_Error_GetName(rc), OTF2_Error_GetDescription(rc));
    }
    otf2_archive = NULL;

    if( parsec_profiling_raise_error )
        return PARSEC_ERROR;

    return 0;
}

uint64_t parsec_profiling_get_time(void) {
    return diff_time(parsec_start_time, take_time());
}

void parsec_profiling_enable(void)
{
    parsec_profile_enabled = 1;
}

void parsec_profiling_disable(void)
{
    parsec_profile_enabled = 0;
}

void profiling_save_dinfo(const char *key, double value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%g", value);
    (void)rv;
    parsec_profiling_add_information(key, svalue);
    free(svalue);
}

void profiling_save_iinfo(const char *key, int value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%d", value);
    (void)rv;
    parsec_profiling_add_information(key, svalue);
    free(svalue);
}

void profiling_save_uint64info(const char *key, unsigned long long int value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%llu", value);
    (void)rv;
    parsec_profiling_add_information(key, svalue);
    free(svalue);
}

void profiling_save_sinfo(const char *key, char* svalue)
{
    parsec_profiling_add_information(key, svalue);
}

void profiling_stream_save_dinfo(parsec_profiling_stream_t* stream,
                                 const char *key, double value)
{
    char *svalue;
    int rv = asprintf(&svalue, "%g", value);
    (void)rv;
    parsec_profiling_stream_add_information(stream, key, svalue);
    free(svalue);
}

void profiling_stream_save_iinfo(parsec_profiling_stream_t* stream,
                                 const char *key, int value)
{
    char *svalue;
    int rv = asprintf(&svalue, "%d", value);
    (void)rv;
    parsec_profiling_stream_add_information(stream, key, svalue);
    free(svalue);
}

void profiling_stream_save_uint64info(parsec_profiling_stream_t* stream,
                                      const char *key, unsigned long long int value)
{
    char *svalue;
    int rv = asprintf(&svalue, "%llu", value);
    (void)rv;
    parsec_profiling_stream_add_information(stream, key, svalue);
    free(svalue);
}

void profiling_stream_save_sinfo(parsec_profiling_stream_t* stream,
                                 const char *key, char* svalue)
{
    parsec_profiling_stream_add_information(stream, key, svalue);
}

int parsec_profiling_fini( void )
{
    parsec_profiling_stream_t *t;

    if( !__profile_initialized ) return PARSEC_ERR_NOT_SUPPORTED;

    if( 0 != parsec_profiling_dbp_dump() ) {
        return PARSEC_ERROR;
    }

    while( (t = (parsec_profiling_stream_t*)parsec_list_nolock_pop_front(&threads)) ) {
        free(t);
    }
    PARSEC_OBJ_DESTRUCT(&threads);

    parsec_profiling_dictionary_flush();
    start_called = 0;  /* Allow the profiling to be reinitialized */
    parsec_profile_enabled = 0;  /* turn off the profiling */
    __profile_initialized = 0;  /* not initialized */

    MPI_Comm_free(&parsec_otf2_profiling_comm);
    parsec_profiling_mpi_on = 0;

    return 0;
}
