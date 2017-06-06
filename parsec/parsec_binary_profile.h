/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dbp_h
#define _dbp_h

/**
 * @defgroup parsec_internal_profiling Tracing System
 * @ingroup parsec_internal
 * @brief The PaRSEC profiling system allows to expose information
 *   about the DAG and the runtime engine for analysis.
 * @addtogroup parsec_internal_profiling
 * @{
 */

#include <inttypes.h>
#include <pthread.h>

#include "parsec/class/list.h"

typedef struct parsec_profiling_output_base_event_s {
    uint16_t  key;
    uint16_t  flags;
    uint32_t  handle_id;
    uint64_t  event_id;
    uint64_t  timestamp;
} parsec_profiling_output_base_event_t;

typedef struct parsec_profiling_output_s {
    parsec_profiling_output_base_event_t event;
    char                                info[1];
} parsec_profiling_output_t;

#define PROFILING_BUFFER_TYPE_EVENTS      1
#define PROFILING_BUFFER_TYPE_DICTIONARY  2
#define PROFILING_BUFFER_TYPE_THREAD      3
#define PROFILING_BUFFER_TYPE_GLOBAL_INFO 4
#define PROFILING_BUFFER_TYPE_HEADER      5
typedef struct parsec_profiling_buffer_s {
    off_t    this_buffer_file_offset;    /* Used by the malloc / write method. MUST BE THE FIRST ELEMENT */
    off_t    next_buffer_file_offset;
    union {
        int64_t  nb_events;              /* Used by BUFFER_TYPE_EVENTS     */
        int64_t  nb_dictionary_entries;  /* Used by BUFFER_TYPE_DICTIONARY */
        int64_t  nb_infos;               /* Used by BUFFER_TYPE_GLOBAL_INFO*/
        int64_t  nb_threads;             /* Used by BUFFER_TYPE_THREAD */
    } this_buffer;
    char     buffer_type;
    char     buffer[1];
} parsec_profiling_buffer_t;

typedef struct {
    int32_t info_size;                 /* Number of bytes in this structure for the info */
    int32_t value_size;                /* Number of bytes in this structure for the value 
                                        * Note: info_size + value_size must be <= profiling_buffer_available_bytes */
    char info_and_value[1];            /* Bytes for info and value. */
} parsec_profiling_info_buffer_t;

typedef struct {
    char    name[64];                  /* Name of the key */
    char    attributes[128];           /* Attributes for that key (will be added to the XML) */
    int32_t keyinfo_convertor_length;  /* Number of bytes in the convertor below */
    int32_t keyinfo_length;            /* Number of bytes to reserve for this key info */
    char    convertor[1];              /* Follow: convertor bytes. */
} parsec_profiling_key_buffer_t;

typedef struct {
    int64_t  next_thread_offset;         /* Offset of the next thread buffer (-1 if last thread) */
    uint64_t nb_events;                  /* Number of events captured by this thread */
    char     hr_id[128];                 /* Unique ID of this thread */
    int64_t  first_events_buffer_offset; /* Offset of the first events buffer for this thread */
    int32_t  nb_infos;                   /* Number of infos that follow in this thread */
    parsec_profiling_info_buffer_t infos[1];/* First profiling_info_buffer for this thread */
} parsec_profiling_thread_buffer_t;

/**
 * Structure of a PaRSEC Binary Profile:
 */
typedef struct {
    int64_t  this_buffer_file_offset;  /* Must be 0. Used by the malloc / write method. MUST BE THE FIRST ELEMENT */
    char    magick[25];          /* Must be "#PARSEC BINARY PROFILE " */
    int64_t byte_order;          /* The writer put 0x0123456789ABCDEF */
    int32_t profile_buffer_size; /* Size of profile_*_buffers */
    char    hr_id[128];          /* 128 bytes to identify the application "uniquely" */
    int32_t dictionary_size;     /* Number of dictionary entries */
    int64_t dictionary_offset;   /* Offset of the first dictionary profiling_buffer */
    int32_t info_size;           /* Number of global info entries */
    int64_t info_offset;         /* Offset of the first info profiling_buffer */
    int32_t rank;                /* Rank of the process that generated this profile */
    int32_t worldsize;           /* Worldsize of the MPI application that generated this profile */
    int32_t nb_threads;          /* Number of threads in this profile */
    int64_t thread_offset;       /* Offset of the first thread profiling_buffer */
    /* Padding to align on profile_buffer_size -- required to allow for mmaping of buffers */
} parsec_profiling_binary_file_header_t;

typedef struct parsec_profiling_info_s {
    struct parsec_profiling_info_s *next;
    char                          *key;
    char                          *value;
} parsec_profiling_info_t;

struct parsec_thread_profiling_s {
    parsec_list_item_t        list;
    int64_t                  next_event_position; /* When in write mode, points to the next available storage byte
                                                   *   in current_events_buffer */
    char                    *hr_id;
    uint64_t                 nb_events;
    parsec_profiling_info_t  *infos;
    off_t                    first_events_buffer_offset; /* Offset (in the file) of the first events buffer */
    pthread_t                thread_owner;
    off_t                     current_events_buffer_offset;
    parsec_profiling_buffer_t *current_events_buffer;     /* points to the events buffer in which we are writing. */
};

typedef struct {
    char *name;
    char *attributes;
    char *convertor;
    int32_t info_length;
} parsec_profiling_key_t;

#define PARSEC_PROFILING_MAGICK "#PARSEC BINARY PROFILE "

/** here key is the key given to the USER */
#define BASE_KEY(key)     ((key) >> 1)
#define EVENT_LENGTH(key, has_info) (sizeof(parsec_profiling_output_base_event_t) + \
                                     ((has_info) ? parsec_prof_keys[BASE_KEY(key)].info_length : 0))
#define EVENT_HAS_INFO(EV)  ((EV)->event.flags & PARSEC_PROFILING_EVENT_HAS_INFO)
#define KEY_IS_END(key)   ((key) == END_KEY(BASE_KEY(key)))
#define KEY_IS_START(key) ((key) == START_KEY(BASE_KEY(key)))
/** here keys are the internal key */
#define START_KEY(key)    (((key) << 1) + 0)
#define END_KEY(key)      (((key) << 1) + 1)

/** @} */

#endif /* _dbp_h */
