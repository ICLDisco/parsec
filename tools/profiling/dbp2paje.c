/*
 * Copyright (c) 2011-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#undef HAVE_MPI

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <stdarg.h>
#include <sys/time.h>

#include <unistd.h>
#include <getopt.h>

#include "dague/profiling.h"
#include "dague/dague_binary_profile.h"
#include "dbpreader.h"

#ifdef DEBUG
#undef DEBUG
#endif

#if defined(DAGUE_DEBUG_NOISIER)
#define DEBUG(...) output(__VA_ARGS__)
#else
#define DEBUG(toto) do {} while(0)
#endif

#ifdef WARNING
#undef WARNING
#endif
#define WARNING(...) output(__VA_ARGS__)

static void output(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
}

#include <GTG.h>
#include <GTGPaje.h>

struct {
    int split_events_box_at_start;
    int split_events_link;
    int name_all_containers;
    char *outfile;
    char **files;
    int nbfiles;
    int progress;
    int stats;
} USERFLAGS;

static void parse_arguments_error(char *message)
{
    fprintf(stderr,
            "dbp2paje error: %s\n"
            " Usage: dbp2paje [options] profile0 profile1 profile2 ...\n"
            "   where profile0 to profilen are DAGuE Binary Profile files corresponding to a single run\n"
            "   and options consists of the following:\n"
            "\n"
            " General Options\n"
            "   -o|--out <outfile>         Output file base name (default: 'out')\n"
            "   -p|--progress              Disable progress bar (default: enable progress bar)\n"
            "   -s|--stats                 Disable statistics on the profiles (default: enable statistics)\n"
            " Split Event Options\n"
            "  (split events are events that start on a thread and terminate on another)\n"
            "   -b|--box-split-events      Enable boxes for the split events. Without this option, a box on the\n"
            "                              thread that started the event will be shown\n"
            "   -l|--link-split-events     Disable links for the split events. Without this option, a link connecting\n"
            "                              the beginning of this event and the end of this event will be shown.\n"
            "\n", message);
    exit(1);
}

static void parse_arguments(int argc, char **argv)
{
    int c, option_index;
    static struct option long_options[] = {
        {"out", 1, 0, 'o'},
        {"progress", 0, 0, 'p'},
        {"stats", 0, 0, 's'},
        {"box-split-events", 0, 0, 'b'},
        {"link-split-events", 0, 0, 'l'},
        {"name-all-containers", 0, 0, 'n'},
        {"help", 0, 0, 'h'},
        {0, 0, 0, 0}
    };

    USERFLAGS.outfile = strdup("out");
    USERFLAGS.progress = 1;
    USERFLAGS.stats = 1;
    USERFLAGS.split_events_box_at_start = 0;
    USERFLAGS.split_events_link = 1;
    USERFLAGS.name_all_containers = 0;

    while(1) {
        c = getopt_long(argc, argv, "o:psblnh", long_options, &option_index);
        if(-1 == c) {
            break;
        }
        switch(c) {
        case 'o':
            free(USERFLAGS.outfile);
            USERFLAGS.outfile = strdup(optarg);
            break;
        case 'p':
            USERFLAGS.progress = 0;
            break;
        case 's':
            USERFLAGS.stats = 0;
            break;
        case 'b':
            USERFLAGS.split_events_box_at_start = 1;
            break;
        case 'l':
            USERFLAGS.split_events_link = 0;
            break;
        case 'n':
            USERFLAGS.name_all_containers = 1;
            break;
        case 'h':
        default:
            parse_arguments_error("Unrecognized option");
        }
    }

    if( optind < argc ) {
        USERFLAGS.nbfiles = argc-optind;
        USERFLAGS.files   = &argv[optind];
    } else {
        parse_arguments_error("You must provide at least a profile file to convert");
    }

}

#define max(a, b) ((a)>(b)?(a):(b))

FILE *pajeGetProcFile();

static trace_return_t pajeSetState2(varPrec time, const char* type,
                                    const char *cont, const char* val)
{
    FILE *procFile = pajeGetProcFile();
    if (procFile){
        fprintf (procFile, "10 %.13e \"%s\" \"%s\" \"%s\"\n",
                 time, type, cont, val);
        return TRACE_SUCCESS;
    }
    return TRACE_ERR_WRITE;
}

typedef struct {
    int nb_matchsuccess;
    int nb_matchthreads;
    int nb_matcherror;
} dico_stat_t;

typedef struct {
    char *name;
    dico_stat_t *stats;
} thread_stat_t;

static thread_stat_t **dico_stat = NULL;
static int            *stat_columns = NULL;
static dico_stat_t    *current_stat = NULL;

static uint64_t total_events = 0, events_read = 0, events_output = 0, events_to_output = 0;
static int      total_threads = 0, threads_done = 0;
static int      total_files = 0, files_done = 0;
static struct timeval start_run;
static struct timeval last_display;

typedef struct {
    dague_list_item_t super;
    uint64_t        event_id;
    uint64_t        start;
    uint64_t        end;
    int             key;
    uint32_t        handle_id;
    const dbp_thread_t   *start_thread;
    const dbp_thread_t   *end_thread;
    size_t          start_info_size;
    size_t          end_info_size;
    char            infos[1];
} consolidated_event_t;

static void progress_bar_init(uint64_t nb_events, int nb_threads, int nb_files)
{
    if( !USERFLAGS.progress )
        return;
    total_events = nb_events;
    total_threads = nb_threads;
    total_files = nb_files;

    events_read = 0;
    events_output = 0;
    events_to_output = 0;
    threads_done = 0;
    files_done = 0;

    gettimeofday(&start_run, NULL);
}

static void progress_bar_update(int force)
{
    struct timeval now, diff;
    double delta;
    char eta[64], to_output[64];

    if( !USERFLAGS.progress )
        return;
    gettimeofday(&now, NULL);
    timersub(&now, &last_display, &diff);
    delta = (double)diff.tv_sec + (double)diff.tv_usec / 1000000.0;

    if( (delta < 1.0) && !force )
        return;

    timersub(&now, &start_run, &diff);
    delta = (double)diff.tv_sec + (double)diff.tv_usec / 1000000.0;
    if( events_read > 0 ) {
        sprintf(eta, "%fs", ((double)total_events-(double)events_read)*delta/(double)events_read);
    } else {
        sprintf(eta, " -- ");
    }

    if( events_to_output > 0 ) {
        sprintf(to_output, "%4.1f%%", (double)events_output*100.0/(double)events_to_output);
    } else {
        sprintf(to_output, " -- ");
    }

    fprintf(stderr, "\r%d/%d files done; %d/%d threads done; %4.1f%% events read (%s of those events have been output); ETA: %s                        %s",
            files_done, total_files,
            threads_done, total_threads,
            (double)events_read*100.0/(double)total_events,
            to_output,
            eta,
            force ? "\n" : "");
    fflush(stderr);

    gettimeofday(&last_display, NULL);
}

static void progress_bar_file_done(void)
{
    if( !USERFLAGS.progress )
        return;
    files_done++;
    progress_bar_update(0);
}

static void progress_bar_thread_done(void)
{
    if( !USERFLAGS.progress )
        return;
    threads_done++;
    progress_bar_update(0);
}

static void progress_bar_event_read(void)
{
    if( !USERFLAGS.progress )
        return;
    events_read++;
    progress_bar_update(0);
}

static void progress_bar_event_to_output(void)
{
    if( !USERFLAGS.progress )
        return;
    events_to_output++;
    progress_bar_update(0);
}

static void progress_bar_event_output(void)
{
    if( !USERFLAGS.progress )
        return;
    events_output++;
    progress_bar_update(0);
}

static void progress_bar_end(void)
{
    if( !USERFLAGS.progress )
        return;
    progress_bar_update(1);
}

typedef uint64_t hash_key_t;

typedef struct uidentry {
    struct uidentry *next;
    hash_key_t       hash_key;
    char            *uid;
    char            *long_uid;
    char            *alias;
} uidentry_t;

#define UID_HASH_LEN 256
static uidentry_t *UIDs[UID_HASH_LEN] = { NULL, };

/**
 * Improve the hash function to avoid as much as possible the collisions. Use a
 * murmur3 with a 32 bits length derived function.
 */
static hash_key_t uid_hash_va(const char *first, va_list va)
{
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    hash_key_t h = 0, k;
    uint32_t nbytes, tbytes = 0;
    const char* arg = first;

    do {
        nbytes = strlen(arg);
        int nblocks = nbytes / 4;
        const uint32_t *blocks = (const uint32_t *)(arg);
        const uint8_t *tail = (const uint8_t *)(arg + (nblocks * 4));

        for(; nblocks > 0; nblocks--, blocks++) {
            k = *blocks;

            k *= c1;
            k = (k << 15) | (k >> (32 - 15));
            k *= c2;

            h ^= k;
            h = (h << 13) | (h >> (32 - 13));
            h = (h * 5) + 0xe6546b64;
        }

        /* epilogue */
        k = 0;
        switch (nbytes & 3) {
        case 3:
            k ^= tail[2] << 16;
        case 2:
            k ^= tail[1] << 8;
        case 1:
            k ^= tail[0];
            k *= c1;
            k = (k << 13) | (k >> (32 - 15));
            k *= c2;
            h ^= k;
        }
        tbytes += nbytes;  /* update the total number of bytes */

        arg = va_arg(va, const char*);
    } while( NULL != arg );

    /* The final epilogue */
    h ^= tbytes;

    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

static uidentry_t *uidhash_lookup_create_entry(const char *first, ...)
{
    static int nextid = 0;
    uidentry_t *n;
    va_list va;
    int rc;
    hash_key_t h;
    int idx;
    size_t length = strlen(first);
    char *where;
    const char *arg;

    va_start(va, first);
    h = uid_hash_va(first, va);
    idx = h % UID_HASH_LEN;
    va_end(va);

    /* We don't support collisions */
    for(n = UIDs[idx]; NULL != n; n = n->next) {
        if( n->hash_key == h )
            return n;
    }

    /* Compute the length of all strings passed as arguments */
    va_start(va, first);
    while( NULL != (arg = va_arg(va, const char*)) ) {
        length += strlen(arg);
    }
    va_end(va);

    n = (uidentry_t*)malloc( sizeof(uidentry_t) );
    n->long_uid = (char*)malloc(length+1);
    /* Copy all strings in the long_uid */
    va_start(va, first);
    where = n->long_uid;
    arg = first;
    do {
        length = strlen(arg);
        strncpy( where, arg, length );
        where += length;
    } while( NULL != (arg = va_arg(va, const char*)) );
    where[0] = '\0';  /* NULL terminated */
    va_end(va);
    n->alias = NULL;
    n->hash_key = h;
    rc = asprintf(&n->uid, "%X", nextid++); assert(rc!=-1);
    n->next = UIDs[idx];
    UIDs[idx] = n;

    return n;
}

static char *getThreadContainerIdentifier( const char* mpi_alias, const dbp_thread_t *th )
{
    uidentry_t *n;
    const char *identifier = dbp_thread_get_hr_id(th);

    /* Register the full identifier in hashtable */
    n = uidhash_lookup_create_entry(mpi_alias, identifier, NULL);
    if (n->alias != NULL) {
        return n->alias;
    }
    else {
        assert(0);
        return "";
    }
}

static char *registerThreadContainerIdentifier( const char *mpi_alias, dbp_thread_t *th )
{
    uidentry_t *n, *p;
    int parent_id, son_id;
    int rc;
    char binding_info[1024]; /**< Risk of buffer overflow. A better approach should be used... */

    const char *identifier = dbp_thread_get_hr_id(th);

    /* Register the full identifier in hashtable */
    n = uidhash_lookup_create_entry(mpi_alias, identifier, NULL);
    if (n->alias != NULL) {
        return n->alias;
    }

    /* Check if this is a GPU/stream couple */
    if ( sscanf( identifier, DAGUE_PROFILE_STREAM_STR, &parent_id, &son_id ) == 2 )
    {
        char *gpu_name, *stream_name;

        /* Create name */
        rc = asprintf( &gpu_name,    "GPU %d",    parent_id); assert(rc!=-1);
        switch (son_id) {
        case 0:
            rc = asprintf( &stream_name, "Push" ); assert(rc!=-1);
            break;
        case 1:
            rc = asprintf( &stream_name, "Pop" ); assert(rc!=-1);
            break;
        default:
            rc = asprintf( &stream_name, "Stream %d", son_id - 2 ); assert(rc!=-1);
        }

        /* Register the new containers */
        p = uidhash_lookup_create_entry(mpi_alias, gpu_name, NULL);
        if (p->alias == NULL) {
            rc = asprintf( &(p->alias), "%sG%d", mpi_alias, parent_id); assert(rc!=-1);
            addContainer (0.00000, p->alias, "CT_VP", mpi_alias, gpu_name, "");
        }

        rc = asprintf( &(n->alias), "%sS%d", p->alias, son_id   ); assert(rc!=-1);
        addContainer (0.00000, n->alias, "CT_T", p->alias, stream_name, "");

        free(gpu_name); free(stream_name);
    }
    else if ( sscanf( identifier, DAGUE_PROFILE_THREAD_STR, &son_id, &parent_id, binding_info ) == 3 )
    {
        char *vp_name, *thrd_name;

        /* Create name */
        rc = asprintf( &vp_name,   "VP %d",     parent_id); assert(rc!=-1);
        rc = asprintf( &thrd_name, "Thread %d", son_id   ); assert(rc!=-1);

        /* Register the new containers */
        p = uidhash_lookup_create_entry(mpi_alias, vp_name, NULL);
        if (p->alias == NULL) {
            rc = asprintf( &(p->alias),   "%sV%d", mpi_alias, parent_id); assert(rc!=-1);
            addContainer (0.00000, p->alias, "CT_VP", mpi_alias, vp_name, "");
        }

        rc = asprintf( &(n->alias), "%sT%d", p->alias,  son_id   ); assert(rc!=-1);
        addContainer (0.00000, n->alias, "CT_T",  p->alias, thrd_name, "");

        free(vp_name); free(thrd_name);
    }
    else {
        rc = asprintf( &(n->alias), "%sT%s", mpi_alias, n->uid ); assert(rc!=-1);
        addContainer (0.00000, n->alias, "CT_T", mpi_alias, identifier, "");
    }

    return n->alias;
}


static int merge_event( dague_list_t *list, consolidated_event_t *cev )
{
    dague_list_item_t *it;
    consolidated_event_t *lev, *next;
    int broken = 0;

    next = NULL;
    for( it = DAGUE_LIST_ITERATOR_LAST(list);
         it != DAGUE_LIST_ITERATOR_BEGIN(list);
         it = DAGUE_LIST_ITERATOR_PREV(it) ) {
        lev = (consolidated_event_t*)it;
        if( cev->start >= lev->start ) {
            if( ((cev->start < lev->end) ||
                 ((next != NULL) && (cev->end > next->start) )) &&
                (cev->start_thread == cev->end_thread) ) {
                broken = 1;
            }
            dague_list_nolock_add_after( list,
                                         it,
                                         (dague_list_item_t*)cev );
            return broken;
        }
        next = lev;
    }
    if( (next != NULL) && (cev->end > next->start) &&
        (cev->start_thread == cev->end_thread) ) {
        broken = 1;
    }
    dague_list_nolock_push_front( list, (dague_list_item_t*)cev );
    return broken;
}

static uint64_t *step_height(dague_list_t *list, int *level)
{
    dague_list_item_t *e;
    consolidated_event_t *cev;
    int s, nb_steps = 0;
    static int allocated_dates = 0;
    static uint64_t *dates = NULL;

    e = DAGUE_LIST_ITERATOR_FIRST(list);
    cev = (consolidated_event_t*)e;

    for( ; e != DAGUE_LIST_ITERATOR_END(list);
         e = DAGUE_LIST_ITERATOR_NEXT(e) ) {
        cev = (consolidated_event_t*)e;

        if( cev->start_thread == cev->end_thread ||
            USERFLAGS.split_events_box_at_start ) {
            for(s = 0; s < nb_steps; s++) {
                if( dates[s] <= cev->start ) {
                    dates[s] = cev->end;
                    break;
                }
            }
            if (s == nb_steps) {
                nb_steps++;
                if( nb_steps > allocated_dates ) {
                    allocated_dates = nb_steps;
                    dates = (uint64_t*)realloc(dates, nb_steps * sizeof(uint64_t));
                }
                dates[s] = cev->end;
            }
        }
    }
    memset(dates, 0, nb_steps * sizeof(uint64_t));
    *level = nb_steps;
    return dates;
}

static int dump_one_event( dague_list_t *consolidated_events,
                           const dbp_multifile_reader_t *dbp,
                           const dbp_event_iterator_t *istart, const dbp_event_t *estart,
                           const dbp_event_iterator_t *iend,   const dbp_event_t *eend )
{
    uint64_t start, end;
    consolidated_event_t *cev;

    start = dbp_event_get_timestamp( estart );
    end   = dbp_event_get_timestamp( eend   );

    assert( start <= end );

    cev = (consolidated_event_t*)malloc(sizeof(consolidated_event_t) +
                                        dbp_event_info_len(estart, dbp) +
                                        dbp_event_info_len(eend,   dbp) );
    OBJ_CONSTRUCT(cev, dague_list_item_t);

    cev->event_id  = dbp_event_get_event_id(  estart );
    cev->handle_id = dbp_event_get_handle_id( estart );
    cev->start = start;
    cev->end   = end;
    cev->start_thread = dbp_iterator_thread(istart);
    cev->end_thread   = dbp_iterator_thread(iend);
    cev->key = BASE_KEY(dbp_event_get_key(estart));
    cev->start_info_size = dbp_event_info_len(estart, dbp);
    cev->end_info_size   = dbp_event_info_len(eend,   dbp);
    memcpy(cev->infos, dbp_event_get_info( estart ), cev->start_info_size);
    memcpy(cev->infos + cev->start_info_size, dbp_event_get_info( eend ), cev->end_info_size);

    progress_bar_event_to_output();

    merge_event( consolidated_events, cev );

    return 0;
}

static int dump_one_paje( const dbp_multifile_reader_t *dbp,
                          const dbp_thread_t *th,
                          const char *cont_mpi_name,
                          const char *cont_thread_name )
{
    unsigned int key;
    int s;
    char keyid[64];
    dbp_event_iterator_t *pit, *nit;
    dague_list_t consolidated_events;
    consolidated_event_t *cev;
    static int linkuid = 0;
    char linkid[64];
    char cont_step_name[64];
    const dbp_event_t *e, *g;
    char *cont_src;
    char *cont_dst;
    uint64_t *steps_end_dates;
    int nb_steps;

    pit = dbp_iterator_new_from_thread( th );
    OBJ_CONSTRUCT( &consolidated_events, dague_list_t );
    while( (e = dbp_iterator_current(pit)) != NULL ) {
        if( KEY_IS_START( dbp_event_get_key(e) ) ) {

            key = BASE_KEY(dbp_event_get_key(e));

            if ( ! USERFLAGS.split_events_box_at_start ) {
                const dbp_event_t *ref = dbp_iterator_current(pit);
                nit = dbp_iterator_new_from_iterator(pit);
                if ( !dbp_iterator_move_to_matching_event(nit, ref, 0) ) {
                    dbp_iterator_delete(nit);
                    nit = NULL;
                }
                else {
                    /* Check if nit is a rescheduled event,
                     * if yes, then add 2 events, one local for submission,
                     * one for the communication to the starting event on the device */
                    g = dbp_iterator_current(nit);
                    dump_one_event( &consolidated_events, dbp,
                                    pit, e, nit, g );
                    current_stat[ key ].nb_matchsuccess++;

                    /* Add the arrow to the rescheduled event */
                    if (USERFLAGS.split_events_link &&
                        ( dbp_event_get_flags(g) & DAGUE_PROFILING_EVENT_RESCHEDULED ))
                    {
                        dbp_event_iterator_t *nit2;
                        /* Search for next starting event on another device */
                        nit2 = dbp_iterator_find_matching_event_all_threads(nit, 1);
                        if( NULL == nit ) {
                            /* Argh, couldn't find the end in this trace */
                            WARNING("   Event of class %s id %"PRIu32":%"PRIu64" rescheduled at %lu does not have a new starting point anywhere\n",
                                     dbp_dictionary_name(dbp_reader_get_dictionary(dbp, BASE_KEY(dbp_event_get_key(e)))),
                                     dbp_event_get_handle_id(e), dbp_event_get_event_id(e),
                                     dbp_event_get_timestamp(e));
                            current_stat[ key ].nb_matcherror++;
                        }
                        else {
                            if( dbp_iterator_thread(nit) != dbp_iterator_thread(nit2) ) {
                                current_stat[ key ].nb_matchthreads++;
                            }

                            dump_one_event( &consolidated_events, dbp,
                                            nit, g, nit2, dbp_iterator_current(nit2));

                            dbp_iterator_delete(nit2);
                        }
                    }
                    dbp_iterator_delete(nit);
                }
            } else {
                dbp_event_iterator_t *nit2 = NULL;
                nit = dbp_iterator_find_matching_event_all_threads(pit, 0);

                g = dbp_iterator_current(nit);
                if ( dbp_event_get_flags(g) & DAGUE_PROFILING_EVENT_RESCHEDULED ) {
                    nit2 = nit;
                    nit = dbp_iterator_find_matching_event_all_threads(nit2, 0);
                    assert(nit != nit2);
                }

                if( NULL == nit ) {
                    /* Argh, couldn't find the end in this trace */
                    WARNING("   Event of class %s id %"PRIu32":%"PRIu64" at %lu does not have a match anywhere\n",
                             dbp_dictionary_name(dbp_reader_get_dictionary(dbp, BASE_KEY(dbp_event_get_key(e)))),
                             dbp_event_get_handle_id(e), dbp_event_get_event_id(e),
                             dbp_event_get_timestamp(e));

                    current_stat[ key ].nb_matcherror++;
                }
                else {
                    if( dbp_iterator_thread(nit) != dbp_iterator_thread(pit) ) {
                        current_stat[ key ].nb_matchthreads++;
                    }
                    current_stat[ key ].nb_matchsuccess++;

                    dump_one_event( &consolidated_events, dbp,
                                    pit, e, nit, dbp_iterator_current(nit));

                    dbp_iterator_delete(nit);
                }
                if (nit2 != NULL)
                    dbp_iterator_delete(nit2);
            }
        }
        progress_bar_event_read();
        dbp_iterator_next(pit);
    }
    dbp_iterator_delete(pit);

    steps_end_dates = step_height(&consolidated_events, &nb_steps);
    for(s = 0; s < nb_steps; s++) {
        sprintf(cont_step_name, "%s-%d", cont_thread_name, s);
        addContainer(0.00000, cont_step_name, "CT_ST", cont_thread_name, USERFLAGS.name_all_containers ? cont_step_name : " ", "");
    }

    while( NULL != (cev = (consolidated_event_t*)dague_list_nolock_pop_front( &consolidated_events ) ) ) {
        sprintf(keyid, "K-%d", cev->key);

        if( (cev->start_thread == cev->end_thread) ||
            USERFLAGS.split_events_box_at_start )
        {
            for(s = 0; s < nb_steps; s++) {
                if( steps_end_dates[s] <= cev->start ) {
                    steps_end_dates[s] = cev->end;
                    break;
                }
            }
            assert( s < nb_steps );
            sprintf(cont_step_name, "%s-%d", cont_thread_name, s);
            pajeSetState2( ((double)cev->start) * 1e-3, "ST_TS", cont_step_name, keyid );
            pajeSetState2( ((double)cev->end)   * 1e-3, "ST_TS", cont_step_name, "Wait");
        }
        if( (cev->start_thread != cev->end_thread) &&
            USERFLAGS.split_events_link ) {
            sprintf(linkid, "L-%d", linkuid);
            linkuid++;
            cont_src = getThreadContainerIdentifier( cont_mpi_name, cev->start_thread );
            cont_dst = getThreadContainerIdentifier( cont_mpi_name, cev->end_thread );
            startLink( ((double)cev->start) * 1e-3, "LT_TL", cont_mpi_name, cont_src, cont_dst, keyid, linkid);
            endLink(   ((double)cev->end)   * 1e-3, "LT_TL", cont_mpi_name, cont_src, cont_dst, keyid, linkid);
        }
        free(cev);
        progress_bar_event_output();
    }

    OBJ_DESTRUCT( &consolidated_events );
    return 0;
}

static int dague_profiling_dump_paje( const char* filename, const dbp_multifile_reader_t *dbp )
{
    int i, t, ifd;
    dbp_dictionary_t *dico;
    dbp_file_t *file;
    dbp_thread_t *th;
    gtg_color_t color;
    unsigned long int color_code;
    char dico_id[64];
    char cont_mpi_name[64];
    char *cont_thread_name;
    char name[64];

    setTraceType (PAJE);
    initTrace (filename, 0, GTG_FLAG_NONE);
    addContType ("CT_Appli", "0", "Application");
    addContType ("CT_P", "CT_Appli", "Process");

    addContType ("CT_VP", "CT_P",  "VP"     );
    addContType ("CT_T",  "CT_VP", "Thread" );
    addContType ("CT_ST", "CT_T",  "Thread Event");
    addStateType("ST_TS", "CT_ST", "Thread State");

    addLinkType ("LT_TL", "Split Event Link", "CT_P", "CT_T", "CT_T");

    /* Create root container of the application */
    addContainer (0.00000, "Appli", "CT_Appli", "0", dbp_file_hr_id(dbp_reader_get_file(dbp, 0)), "");

    /* Add all possible states from the dictionnary */
    addEntityValue ("Wait", "ST_TS", "Waiting", GTG_LIGHTGREY);

    for(i = 0; i < dbp_reader_nb_dictionary_entries(dbp); i++) {
        dico = dbp_reader_get_dictionary(dbp, i);
        color_code = strtoul( dbp_dictionary_attributes(dico), NULL, 16);
        color = gtg_color_create(dbp_dictionary_name(dico),
                                 GTG_COLOR_GET_RED(color_code),
                                 GTG_COLOR_GET_GREEN(color_code),
                                 GTG_COLOR_GET_BLUE(color_code));
        sprintf(dico_id, "K-%d", i);
        addEntityValue (dico_id, "ST_TS", dbp_dictionary_name(dico), color);
        gtg_color_free(color);
    }

    /* Create the containers architecture */
    for(ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        file = dbp_reader_get_file(dbp, ifd);

        sprintf(name,          "MPI-%d", dbp_file_get_rank(file));
        sprintf(cont_mpi_name, "M%d",    dbp_file_get_rank(file));
        addContainer (0.00000, cont_mpi_name, "CT_P", "Appli", name, "");
        for(t = 0; t < dbp_file_nb_threads(file); t++) {
            th = dbp_file_get_thread(file, t);

            cont_thread_name = registerThreadContainerIdentifier( cont_mpi_name, th );
            assert(cont_thread_name != NULL);
            {
                int l;
                l = 3 + snprintf(NULL, 0, "#  %s", cont_thread_name);
                if( l > stat_columns[0] )
                    stat_columns[0] = l;
                dico_stat[ifd][t].name = strdup(cont_thread_name);
            }
        }
    }

    for(ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        file = dbp_reader_get_file(dbp, ifd);

        sprintf(cont_mpi_name, "M%d", dbp_file_get_rank(file));
        for(t = 0; t < dbp_file_nb_threads(file); t++) {
            th = dbp_file_get_thread(file, t);
            cont_thread_name = getThreadContainerIdentifier( cont_mpi_name, th );
            assert(cont_thread_name != NULL);
            current_stat = dico_stat[ifd][t].stats;
            dump_one_paje(dbp, th, cont_mpi_name, cont_thread_name);
            progress_bar_thread_done();
        }
        progress_bar_file_done();
    }

    return 0;
}

int main(int argc, char *argv[])
{
    dbp_multifile_reader_t *dbp;
    dbp_file_t *file;
    uint64_t nb_events = 0;
    int nb_threads = 0;
    int i, j, k;

    parse_arguments(argc, argv);

    dbp = dbp_reader_open_files(USERFLAGS.nbfiles, USERFLAGS.files);

    if( NULL == dbp )
        return 1;

    if( dbp_reader_nb_files(dbp) == 0 ) {
        fprintf(stderr, "Unable to open any of the files. Aborting.\n");
        exit(1);
    }

    for(i = 0; i < dbp_reader_nb_files(dbp); i++) {
        file = dbp_reader_get_file(dbp, i);
        nb_threads += dbp_file_nb_threads(file);
        for(j = 0; j < dbp_file_nb_threads(file); j++) {
            nb_events += dbp_thread_nb_events( dbp_file_get_thread(file, j));
        }
    }

    progress_bar_init(nb_events, nb_threads, dbp_reader_nb_files(dbp));

    dico_stat = (thread_stat_t**)malloc(dbp_reader_nb_files(dbp) * sizeof(thread_stat_t*));
    for(i = 0; i < dbp_reader_nb_files(dbp); i++) {
        dico_stat[i] = (thread_stat_t*)calloc(dbp_file_nb_threads( dbp_reader_get_file(dbp, i) ), sizeof(thread_stat_t));
        for(j = 0; j < dbp_file_nb_threads( dbp_reader_get_file(dbp, i) ); j++) {
            dico_stat[i][j].stats = (dico_stat_t*)calloc(dbp_reader_nb_dictionary_entries(dbp), sizeof(dico_stat_t));
        }
    }
    stat_columns = (int*)calloc(dbp_reader_nb_dictionary_entries(dbp)+1, sizeof(int));

    for(i = 0; i < dbp_reader_nb_files(dbp); i++) {
        int l;
        file = dbp_reader_get_file(dbp, i);
        l = 3 + snprintf(NULL, 0, "#%s Rank %d/%d",
                         dbp_file_hr_id(file),
                         dbp_file_get_rank(file),
                         dbp_reader_worldsize(dbp));
        if( l > stat_columns[0] )
            stat_columns[0] = l;
    }

    dague_profiling_dump_paje( USERFLAGS.outfile, dbp );

    progress_bar_end();

    for(k = 0 ; k < dbp_reader_nb_dictionary_entries(dbp); k = k+1 ) {
        int l = strlen(dbp_dictionary_name(dbp_reader_get_dictionary(dbp, k)));
        stat_columns[k+1] = stat_columns[k] + max(l + 2, 16);
    }

    if( USERFLAGS.stats ) {
        printf("#Stats:\n");
        printf("#Thread   ");
        for(k = 0 ; k < dbp_reader_nb_dictionary_entries(dbp); k = k+1 ) {
            printf("[%dG%s", stat_columns[k], dbp_dictionary_name(dbp_reader_get_dictionary(dbp, k)));
        }
        printf("\n");
        for(i = 0; i < dbp_reader_nb_files(dbp); i++) {
            file = dbp_reader_get_file(dbp, i);
            printf("#%s Rank %d/%d\n",
                   dbp_file_hr_id(file),
                   dbp_file_get_rank(file),
                   dbp_reader_worldsize(dbp));
            for(j = 0; j < dbp_file_nb_threads(file); j++) {
                printf("#  %s", dico_stat[i][j].name);

                for(k = 0; k < dbp_reader_nb_dictionary_entries(dbp); k++) {
                    printf("[%dG[%dm%d[0m/[%dm%d[0m/[%dm%d[0m",
                           stat_columns[k],
                           dico_stat[i][j].stats[k].nb_matchsuccess > 0 ? 32 : 2,
                           dico_stat[i][j].stats[k].nb_matchsuccess,
                           dico_stat[i][j].stats[k].nb_matchthreads > 0 ? 35 : 2,
                           dico_stat[i][j].stats[k].nb_matchthreads,
                           dico_stat[i][j].stats[k].nb_matcherror > 0 ? 31 : 2,
                           dico_stat[i][j].stats[k].nb_matcherror);
                }

                printf("\n");
            }
        }
    }

    dbp_reader_destruct(dbp);
    endTrace();

    return 0;
}
