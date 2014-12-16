/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#undef HAVE_MPI

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>

#include "profiling.h"
#include "dbp.h"
#include "dbpreader.h"

static char **type_strings = NULL;
static int   *type_indexes = NULL;
static int  nbtypes = 0;

typedef struct memalloc_event_s {
    uint64_t time;
    uint64_t ptr;
    size_t size;

    struct memalloc_event_s *buddy;
    struct memalloc_event_s *next_in_time;
} memalloc_event_t;

static memalloc_event_t *EVENTS;

void insert_event(memalloc_event_t *event) {
    memalloc_event_t *e, *p;
    for(p = NULL, e = EVENTS;
        NULL != e && e->time < event->time;
        e = e->next_in_time )  p = e;
    if( NULL == p ) {
        event->next_in_time = EVENTS;
        EVENTS = event;
    } else {
        event->next_in_time = p->next_in_time;
        p->next_in_time = event;
    }
}

#if DAGUE_DEBUG_VERBOSE >= 2
#define DEBUG(toto) output toto
#else
#define DEBUG(toto) do {} while(0)
#endif
#define WARNING(toto) output toto

static void output(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
}

static void find_memory_ref_in_thread(const dbp_multifile_reader_t *dbp, int dico_id, int nid, int tid)
{
    uint64_t k;
    uint64_t start, end;
    dbp_event_iterator_t *it;
    const dbp_event_t *e;
    const dbp_thread_t *th = dbp_file_get_thread( dbp_reader_get_file(dbp, nid), tid);
    memalloc_event_t *m = NULL;
    size_t *info;

    k = 0;
    it = dbp_iterator_new_from_thread( th );
    while( (e = dbp_iterator_current(it)) != NULL ) {
        k++;
        if( (k % 10000) == 9999 ) {
            printf("."); fflush(stdout);
        }
        if( BASE_KEY( dbp_event_get_key(e) ) != dico_id ) {
            dbp_iterator_next(it);
            continue;
        }

        m = (memalloc_event_t*)calloc(1, sizeof(memalloc_event_t));
        m->time = dbp_event_get_timestamp( e );
        m->ptr = dbp_event_get_event_id(e);

        if( KEY_IS_START( dbp_event_get_key(e) ) ) {
            if( (dbp_event_get_flags( e ) & DAGUE_PROFILING_EVENT_HAS_INFO) &&
                (dbp_event_info_len(e, dbp) == sizeof(size_t)) ) {
                info = dbp_event_get_info(e);
                m->size = *info;
                if( m->size == 0 ) {
                    /* Ignore this */
                    free(m);
                    continue;
                } else {
                    insert_event(m);
                }
            } else {
                if( !(dbp_event_get_flags( e ) & DAGUE_PROFILING_EVENT_HAS_INFO) ) {
                    fprintf(stderr, "Event has no size information. Cannot trace this event!\n");
                } else {
                    fprintf(stderr, "Event has an information of size %d, not %lu. Cannot trace this event!\n",
                            dbp_event_info_len(e, dbp), sizeof(size_t));
                }
                free(m);
                dbp_iterator_next(it);
                continue;
            }
        } else {
            m->size = 0; /* Marks this as a free */
            insert_event(m);
        }
        dbp_iterator_next(it);
    }
    dbp_iterator_delete(it);
}

static int find_references( const char* filename, char *dico_name, const dbp_multifile_reader_t *dbp)
{
    int i, ifd, t;
    dbp_dictionary_t *dico;
    FILE* tracefile;
    memalloc_event_t *e, *a, *f, *p;
    long long int allocated;
    int dico_id;
    char *dico_string;

    for(i = 0; i < dbp_reader_nb_dictionary_entries(dbp); i++) {
        dico = dbp_reader_get_dictionary(dbp, i);
        if( !strcmp(dbp_dictionary_name(dico), dico_name) ) {
            dico_id = i;
            dico_string = strdup( dbp_dictionary_name(dico) );
            break;
        }
    }
    if( i == dbp_reader_nb_dictionary_entries(dbp)) {
        fprintf(stderr, "Unable to find the dictionary entry called '%s'\n", dico_name);
        return -1;
    }

    tracefile = fopen(filename, "w");
    if( NULL == tracefile ) {
        fprintf(stderr, "Unable to open %s in write mode: %s\n", filename, strerror(errno));
        return -1;
    }

    fprintf(tracefile, "#Date ");
    for( ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        fprintf(tracefile, "Rank%d ", ifd);
    }
    fprintf(tracefile, "\n");

    for(ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        for(t = 0; t < dbp_file_nb_threads(dbp_reader_get_file(dbp, ifd)); t++) {
            find_memory_ref_in_thread(dbp, dico_id, ifd, t);
            printf("Found all memory references in thread %d of node %d\n", t, ifd);
        }
        printf("Found all memory references in node %d\n", ifd);

        for(a = EVENTS; NULL != a; a = a->next_in_time) {
            if( a->size == 0 )
                continue;
            for(f = a->next_in_time; NULL != f; f = f->next_in_time) {
                if( a->ptr == f->ptr ) {
                    a->buddy = f;
                    f->buddy = a;
                    break;
                }
            }
        }

        allocated = 0;
        for(e = EVENTS; NULL != e; e = e->next_in_time) {
            if( e->size > 0 ) {
                allocated += e->size;
            } else {
                if( e->buddy == NULL ) {
                    fprintf(stderr, "Free without malloc...\n");
                    continue;
                } else {
                    allocated -= e->buddy->size;
                }
            }
            fprintf(tracefile, "%llu %d %lld\n", e->time, ifd, allocated);
        }

        p = NULL;
        for(e = EVENTS; NULL != e; e = e->next_in_time) {
            if(p) free(p);
            p = e;
        }
        if(p) free(p);
        EVENTS = NULL;
    }

    fclose(tracefile);
    return 0;
}

int main(int argc, char *argv[])
{
    dbp_multifile_reader_t *dbp;

    dbp = dbp_reader_open_files(argc-1, argv+1);
    printf("DBP files read\n");
    if( NULL == dbp ) {
        return 1;
    }

    find_references("out.dta", "ARENA_MEMORY", dbp);

    dbp_reader_close_files(dbp);

    return 0;
}
