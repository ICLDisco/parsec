#include "dague_config.h"
#undef HAVE_MPI

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "profiling.h"
#include "dbp.h"

#if defined(DAGUE_DEBUG_VERBOSE1)
#define DEBUG(toto) printf toto
#else
#define DEBUG(toto) do {} while(0)
#endif

#include <GTG.h>
#include <GTGPaje.h>

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
    int fd;
    char *filename;
    dague_profiling_binary_file_header_t head;
} dbp_file_t;

typedef struct {
    const dague_thread_profiling_t  *profile;
    int                              fd;
    dague_profiling_buffer_t        *current_events_buffer;
    int64_t                          current_event_position;
    int64_t                          current_event_index;
    int64_t                          current_buffer_position;
} dague_profiling_iterator_t;

typedef struct {
    int nb_matched_samethread;
    int nb_matched_differentthread;
    int nb_matcherror;
} dico_stat_t;

typedef struct {
    char *name;
    dico_stat_t *stats;
} thread_stat_t;

static thread_stat_t **dico_stat = NULL;
static int            *stat_columns = NULL;
static dico_stat_t    *current_stat = NULL;

/* All files */
static dbp_file_t **files;
static int nb_files;

/* Buffer constants */
static int event_buffer_size = 0;
static int event_avail_space = 0;

/* Process-global dictionnary */
static unsigned int dague_prof_keys_count;
static dague_profiling_key_t* dague_prof_keys;

/* List of threads */
static dague_list_t threads;
static char *hr_id = NULL;

static void release_events_buffer(dague_profiling_buffer_t *buffer)
{
    if( NULL == buffer )
        return;
    if( munmap(buffer, event_buffer_size) == -1 ) {
        fprintf(stderr, "Warning profiling system: unmap of the events backend file at %p failed: %s\n",
                buffer, strerror(errno));
    }
}

static dague_profiling_buffer_t *refer_events_buffer( int fd, int64_t offset )
{
    dague_profiling_buffer_t *res;
    res = mmap(NULL, event_buffer_size, PROT_READ, MAP_SHARED, fd, offset);
    if( MAP_FAILED == res )
        return NULL;
    return res;
}

static dague_profiling_iterator_t *iterator_new( const dague_thread_profiling_t *p, int fd )
{
    dague_profiling_iterator_t *res = (dague_profiling_iterator_t*)malloc(sizeof(dague_profiling_iterator_t));
    res->profile = p;
    res->current_event_position = 0;
    res->current_event_index = 0;
    res->current_buffer_position = (off_t)-1;
    res->current_events_buffer  = NULL;
    res->fd = fd;
    return res;
}

static dague_profiling_iterator_t *iterator_new_from_iterator( const dague_profiling_iterator_t *it )
{
    dague_profiling_iterator_t *res = (dague_profiling_iterator_t*)malloc(sizeof(dague_profiling_iterator_t));
    res->profile = it->profile;
    res->current_event_position = it->current_event_position;
    res->current_event_index = it->current_event_index;
    res->current_buffer_position = it->current_buffer_position;
    res->current_events_buffer = refer_events_buffer( it->fd, res->current_buffer_position );
    res->fd = it->fd;
    return res;
}

static dague_profiling_output_t   *iterator_current(dague_profiling_iterator_t *it)
{
    if( it->current_events_buffer == NULL )
        return NULL;
    return (dague_profiling_output_t*)&it->current_events_buffer->buffer[it->current_event_position];
}

static dague_profiling_output_t   *iterator_first(dague_profiling_iterator_t *it)
{
    if( it->current_events_buffer != NULL ) {
        release_events_buffer( it->current_events_buffer );
        it->current_events_buffer = NULL;
    }

    it->current_events_buffer = refer_events_buffer( it->fd, it->profile->first_events_buffer_offset );
    it->current_buffer_position = it->profile->first_events_buffer_offset;
    it->current_event_position = 0;
    return iterator_current(it);
}

static dague_profiling_output_t   *iterator_next(dague_profiling_iterator_t *it)
{
    size_t elen;
    dague_profiling_output_t *current;
    off_t next_off;
    
    current = iterator_current(it);
    if( NULL == current )
        return NULL;
    elen = EVENT_LENGTH(current->event.key, EVENT_HAS_INFO(current));
    assert( it->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );
    if( it->current_event_index+1 >= it->current_events_buffer->this_buffer.nb_events ) {
        next_off = it->current_events_buffer->next_buffer_file_offset;
        release_events_buffer( it->current_events_buffer );
        it->current_event_position = 0;
        it->current_event_index = 0;
        it->current_events_buffer = refer_events_buffer( it->fd, next_off );
        it->current_buffer_position = next_off;

        if( NULL == it->current_events_buffer )
            return NULL;
    } else {
        it->current_event_position += elen;
        it->current_event_index++;
    }
    assert( it->current_event_position <= event_avail_space );
    assert( it->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );

    current = iterator_current(it);
    assert((current == NULL) ||
           (current->event.timestamp.tv_sec != 0));

    return iterator_current(it);
}

static void iterator_delete(dague_profiling_iterator_t *it)
{
    if( NULL != it->current_events_buffer )
        release_events_buffer(it->current_events_buffer);
    free(it);
}

static int find_matching_event_in_profile(const dague_profiling_iterator_t *start_it,
                                          const dague_profiling_output_t *ref,
                                          dague_profiling_output_t **out,
                                          size_t *out_len)
{
    dague_profiling_output_t *e;
    size_t elen;
    dague_profiling_iterator_t *it;

    it = iterator_new_from_iterator( start_it );
    e = iterator_current( it );
    while( NULL != e ) {
        if( (e->event.id == ref->event.id) &&
            (e->event.key == END_KEY(BASE_KEY(ref->event.key))) ) {
            if( e->event.id != 0 ||
                time_less(ref->event.timestamp, e->event.timestamp) ||
                (diff_time(ref->event.timestamp, e->event.timestamp) == 0) ) {
                elen = EVENT_LENGTH(e->event.key, EVENT_HAS_INFO(e));
                if( *out_len < elen ) {
                    *out_len = elen;
                    *out = (dague_profiling_output_t*)realloc(*out, elen);
                }
                memcpy(*out, e, elen);

                /*
                fprintf(stderr, "end event of key %d (%s) id %lu is event %ld->%ld in buffer @%ld\n",
                        BASE_KEY(ref->event.key),
                        dague_prof_keys[ BASE_KEY(ref->event.key) ].name,
                        e->event.id,
                        it->current_event_position, (long int)(it->current_event_position + elen),
                        it->current_buffer_position);
                */

                iterator_delete(it);
                return 1;
            } else if ( e->event.id != 0 ) {
                WARNING(("Event with ID %d appear in reverse order: start is at %d.%09d, end is at %d.%09d\n",
                         e->event.id,
                         (int)ref->event.timestamp.tv_sec, (int)ref->event.timestamp.tv_nsec,
                         (int)e->event.timestamp.tv_sec, (int)e->event.timestamp.tv_nsec));
            }
        }
        e = iterator_next( it );
    }
    iterator_delete(it);
    return 0;
}

static void dump_whole_trace(int fd)
{
    const dague_thread_profiling_t *profile;
    dague_profiling_iterator_t *pit;
    dague_list_item_t *it;
    int i;

    for( i = 0; i < dague_prof_keys_count; i++ ) {
        if( NULL == dague_prof_keys[i].name ) {
            break;
        }
        DEBUG(("TRACE event [%d:%d] name <%s> attributes <%s> info_length %d\n",
               START_KEY(i), END_KEY(i), dague_prof_keys[i].name, dague_prof_keys[i].attributes,
               dague_prof_keys[i].info_length));
    }

    for(it = (dague_list_item_t *)threads.ghost_element.list_next;
        it != &threads.ghost_element;
        it = (dague_list_item_t *)it->list_next) {
        profile = (dague_thread_profiling_t*)it;
        pit = iterator_new( profile, fd );
#if defined(DAGUE_DEBUG_VERBOSE1)
        {
            const dague_profiling_output_t *event;
            for( event = iterator_first( pit );
                 NULL != event;
                 event = iterator_next( pit ) ) {
                dague_time_t zero = ZERO_TIME;
                DEBUG(("TRACE %d/%lu on %p (timestamp %llu)\n", event->event.key, event->event.id, profile,
                       diff_time(zero, event->event.timestamp)));
            }
        }
#endif
        iterator_delete(pit);
    };
}

#define CONSOLIDATED_EVENT_TYPE_UNDEF 0
#define CONSOLIDATED_EVENT_TYPE_STATE 1
#define CONSOLIDATED_EVENT_TYPE_ARROW 2

typedef struct {
    dague_list_item_t super;
    int             type;
    uint64_t        id;
    uint64_t        start;
    uint64_t        end;
    int             key;
    size_t          start_info_size;
    size_t          end_info_size;
    char            infos[1];
} consolidated_event_t;

static int merge_event( dague_list_t *list, consolidated_event_t *cev )
{
    dague_list_item_t *it;
    consolidated_event_t *lev, *prev;
    int broken = 0;

    prev = NULL;
    for( it = DAGUE_LIST_ITERATOR_FIRST(list);
         it != DAGUE_LIST_ITERATOR_END(list);
         it = DAGUE_LIST_ITERATOR_NEXT(it) ) {
        lev = (consolidated_event_t*)it;
        if( lev->start >= cev->start ) {
            if( (cev->end > lev->start) ||
                ((prev != NULL) && (cev->start < prev->end) ) ) {
                broken = 1;
            }
            dague_list_nolock_add_before( list,
                                          it,
                                          (dague_list_item_t*)cev );
            return broken;
        }
        prev = lev;
    }
    if( (prev != NULL) && (cev->start < prev->end) ) {
        broken = 1;
    }
    dague_list_nolock_push_back( list, (dague_list_item_t*)cev );
    return broken;
}

static int dague_profiling_dump_one_paje( const dague_thread_profiling_t *profile, 
                                          char *cont_thread_name,
                                          int backend_fd,
                                          dague_time_t relative )
{
    unsigned int pos, key, broken = 0;
    uint64_t start, end;
    static int displayed_error_message = 0;
    char *infostr = malloc(4);
    int event_not_found;
    dague_thread_profiling_t *op;
    dague_profiling_output_t *start_event;
    dague_profiling_output_t *end_event = NULL;
    size_t end_event_size = 0;
    char keyid[64];
    dague_profiling_iterator_t *pit, *nit;
    dague_list_t consolidated_events;
    consolidated_event_t *cev;
    static int linkuid = 0;
    char linkid[64];

    pit = iterator_new( profile, backend_fd );
    dague_list_construct( &consolidated_events );
    for( start_event = iterator_first( pit );
         NULL != start_event;
         start_event = iterator_next( pit ) ) {

        if( KEY_IS_END( start_event->event.key ) )
            continue;

        pos = BASE_KEY(start_event->event.key);

        /*
        fprintf(stderr, "start event of key %d (%s) id %lu is event %ld->%ld in buffer @%ld\n",
                (int)pos,
                dague_prof_keys[ pos ].name,
                start_event->event.id,
                pit->current_event_position, pit->current_event_position,
                pit->current_buffer_position);
        */

        if( 0 == find_matching_event_in_profile(pit, start_event, &end_event, &end_event_size) ) {
            /* Argh, couldn't find the end in this profile */

            event_not_found = 1;
            /* It has an id, let's look somewhere in another profile, maybe it's end has been
             * logged by another thread
             */
            DAGUE_ULIST_ITERATOR(&threads, it, {
                    op = (dague_thread_profiling_t*)it;
                    if( op == profile )
                        continue;

                    nit = iterator_new( op, backend_fd );
                    if( 1 == find_matching_event_in_profile(nit, start_event, &end_event, &end_event_size) ) {
                        iterator_delete(nit);
                        event_not_found = 0;
                        break;
                    }
                    iterator_delete(nit);
                });

            if( event_not_found ) {
                /* Couldn't find the end, or no id. Bad. */

                WARNING(("Profiling: end event of key %u (%s) id %lu was not found for ID %s\n",
                         pos, dague_prof_keys[pos].name, start_event->event.id, profile->hr_id));

                find_matching_event_in_profile(pit, start_event, &end_event, &end_event_size);

                if( !displayed_error_message ) {
                    dump_whole_trace( backend_fd );
                    displayed_error_message = 1;
                }
                current_stat[ BASE_KEY(start_event->event.key) ].nb_matcherror++;
                continue;
            } else {
                current_stat[ BASE_KEY(start_event->event.key) ].nb_matched_differentthread++;
            }
        } else {
            key = BASE_KEY(start_event->event.key);
            assert( END_KEY(key) == end_event->event.key );
            assert( START_KEY(key) == start_event->event.key );
            assert( start_event != end_event );

            start = diff_time( relative, start_event->event.timestamp );
            end = diff_time( relative, end_event->event.timestamp );

            assert( start <= end );

            cev = (consolidated_event_t*)malloc(sizeof(consolidated_event_t) +
                                                (EVENT_HAS_INFO( start_event ) ? dague_prof_keys[key].info_length : 0) +
                                                (EVENT_HAS_INFO( end_event ) ? dague_prof_keys[key].info_length : 0) );
            cev->type = CONSOLIDATED_EVENT_TYPE_UNDEF;
            cev->id = start_event->event.id;
            cev->start = start;
            cev->end = end;
            cev->key = BASE_KEY( start_event->event.key );
            cev->start_info_size = (EVENT_HAS_INFO( start_event ) ? dague_prof_keys[key].info_length : 0);
            cev->end_info_size = (EVENT_HAS_INFO( end_event ) ? dague_prof_keys[key].info_length : 0);
            memcpy(cev->infos, start_event->info, cev->start_info_size);
            memcpy(cev->infos + cev->start_info_size, end_event->info, cev->end_info_size);

            broken = merge_event( &consolidated_events, cev ) || broken;
        }
    }
    iterator_delete(pit);

    while( NULL != (cev = (consolidated_event_t*)dague_list_nolock_pop_front( &consolidated_events ) ) ) {
        current_stat[ cev->key ].nb_matched_samethread++;

        sprintf(keyid, "K-%d", cev->key);
        if( !broken ) {
            pajeSetState2( ((double)cev->start) * 1e-3, "ST_TS", cont_thread_name, keyid );
            pajeSetState2( ((double)cev->end) * 1e-3, "ST_TS", cont_thread_name, "Wait");
        } else {
            sprintf(linkid, "L-%d", linkuid);
            linkuid++;
            startLink( ((double)cev->start) * 1e-3, "LT_TL", cont_thread_name, cont_thread_name, cont_thread_name, keyid, linkid);
            endLink( ((double)cev->end) * 1e-3, "LT_TL", cont_thread_name, cont_thread_name, cont_thread_name, keyid, linkid);
        }

        free(cev);
    }
    dague_list_destruct( &consolidated_events );
    free(infostr);

    return 0;
}

static void free_thread_heads(void)
{
    dague_thread_profiling_t *profile;
    dague_profiling_info_t *nfo, *next;
    while( NULL != (profile = (dague_thread_profiling_t*)dague_list_pop_front(&threads)) ) {
        for(nfo = profile->infos; nfo != NULL; nfo = next) {
            next = nfo->next;

            free(nfo->key);
            free(nfo->value);
            free(nfo);
        }
        free(profile);
    }
}

static int process_thread_info(dague_thread_profiling_t * res, int nb_infos, char *br )
{
    dague_profiling_info_buffer_t *ib;
    dague_profiling_info_t *nfo;
    int i, pos = 0;

    res->infos = NULL;
    for(i = 0; i < nb_infos; i++) {
        ib = (dague_profiling_info_buffer_t*)&(br[pos]);
        pos += ib->info_size + ib->value_size;
        nfo = (dague_profiling_info_t *)malloc(sizeof(dague_profiling_info_t));
        nfo->next = res->infos;
        res->infos = nfo;
        nfo->key = (char*)malloc(ib->info_size+1);
        memcpy(nfo->key, ib->info_and_value, ib->info_size);
        nfo->key[ib->info_size] = '\0';
        nfo->value = (char*)malloc(ib->value_size+1);
        memcpy(nfo->value, ib->info_and_value + ib->info_size, ib->value_size);
        nfo->value[ib->value_size] = '\0';
    }

    return pos;
}

static int load_thread_heads(int ifd)
{
    dague_thread_profiling_t *res;
    dague_profiling_thread_buffer_t *br;
    dague_profiling_buffer_t *b, *n;
    int nb, nbthis, pos;
    dague_list_construct( &threads );

    pos = 0;
    nb = files[ifd]->head.nb_threads;
    b = refer_events_buffer(files[ifd]->fd, files[ifd]->head.thread_offset);
    nbthis = b->this_buffer.nb_threads;
    while( nb > 0 ) {
        assert(PROFILING_BUFFER_TYPE_THREAD == b->buffer_type);
        assert(nbthis > 0);

        br = (dague_profiling_thread_buffer_t*)&(b->buffer[pos]);

        res = (dague_thread_profiling_t*)malloc( sizeof(dague_thread_profiling_t) );
        res->next_event_position = -1; /* No need for a next event position */
        res->nb_events = br->nb_events;
        res->hr_id = (char*)malloc(128);
        strncpy(res->hr_id, br->hr_id, 128);
        res->first_events_buffer_offset = br->first_events_buffer_offset;
        res->current_events_buffer = NULL;
        DAGUE_LIST_ITEM_CONSTRUCT( res );
        dague_list_fifo_push( &threads, (dague_list_item_t*)res );

        nbthis--;
        nb--;
        pos += sizeof(dague_profiling_thread_buffer_t) - sizeof(dague_profiling_info_buffer_t);
        pos += process_thread_info( res, br->nb_infos, (char*)br->infos );

        if( nbthis == 0 && nb > 0 ) {
            assert( b->next_buffer_file_offset != -1 );
            n = refer_events_buffer(files[ifd]->fd, b->next_buffer_file_offset);
             if( NULL == n ) {
                fprintf(stderr, "Unable to read thread entry: Profile file %s broken\n",
                        files[ifd]->filename);
                release_events_buffer( b );
                return -1;
            }
            assert( PROFILING_BUFFER_TYPE_THREAD == n->buffer_type );
            release_events_buffer( b );
            b = n;

            nbthis = b->this_buffer.nb_threads;

            pos = 0;
        }
    }

    release_events_buffer( b );
    return 0;
}

static char *getThreadContainerIdentifier( const char *prefix, const char *identifier ) {
    const char *r = identifier + strlen(identifier) - 1;
    char *ret;

    while ( *r != ' ' )
        r--;

    asprintf( &ret, "%sT%s", prefix, r+1);
    return ret;
}

static int dague_profiling_dump_paje( const char* filename )
{
    unsigned int i, ifd, tid;
    dague_time_t relative = ZERO_TIME;
    dague_thread_profiling_t* profile;
    dague_list_item_t *it;
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
    addContType ("CT_T", "CT_P", "Thread");
    addStateType ("ST_TS", "CT_T", "Thread State");
    addLinkType ("LT_TL", "Split Event Link", "CT_P", "CT_T", "CT_T");

    addEntityValue ("Wait", "ST_TS", "Waiting", GTG_LIGHTGREY);
    addContainer (0.00000, "Appli", "CT_Appli", "0", hr_id, "");

    for(i = 0; i < dague_prof_keys_count; i++) {
        color_code = strtoul( dague_prof_keys[i].attributes, NULL, 16);
        color = gtg_color_create(dague_prof_keys[i].name,
                                 GTG_COLOR_GET_RED(color_code),
                                 GTG_COLOR_GET_GREEN(color_code),
                                 GTG_COLOR_GET_BLUE(color_code));
        sprintf(dico_id, "K-%u", i);
        addEntityValue (dico_id, "ST_TS", dague_prof_keys[i].name, color);
        gtg_color_free(color);
    }

    relative = files[0]->head.start_time;
    for(ifd = 1; ifd < nb_files; ifd++) {
        if( time_less(files[ifd]->head.start_time, relative) ) {
            relative = files[ifd]->head.start_time;
        }
    }
    if( ifd > 0 ) {
        dague_time_t max_time;
        uint64_t delta_time;

        delta_time = 0;
        max_time = relative;
        for(ifd = 0; ifd < nb_files; ifd++) {
            delta_time += diff_time(relative, files[ifd]->head.start_time);
            if( time_less(max_time, files[ifd]->head.start_time) ) {
                max_time = files[ifd]->head.start_time;
            }
        }
        fprintf(stderr, "-- Time jitter is bounded by %lu "TIMER_UNIT", average is %g "TIMER_UNIT"\n",
                diff_time(relative, max_time),
                (double)delta_time / (double)nb_files);
    }

    for(ifd = 0; ifd < nb_files; ifd++) {
        if( load_thread_heads(ifd) == -1 )
            return -1;

        sprintf(name, "MPI-%d", files[ifd]->head.rank);
        sprintf(cont_mpi_name, "MPI-%d", files[ifd]->head.rank);
        addContainer (0.00000, cont_mpi_name, "CT_P", "Appli", name, cont_mpi_name);
        
        for(it = DAGUE_LIST_ITERATOR_FIRST(&threads), tid = 0;
            it != DAGUE_LIST_ITERATOR_END(&threads);
            it = DAGUE_LIST_ITERATOR_NEXT(it), tid++) {
            profile = (dague_thread_profiling_t*)it;

            cont_thread_name = getThreadContainerIdentifier( cont_mpi_name, profile->hr_id );

            {
                int l;
                l = 3 + snprintf(NULL, 0, "#  %s", cont_thread_name);
                if( l > stat_columns[0] )
                    stat_columns[0] = l;

                dico_stat[ifd][tid].name = strdup(cont_thread_name);
                current_stat = dico_stat[ifd][tid].stats;
            }

            addContainer (0.00000, cont_thread_name, "CT_T", cont_mpi_name, profile->hr_id, cont_thread_name);

            dague_profiling_dump_one_paje(profile, cont_thread_name, files[ifd]->fd, files[ifd]->head.start_time);
        }
    
        free_thread_heads();
    }

    return 0;
}

static int open_files(int argc, char *argv[])
{
    int fd, i, j, p, n;
    dague_profiling_buffer_t dummy_events_buffer;
    
    files = (dbp_file_t**)malloc( (argc-1) * sizeof(dbp_file_t*));
    n = 0;
    for(i = 1; i < argc; i++) {
        fd = open(argv[i], O_RDONLY);
        if( fd == -1 ) {
            fprintf(stderr, "Unable to open %s: %s -- skipped\n", argv[i], strerror(errno));
            continue;
        }
        files[n] = (dbp_file_t*)malloc( sizeof(dbp_file_t) );
        files[n]->fd = fd;
        if( read( fd, &(files[n]->head), sizeof(dague_profiling_binary_file_header_t) ) != sizeof(dague_profiling_binary_file_header_t) ) {
            fprintf(stderr, "File %s does not seem to be a correct DAGUE Binary Profile, ignored\n",
                    argv[i]);
            close(fd);
            free(files[n]);
            continue;
        }
        if( strncmp( files[n]->head.magick, DAGUE_PROFILING_MAGICK, 24 ) ) {
            fprintf(stderr, "File %s does not seem to be a correct DAGUE Binary Profile, ignored\n",
                    argv[i]);
            close(fd);
            free(files[n]);
            continue;
        }
        if( files[n]->head.byte_order != 0x0123456789ABCDEF ) {
            fprintf(stderr, "The profile in file %s has been generated with a different byte ordering. File ignored\n",
                    argv[i]);
            close(fd);
            free(files[n]);
            continue;
        }
        files[n]->filename = argv[i];
        if( n > 0 ) {
            if( strncmp(files[n]->head.hr_id, files[0]->head.hr_id, 128) ) {
                fprintf(stderr, "The profile in file %s has unique id %s, which is not compatible with id %s of file %s. File ignored.\n",
                        files[n]->filename, files[n]->head.hr_id,
                        files[0]->head.hr_id, files[0]->filename);
                close(fd);
                free(files[n]);
                continue;
            }

            if( files[n]->head.profile_buffer_size != files[0]->head.profile_buffer_size ) {
                fprintf(stderr, "The profile in file %s has a buffer size of %d, which is not compatible with the buffer size %d of file %s. File ignored.\n",
                        files[n]->filename, files[n]->head.profile_buffer_size,
                        files[0]->head.profile_buffer_size, files[0]->filename);
                close(fd);
                free(files[n]);
                continue;
            }


            if( files[n]->head.worldsize != files[0]->head.worldsize ) {
                fprintf(stderr, "The profile in file %s has a world size of %d, which is not compatible with the world size %d of file %s. File ignored.\n",
                        files[n]->filename, files[n]->head.worldsize,
                        files[0]->head.worldsize, files[0]->filename);
                close(fd);
                free(files[n]);
                continue;
            }
        }
        n++;
    }
    
    if( files[0]->head.worldsize > n ) {
        fprintf(stderr, "The profile in file %s has a world size of %d, but only %d files can be read in input. The trace will be truncated\n",
                        files[0]->filename, files[0]->head.worldsize,
                        n);
    } else if( files[0]->head.worldsize < n ) {
        fprintf(stderr, "The profile in file %s has a world size of %d, but %d files should be read in input. The trace will be... Strange...\n",
                        files[0]->filename, files[0]->head.worldsize,
                        n);
    } else {
        for(i = 0; i < n; i++) {
            p = 0;
            for(j = 0; j < n; j++) {
                if( files[j]->head.rank == i )
                    p++;
            }
            if( p != 1 ) {
                fprintf(stderr, "The rank %d appears %d times in this collection of profiles... The trace will be... Strange...\n",
                        i, p);
            }
        }
    }

    nb_files = n;
    event_buffer_size = files[0]->head.profile_buffer_size;
    event_avail_space = event_buffer_size -
        ( (char*)&dummy_events_buffer.buffer[0] - (char*)&dummy_events_buffer);

    return n;
}

static int reconciliate_dictionnary(void)
{
    dague_profiling_buffer_t *first_dico, *dico, *next;
    dague_profiling_key_buffer_t *a, *b;
    int i, nb, nbthis, pos;
    dague_prof_keys_count = 0;

    for(i = 1; i < nb_files; i++) {
        if( files[0]->head.dictionary_size != files[i]->head.dictionary_size ) {
            fprintf(stderr,
                    "Current version of dbp2paje does not allow binary profile files to have different dictionary entries.\n"
                    " %s has %d entries, while %s has %d\n",
                    files[0]->filename, files[0]->head.dictionary_size,
                    files[i]->filename, files[i]->head.dictionary_size);
            return -1;
        }

        first_dico = refer_events_buffer( files[0]->fd, files[0]->head.dictionary_offset );
        if( NULL == first_dico ) {
            fprintf(stderr, "Unable to read dictionary entry: Profile file %s broken\n",
                    files[0]->filename);
            return -1;
        }
        assert( PROFILING_BUFFER_TYPE_DICTIONARY == first_dico->buffer_type );

        dico = refer_events_buffer( files[i]->fd, files[i]->head.dictionary_offset );
        if( NULL == dico ) {
            fprintf(stderr, "Unable to read dictionary entry: Profile file %s broken\n",
                    files[i]->filename);
            release_events_buffer(first_dico);
            return -1;
        }
        assert( PROFILING_BUFFER_TYPE_DICTIONARY == dico->buffer_type );

        if( dico->this_buffer.nb_dictionary_entries != first_dico->this_buffer.nb_dictionary_entries ) {
            fprintf(stderr,
                    "Current version of dbp2paje does not allow binary profile files to have different dictionary entries.\n"
                    " %s has %ld entries in some dictionnary buffer, while %s has %ld\n",
                    files[0]->filename, first_dico->this_buffer.nb_dictionary_entries,
                    files[i]->filename, dico->this_buffer.nb_dictionary_entries);
            release_events_buffer( dico );
            release_events_buffer( first_dico );
            return -1;
        }

        nb = files[0]->head.dictionary_size;
        nbthis = first_dico->this_buffer.nb_dictionary_entries;
        pos = 0;
        while( nb > 0 ) {
            a = (dague_profiling_key_buffer_t*)&first_dico->buffer[pos];
            b = (dague_profiling_key_buffer_t*)&dico->buffer[pos];

            if( strncmp(a->name, b->name, 64) ) {
                fprintf(stderr,
                        "Current version of dbp2paje does not allow binary profile files to have different dictionary entries.\n"
                        " %s has an entry number %d with name %s, while the corresponding entry in %s has name %s\n",
                        files[0]->filename, files[0]->head.dictionary_size-nb, a->name,
                        files[i]->filename, b->name);
                release_events_buffer( dico );
                release_events_buffer( first_dico );
                return -1;
            }

            if( strncmp(a->attributes, b->attributes, 128) ) {
                fprintf(stderr,
                        "Current version of dbp2paje does not allow binary profile files to have different dictionary entries.\n"
                        " %s has an entry number %d with attributes %s, while the corresponding entry in %s has attributes %s\n",
                        files[0]->filename, files[0]->head.dictionary_size-nb, a->attributes,
                        files[i]->filename, b->attributes);
                release_events_buffer( dico );
                release_events_buffer( first_dico );
                return -1;
            }

            if( a->keyinfo_length != b->keyinfo_length ) {
                fprintf(stderr,
                        "Current version of dbp2paje does not allow binary profile files to have different dictionary entries.\n"
                        " %s has an entry number %d of %d bytes for its info, while %s's entry has %d bytes for its info\n",
                        files[0]->filename, files[0]->head.dictionary_size-nb, a->keyinfo_length, files[i]->filename, b->keyinfo_length);
                release_events_buffer( dico );
                release_events_buffer( first_dico );
                return -1;
            }

            if( a->keyinfo_convertor_length != b->keyinfo_convertor_length ) {
                fprintf(stderr,
                        "Current version of dbp2paje does not allow binary profile files to have different dictionary entries.\n"
                        " %s has an entry number %d of %d bytes for its convertor, while %s's entry has %d bytes for its convertor\n",
                        files[0]->filename, files[0]->head.dictionary_size-nb, a->keyinfo_convertor_length,
                        files[i]->filename, b->keyinfo_convertor_length);
                release_events_buffer( dico );
                release_events_buffer( first_dico );
                return -1;
            }

            if( strncmp(a->convertor, b->convertor, a->keyinfo_convertor_length) ) {
                fprintf(stderr,
                        "Current version of dbp2paje does not allow binary profile files to have different dictionary entries.\n"
                        " %s has an entry number %d with convertor '%s', while the corresponding entry in %s has convertor '%s'\n",
                        files[0]->filename, files[0]->head.dictionary_size-nb, a->convertor,
                        files[i]->filename, b->convertor);
                release_events_buffer( dico );
                release_events_buffer( first_dico );
                return -1;
            }

            pos += a->keyinfo_convertor_length - 1 + sizeof(dague_profiling_key_buffer_t);
            nb--;
            nbthis--;

            if( nb > 0 && nbthis == 0 ) {
                next = refer_events_buffer( files[0]->fd, first_dico->next_buffer_file_offset );
                if( NULL == next ) {
                    fprintf(stderr, "Unable to read dictionary entry: Profile file %s broken\n",
                            files[0]->filename);
                    release_events_buffer( dico );
                    release_events_buffer( first_dico );
                    return -1;
                }
                assert( PROFILING_BUFFER_TYPE_DICTIONARY == first_dico->buffer_type );
                release_events_buffer( first_dico );
                first_dico = next;

                next = refer_events_buffer( files[i]->fd, dico->next_buffer_file_offset );
                if( NULL == next ) {
                    fprintf(stderr, "Unable to read dictionary entry: Profile file %s broken\n",
                            files[i]->filename);
                    release_events_buffer( first_dico );
                    release_events_buffer( dico );
                    return -1;
                }
                assert( PROFILING_BUFFER_TYPE_DICTIONARY == dico->buffer_type );
                release_events_buffer( dico );
                dico = next;

                pos = 0;
            }
            assert( nb > 0 ||
                    ( first_dico->next_buffer_file_offset == -1 &&
                      dico->next_buffer_file_offset == -1 ) );
            assert( pos < event_avail_space );

        }
        release_events_buffer( dico );
        release_events_buffer( first_dico );
    }

    /* Dictionaries match: take the first in memory */
    first_dico = refer_events_buffer( files[0]->fd, files[0]->head.dictionary_offset );
    if( NULL == first_dico ) {
        fprintf(stderr, "Unable to read dictionary entry: Profile file %s broken\n",
                files[0]->filename);
        return -1;
    }
    assert( PROFILING_BUFFER_TYPE_DICTIONARY == first_dico->buffer_type );

    dague_prof_keys_count = files[0]->head.dictionary_size;
    dague_prof_keys = (dague_profiling_key_t *)calloc(dague_prof_keys_count, sizeof(dague_profiling_key_t));
    nb = files[0]->head.dictionary_size;
    nbthis = first_dico->this_buffer.nb_dictionary_entries;
    pos = 0;
    while( nb > 0 ) {
        a = (dague_profiling_key_buffer_t*)&first_dico->buffer[pos];

        dague_prof_keys[ dague_prof_keys_count - nb ].name = malloc( 64 );
        strncpy(dague_prof_keys[ dague_prof_keys_count - nb ].name, a->name, 64);
        assert( strlen(a->attributes) > 6 );
        dague_prof_keys[ dague_prof_keys_count - nb ].attributes = malloc( 128 );
        strncpy(dague_prof_keys[ dague_prof_keys_count - nb ].attributes, ((char*)a->attributes) + strlen(a->attributes) - 6, 128 );
        dague_prof_keys[ dague_prof_keys_count - nb ].convertor = (char*)malloc(a->keyinfo_convertor_length+1);
        memcpy(dague_prof_keys[ dague_prof_keys_count - nb ].convertor,
               a->convertor,
               a->keyinfo_convertor_length);
        dague_prof_keys[ dague_prof_keys_count - nb ].convertor[a->keyinfo_convertor_length] = '\0';
        dague_prof_keys[ dague_prof_keys_count - nb ].info_length = a->keyinfo_length;

        pos += a->keyinfo_convertor_length - 1 + sizeof(dague_profiling_key_buffer_t);
        nb--;
        nbthis--;

        if( nb > 0 && nbthis == 0 ) {
            next = refer_events_buffer( files[0]->fd, first_dico->next_buffer_file_offset );
            if( NULL == next ) {
                fprintf(stderr, "Unable to read dictionary entry: Profile file %s broken\n",
                        files[0]->filename);
                release_events_buffer( first_dico );
                return -1;
            }
            assert( PROFILING_BUFFER_TYPE_DICTIONARY == first_dico->buffer_type );
            release_events_buffer( first_dico );
            first_dico = next;

            pos = 0;
        }
    }
    release_events_buffer( first_dico );

    return 0;
}

int main(int argc, char *argv[])
{
    int i, j, k;

    open_files(argc, argv);

    if( nb_files <= 0 )
        return 1;
    if( reconciliate_dictionnary() == -1 )
        return 1;

    dico_stat = (thread_stat_t**)malloc(nb_files * sizeof(thread_stat_t*));
    for(i = 0; i < nb_files; i++) {
        dico_stat[i] = (thread_stat_t*)calloc(files[i]->head.nb_threads, sizeof(thread_stat_t));
        for(j = 0; j < files[i]->head.nb_threads; j++) {
            dico_stat[i][j].stats = (dico_stat_t*)calloc(dague_prof_keys_count, sizeof(dico_stat_t));
        }
    }
    stat_columns = (int*)calloc(dague_prof_keys_count+1, sizeof(int));

    for(i = 0; i < nb_files; i++) {
        int l;
        l = 3 + snprintf(NULL, 0, "#%s Rank %d/%d", files[i]->head.hr_id, files[i]->head.rank, files[i]->head.worldsize);
        if( l > stat_columns[0] )
            stat_columns[0] = l;
    }

    dague_profiling_dump_paje( "out" );

    for(k = 0 ; k < dague_prof_keys_count; k = k+1 ) {
        stat_columns[k+1] = stat_columns[k] + 2 +strlen(dague_prof_keys[k].name);
    }

    printf("#Stats:\n");
    printf("#Thread   ");
    for(k = 0 ; k < dague_prof_keys_count; k = k+1 ) {
        printf("[%dG%s", stat_columns[k], dague_prof_keys[k].name);
    }
    printf("\n");
    for(i = 0; i < nb_files; i++) {
        printf("#%s Rank %d/%d\n", files[i]->head.hr_id, files[i]->head.rank, files[i]->head.worldsize);
        for(j = 0; j < files[i]->head.nb_threads; j++) {
            printf("#  %s", dico_stat[i][j].name);

            for(k = 0; k < dague_prof_keys_count; k++) {
                printf("[%dG[%dm%d[0m/[%dm%d[0m/[%dm%d[0m",
                       stat_columns[k],
                       dico_stat[i][j].stats[k].nb_matched_samethread > 0 ? 32 : 2,
                       dico_stat[i][j].stats[k].nb_matched_samethread,
                       dico_stat[i][j].stats[k].nb_matched_differentthread > 0 ? 35 : 2,
                       dico_stat[i][j].stats[k].nb_matched_differentthread,
                       dico_stat[i][j].stats[k].nb_matcherror > 0 ? 31 : 2,
                       dico_stat[i][j].stats[k].nb_matcherror);
            }

            printf("\n");
        }
    }

    endTrace();

    return 0;
}
