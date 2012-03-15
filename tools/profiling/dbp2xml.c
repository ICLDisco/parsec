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
#include "debug.h"

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
static dague_profiling_info_t *dague_profiling_infos = NULL;

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

static void                        iterator_delete(dague_profiling_iterator_t *it)
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
    const dague_profiling_output_t *event;
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
        for( event = iterator_first( pit );
             NULL != event;
             event = iterator_next( pit ) ) {
            DEBUG(("TRACE %d/%lu on %p (timestamp %llu)\n", event->event.key, event->event.id, profile,
                   diff_time(zero, event->event.timestamp)));

        }
        iterator_delete(pit);
    };
}

static int dague_profiling_dump_one_xml( const dague_thread_profiling_t *profile, 
                                         int backend_fd,
                                         FILE *out,
                                         dague_time_t relative )
{
    unsigned int pos, displayed_key;
    uint64_t start, end;
    static int displayed_error_message = 0;
    char *infostr = malloc(4);
    int event_not_found;
    dague_thread_profiling_t *op;
    const dague_profiling_output_t *start_event;
    dague_profiling_output_t *end_event = NULL;
    size_t end_event_size = 0;
    dague_profiling_iterator_t *pit, *nit;

    for( pos = 0; pos < dague_prof_keys_count; pos++ ) {
        displayed_key = 0;
        pit = iterator_new( profile, backend_fd );
        for( start_event = iterator_first( pit );
             NULL != start_event;
             start_event = iterator_next( pit ) ) {

            /* if not my current start_idx key, ignore */
            if( start_event->event.key != START_KEY(pos) ) {
                continue;
            }
            
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
                    } else {
                        iterator_delete(nit);
                    }
                });

                /* Couldn't find the end, or no id. Bad. */
                if( event_not_found ) {

                    WARNING(("Profiling: end event of key %u (%s) id %lu was not found for ID %s\n",
                             END_KEY(pos), dague_prof_keys[pos].name, start_event->event.id, profile->hr_id));

                    if( !displayed_error_message ) {
                        dump_whole_trace( backend_fd );
                        displayed_error_message = 1;
                    }

                    continue;
                }
            }

            start = diff_time( relative, start_event->event.timestamp );
            end = diff_time( relative, end_event->event.timestamp );

            if( displayed_key == 0 ) {
                fprintf(out, "               <KEY ID=\"%u\">\n", pos);
                displayed_key = 1;
            }
            
            fprintf(out, 
                    "                  <EVENT>\n"
                    "                     <ID>%lu</ID>\n"
                    "                     <START>%"PRIu64"</START>\n"
                    "                     <END>%"PRIu64"</END>\n",
                    start_event->event.id,
                    start, end);

            if( EVENT_HAS_INFO(start_event) ) {
                /** TODO fprintf(out, "       <INFO>%s</INFO>\n", infostr); */
            } 
            if( EVENT_HAS_INFO(end_event) ) {
                /** TODO fprintf(out, "       <INFO ATEND=\"true\">%s</INFO>\n", infostr); */
            } 
            fprintf(out, "                  </EVENT>\n");
        }
        if( displayed_key ) {
            fprintf(out, "              </KEY>\n");
        }

        iterator_delete(pit);
    }

    free(infostr);

    return 0;
}

static int64_t find_last_events_buffer_offset( int fd, dague_thread_profiling_t *profile )
{
    (void)profile;
    (void)fd;
    return -1;
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
    int i;
    int pos;

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

static int load_thread_heads(int ifd) {
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

static int dague_profiling_dump_xml( const char* filename )
{
    unsigned int i, ifd;
    int foundone;
    dague_time_t relative = ZERO_TIME, latest = ZERO_TIME;
    dague_thread_profiling_t* profile;
    FILE* tracefile;
    dague_profiling_info_t *info;
    dague_profiling_buffer_t *bstart, *bend;
    int64_t last_events_buffer_offset;
    dague_list_item_t *it;
 
    tracefile = fopen(filename, "w");
    if( NULL == tracefile ) {
        return -1;
    }

    fprintf(tracefile,
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<PROFILING>\n"
            " <IDENTIFIER><![CDATA[%s]]></IDENTIFIER>\n"
            "  <INFOS>\n", hr_id);
    for(info = dague_profiling_infos; info != NULL; info = info->next ) {
        fprintf(tracefile, "    <INFO NAME=\"%s\">%s</INFO>\n", info->key, info->value);
    }

    fprintf(tracefile,
            "  </INFOS>\n"
            "  <DICTIONARY>\n");

    for(i = 0; i < dague_prof_keys_count; i++) {
        fprintf(tracefile,
                "   <KEY ID=\"%u\">\n"
                "    <NAME>%s</NAME>\n"
                "    <ATTRIBUTES><![CDATA[%s]]></ATTRIBUTES>\n"
                "   </KEY>\n",
                i, dague_prof_keys[i].name, dague_prof_keys[i].attributes);
    }
    fprintf(tracefile, " </DICTIONARY>\n");

    foundone = 0;
   
    for(ifd = 0; ifd < nb_files; ifd++) {
        if( load_thread_heads(ifd) == -1 )
            return -1;

        for( it = DAGUE_LIST_ITERATOR_FIRST(&threads);
             it != DAGUE_LIST_ITERATOR_END(&threads);
             it = DAGUE_LIST_ITERATOR_NEXT(it) ) {
            profile = (dague_thread_profiling_t*)it;
            
            if( NULL == (bstart = refer_events_buffer( files[ifd]->fd, profile->first_events_buffer_offset )) )
                continue;
            last_events_buffer_offset = find_last_events_buffer_offset( files[ifd]->fd, profile );
            if( NULL == (bend = refer_events_buffer( files[ifd]->fd, last_events_buffer_offset )) ) {
                release_events_buffer( bstart );
                continue;
            }
            if( !foundone ) {
                relative = ((dague_profiling_output_t *)(bstart->buffer))->event.timestamp;
                latest   = ((dague_profiling_output_t*)(bend->buffer))->event.timestamp;
                foundone = 1;
            } else {
                if( time_less(((dague_profiling_output_t *)(bstart->buffer))->event.timestamp, relative) ) {
                    relative = ((dague_profiling_output_t *)(bstart->buffer))->event.timestamp;
                }
                if( time_less( latest, ((dague_profiling_output_t*)(bend->buffer))->event.timestamp) ) {
                    latest = ((dague_profiling_output_t*)(bend->buffer))->event.timestamp;
                }
            }
            release_events_buffer( bstart );
            release_events_buffer( bend );
        }

        free_thread_heads();
    }

    fprintf(tracefile, "   <DISTRIBUTED_PROFILE TOTAL_DURATION=\"%"PRIu64"\" TIME_UNIT=\""TIMER_UNIT"\">\n",
            diff_time(relative, latest));
    for(ifd = 0; ifd < nb_files; ifd++) {
        if( load_thread_heads(ifd) == -1 )
            return -1;

        fprintf(tracefile,
                "      <NODE FILEID=\"%s\">\n", files[ifd]->filename);
        
        fprintf(tracefile, "         <PROFILES TOTAL_DURATION=\"%"PRIu64"\" TIME_UNIT=\""TIMER_UNIT"\">\n",
                diff_time(relative, latest));
        
        for(it = DAGUE_LIST_ITERATOR_FIRST(&threads);
            it != DAGUE_LIST_ITERATOR_END(&threads);
            it = DAGUE_LIST_ITERATOR_NEXT(it)) {
            profile = (dague_thread_profiling_t*)it;
                
            fprintf(tracefile, 
                    "            <THREAD>\n"
                    "               <IDENTIFIER><![CDATA[%s]]></IDENTIFIER>\n", profile->hr_id);
            dague_profiling_dump_one_xml(profile, files[ifd]->fd, tracefile, relative);
            fprintf(tracefile, 
                    "            </THREAD>\n");
        }
    
        free_thread_heads();

        fprintf(tracefile, 
                "         </PROFILES>\n"
                "      </NODE>\n");
    }
    fprintf(tracefile, 
            "   </DISTRIBUTED_PROFILE>\n"
            "</PROFILING>\n");
    fclose(tracefile);
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
        if( read( fd, &(files[n]->head), sizeof(dbp_file_t) ) != sizeof(dbp_file_t) ) {
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
                    "Current version of dbp2xml does not allow binary profile files to have different dictionary entries.\n"
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
                    "Current version of dbp2xml does not allow binary profile files to have different dictionary entries.\n"
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
                        "Current version of dbp2xml does not allow binary profile files to have different dictionary entries.\n"
                        " %s has an entry number %d with name %s, while the corresponding entry in %s has name %s\n",
                        files[0]->filename, files[0]->head.dictionary_size-nb, a->name,
                        files[i]->filename, b->name);
                release_events_buffer( dico );
                release_events_buffer( first_dico );
                return -1;
            }

            if( strncmp(a->attributes, b->attributes, 128) ) {
                fprintf(stderr, 
                        "Current version of dbp2xml does not allow binary profile files to have different dictionary entries.\n"
                        " %s has an entry number %d with attributes %s, while the corresponding entry in %s has attributes %s\n",
                        files[0]->filename, files[0]->head.dictionary_size-nb, a->attributes,
                        files[i]->filename, b->attributes);
                release_events_buffer( dico );
                release_events_buffer( first_dico );
                return -1;
            }

            if( a->keyinfo_length != b->keyinfo_length ) {
                fprintf(stderr, 
                        "Current version of dbp2xml does not allow binary profile files to have different dictionary entries.\n"
                        " %s has an entry number %d of %d bytes for its info, while %s's entry has %d bytes for its info\n",
                        files[0]->filename, files[0]->head.dictionary_size-nb, a->keyinfo_length, files[i]->filename, b->keyinfo_length);
                release_events_buffer( dico );
                release_events_buffer( first_dico );
                return -1;
            }

            if( a->keyinfo_convertor_length != b->keyinfo_convertor_length ) {
                fprintf(stderr, 
                        "Current version of dbp2xml does not allow binary profile files to have different dictionary entries.\n"
                        " %s has an entry number %d of %d bytes for its convertor, while %s's entry has %d bytes for its convertor\n",
                        files[0]->filename, files[0]->head.dictionary_size-nb, a->keyinfo_convertor_length, 
                        files[i]->filename, b->keyinfo_convertor_length);
                release_events_buffer( dico );
                release_events_buffer( first_dico );
                return -1;
            }
            
            if( strncmp(a->convertor, b->convertor, a->keyinfo_convertor_length) ) {
                fprintf(stderr, 
                        "Current version of dbp2xml does not allow binary profile files to have different dictionary entries.\n"
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
    if( NULL == dico ) {
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
        
        dague_prof_keys[ dague_prof_keys_count - nb ].name = strdup(a->name);
        dague_prof_keys[ dague_prof_keys_count - nb ].attributes = strdup(a->attributes);
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
    open_files(argc, argv);
    if( nb_files <= 0 )
        return 1;
    if( reconciliate_dictionnary() == -1 )
        return 1;
    dague_profiling_dump_xml( "out.xml" );
    return 0;
}
