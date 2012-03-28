#include "dague_config.h"
#undef HAVE_MPI

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <stdarg.h>

#include "profiling.h"
#include "dbp.h"
#include "dbpreader.h"

#if defined(DAGUE_DEBUG_VERBOSE1)
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

#include <GTG.h>
#include <GTGPaje.h>

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

typedef struct {
    dague_list_item_t super;
    uint64_t        id;
    uint64_t        start;
    uint64_t        end;
    int             key;
    const dbp_thread_t   *start_thread;
    const dbp_thread_t   *end_thread;
    size_t          start_info_size;
    size_t          end_info_size;
    char            infos[1];
} consolidated_event_t;

static char *getThreadContainerIdentifier( const char *prefix, const char *identifier )
{
    const char *r = identifier + strlen(identifier) - 1;
    char *ret;

    while( *r != ' ' )
        r--;

    asprintf( &ret, "%sT%s", prefix, r+1);
    return ret;
}

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
            if( ((cev->end > lev->start) ||
                ((prev != NULL) && (cev->start < prev->end) )) &&
                (cev->start_thread == cev->end_thread) ) {
                broken = 1;
            }
            dague_list_nolock_add_before( list,
                                          it,
                                          (dague_list_item_t*)cev );
            return broken;
        }
        prev = lev;
    }
    if( (prev != NULL) && (cev->start < prev->end) &&
        (cev->start_thread == cev->end_thread) ) {
        broken = 1;
    }
    dague_list_nolock_push_back( list, (dague_list_item_t*)cev );
    return broken;
}

static uint64_t *step_height(dague_list_t *list, int *level)
{
    dague_list_item_t *e;
    consolidated_event_t *cev;
    int nb_steps = 0, s;
    uint64_t *dates = NULL;
    
    for( e = DAGUE_LIST_ITERATOR_FIRST(list);
         e != DAGUE_LIST_ITERATOR_END(list);
         e = DAGUE_LIST_ITERATOR_NEXT(e) ) {
        cev = (consolidated_event_t*)e;
        if( cev->start_thread == cev->end_thread ) {
            for(s = 0; s < nb_steps; s++) {
                if( dates[s] <= cev->start ) {
                    dates[s] = cev->end;
                    break;
                }
            }
            if (s == nb_steps) {
                nb_steps++;
                dates = (uint64_t*)realloc(dates, nb_steps * sizeof(uint64_t));
                dates[s] = cev->end;
            }
        }
    }
    memset(dates, 0, nb_steps * sizeof(uint64_t));
    *level = nb_steps;
    return dates;
}

static int dump_one_paje( const dbp_multifile_reader_t *dbp,
                          const dbp_thread_t *th,
                          const char *cont_mpi_name,
                          const char *cont_thread_name )
{
    unsigned int key, broken = 0;
    uint64_t start, end;
    int s;
    char keyid[64];
    dbp_event_iterator_t *pit, *nit;
    dague_list_t consolidated_events;
    consolidated_event_t *cev;
    static int linkuid = 0;
    char linkid[64];
    char cont_step_name[64];
    dague_time_t relative;
    const dbp_event_t *e, *g;
    char *cont_src;
    char *cont_dst;
    uint64_t *steps_end_dates;
    int nb_steps;

    relative = dbp_reader_min_date(dbp);

    pit = dbp_iterator_new_from_thread( th );
    dague_list_construct( &consolidated_events );
    while( (e = dbp_iterator_current(pit)) != NULL ) {
        if( KEY_IS_START( dbp_event_get_key(e) ) ) {
                
            key = BASE_KEY(dbp_event_get_key(e));
            nit = dbp_iterator_find_matching_event_all_threads(pit);

            if( NULL == nit ) {
                /* Argh, couldn't find the end in this trace */
                WARNING(("   Event of class %s id %d at %lu does not have a match anywhere\n",
                         dbp_dictionary_name(dbp_reader_get_dictionary(dbp, BASE_KEY(dbp_event_get_key(e)))),
                         dbp_event_get_id(e),
                         diff_time(relative, dbp_event_get_timestamp(e))));
                
                current_stat[ key ].nb_matcherror++;
            } else {
                g = dbp_iterator_current(nit);

                if( dbp_iterator_thread(nit) != dbp_iterator_thread(pit) ) {
                    current_stat[ key ].nb_matchthreads++;
                }
                current_stat[ key ].nb_matchsuccess++;
                
                start = diff_time( relative, dbp_event_get_timestamp( e ) );
                end = diff_time( relative, dbp_event_get_timestamp( g ) );
                
                assert( start <= end );
                
                cev = (consolidated_event_t*)malloc(sizeof(consolidated_event_t) +
                                                    dbp_event_info_len(e, dbp) +
                                                    dbp_event_info_len(g, dbp) );
                cev->id = dbp_event_get_id(e);
                cev->start = start;
                cev->end = end;
                cev->start_thread = dbp_iterator_thread(pit);
                cev->end_thread = dbp_iterator_thread(nit);
                cev->key = key;
                cev->start_info_size = dbp_event_info_len(e, dbp);
                cev->end_info_size = dbp_event_info_len(g, dbp);
                memcpy(cev->infos, dbp_event_get_info( e ), cev->start_info_size);
                memcpy(cev->infos + cev->start_info_size, dbp_event_get_info( g ), cev->end_info_size);
                
                broken = merge_event( &consolidated_events, cev ) || broken;
                dbp_iterator_delete(nit);
            }
        }
        dbp_iterator_next(pit);
    }
    dbp_iterator_delete(pit);

    if( broken ) {
        steps_end_dates = step_height(&consolidated_events, &nb_steps);
    } else {
        nb_steps = 1;
        steps_end_dates = (uint64_t*)calloc(1, sizeof(uint64_t));
    }
    for(s = 0; s < nb_steps; s++) {
        sprintf(cont_step_name, "%s-%d", cont_thread_name, s);
        addContainer(0.00000, cont_step_name, "CT_S", cont_thread_name, " ", "");
    }

    while( NULL != (cev = (consolidated_event_t*)dague_list_nolock_pop_front( &consolidated_events ) ) ) {
        sprintf(keyid, "K-%d", cev->key);
        if( cev->start_thread == cev->end_thread ) {
            for(s = 0; s < nb_steps; s++) {
                if( steps_end_dates[s] <= cev->start ) {
                    steps_end_dates[s] = cev->end;
                    break;
                }
            }
            assert( s < nb_steps );
            sprintf(cont_step_name, "%s-%d", cont_thread_name, s);
            pajeSetState2( ((double)cev->start) * 1e-3, "ST_TS", cont_step_name, keyid );
            pajeSetState2( ((double)cev->end) * 1e-3, "ST_TS", cont_step_name, "Wait");
        } else {
            sprintf(linkid, "L-%d", linkuid);
            linkuid++;
            cont_src = getThreadContainerIdentifier( cont_mpi_name, dbp_thread_get_hr_id(cev->start_thread) );
            cont_dst = getThreadContainerIdentifier( cont_mpi_name, dbp_thread_get_hr_id(cev->end_thread) );
            startLink( ((double)cev->start) * 1e-3, "LT_TL", cont_mpi_name, cont_src, cont_dst, keyid, linkid);
            endLink( ((double)cev->end) * 1e-3, "LT_TL", cont_mpi_name, cont_src, cont_dst, keyid, linkid);
            free(cont_src);
            free(cont_dst);
        }
        free(cev);
    }

    free(steps_end_dates);
    dague_list_destruct( &consolidated_events );
    
    return 0;
}

static int dague_profiling_dump_paje( const char* filename, const dbp_multifile_reader_t *dbp )
{
    unsigned int i, t, ifd;
    dague_time_t relative = ZERO_TIME;
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
    addContType ("CT_T", "CT_P", "Thread");
    addContType ("CT_S", "CT_T", "State");
    addStateType ("ST_TS", "CT_S", "Thread State");
    addLinkType ("LT_TL", "Split Event Link", "CT_P", "CT_T", "CT_T");

    addEntityValue ("Wait", "ST_TS", "Waiting", GTG_LIGHTGREY);
    addContainer (0.00000, "Appli", "CT_Appli", "0", dbp_file_hr_id(dbp_reader_get_file(dbp, 0)), "");

    for(i = 0; i < dbp_reader_nb_dictionary_entries(dbp); i++) {
        dico = dbp_reader_get_dictionary(dbp, i);
        color_code = strtoul( dbp_dictionary_attributes(dico), NULL, 16);
        color = gtg_color_create(dbp_dictionary_name(dico),
                                 GTG_COLOR_GET_RED(color_code),
                                 GTG_COLOR_GET_GREEN(color_code),
                                 GTG_COLOR_GET_BLUE(color_code));
        sprintf(dico_id, "K-%u", i);
        addEntityValue (dico_id, "ST_TS", dbp_dictionary_name(dico), color);
        gtg_color_free(color);
    }

    if( dbp_reader_nb_files(dbp) > 1 ) {
        dague_time_t max_time;
        uint64_t delta_time;

        delta_time = 0;
        max_time = dbp_reader_min_date(dbp);
        for(ifd = 1; ifd < dbp_reader_nb_files(dbp); ifd++) {
            file = dbp_reader_get_file(dbp, ifd);
            delta_time += diff_time(relative, dbp_file_get_min_date( file ));
            if( time_less(max_time, dbp_file_get_min_date( file )) ) {
                max_time = dbp_file_get_min_date( file );
            }
        }
        fprintf(stderr, "-- Time jitter is bounded by %lu "TIMER_UNIT", average is %g "TIMER_UNIT"\n",
                diff_time(relative, max_time),
                (double)delta_time / (double)dbp_reader_nb_files(dbp));
    }

    for(ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        file = dbp_reader_get_file(dbp, ifd);

        sprintf(name, "MPI-%d", dbp_file_get_rank(file));
        sprintf(cont_mpi_name, "MPI-%d", dbp_file_get_rank(file));
        addContainer (0.00000, cont_mpi_name, "CT_P", "Appli", name, "");
        for(t = 0; t < dbp_file_nb_threads(file); t++) {
            th = dbp_file_get_thread(file, t);

            cont_thread_name = getThreadContainerIdentifier( cont_mpi_name, dbp_thread_get_hr_id(th) );
            {
                int l;
                l = 3 + snprintf(NULL, 0, "#  %s", cont_thread_name);
                if( l > stat_columns[0] )
                    stat_columns[0] = l;

                dico_stat[ifd][t].name = strdup(cont_thread_name);
                current_stat = dico_stat[ifd][t].stats;
            }
            addContainer (0.00000, cont_thread_name, "CT_T", cont_mpi_name, dbp_thread_get_hr_id(th), "");
        }
    }

    for(ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        file = dbp_reader_get_file(dbp, ifd);

        sprintf(name, "MPI-%d", dbp_file_get_rank(file));
        sprintf(cont_mpi_name, "MPI-%d", dbp_file_get_rank(file));
        for(t = 0; t < dbp_file_nb_threads(file); t++) {
            th = dbp_file_get_thread(file, t);
            cont_thread_name = getThreadContainerIdentifier( cont_mpi_name, dbp_thread_get_hr_id(th) );
            dump_one_paje(dbp, th, cont_mpi_name, cont_thread_name);
        }
    }

    return 0;
}

int main(int argc, char *argv[])
{
    dbp_multifile_reader_t *dbp;
    dbp_file_t *file;
    int i, j, k;

    dbp = dbp_reader_open_files(argc, argv);
    
    if( NULL == dbp )
        return 1;

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
    
    dague_profiling_dump_paje( "out", dbp );
    
    for(k = 0 ; k < dbp_reader_nb_dictionary_entries(dbp); k = k+1 ) {
        int l = strlen(dbp_dictionary_name(dbp_reader_get_dictionary(dbp, k)));
        stat_columns[k+1] = stat_columns[k] + max(l + 2, 16);
    }

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

    endTrace();

    return 0;
}
