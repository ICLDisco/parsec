/*
 * Copyright (c)      2012 The University of Tennessee and The University
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

#include "profiling.h"
#include "dbp.h"
#include "dbpreader.h"
#include "graph.h"
#include "animation.h"

#include <glob.h>

#define NBFRAMES 200

static void usage(const char *prg)
{
    fprintf(stderr, 
            "Usage: %s <basename>\n"
            "  Opens <basename>.[0-9]*.profile to serve as binary profile traces of a DAGuE execution\n"
            "    and <basename>-[0-9]+.dot to serve as a DAG representation of the same execution\n"
            "  and creates <basename>-[0-9]+.png files that represent on the same figure the\n"
            "    evolution of the execution in a Gantt chart and in the DAG\n",
            prg);
    exit(1);
}

static void sample(const dbp_multifile_reader_t *dbp,
                   dbp_event_iterator_t **allthreads, 
                   int nbthreads, 
                   dague_time_t sframe, 
                   float top)
{
    int t;
    const dbp_event_t *dbp_e;
    dbp_dictionary_t *dico;
    char *tname;
    unsigned long long oid;
    unsigned int node;
    int key;
    char *g;
    unsigned int len;

    printf("Sampling from current position up to for %f\n", top);

    for(t = 0; t < nbthreads; t++) {
        dbp_e = dbp_iterator_current( allthreads[t] );
        while( dbp_e && diff_time(sframe, dbp_event_get_timestamp(dbp_e)) < top ) {

            key = dbp_event_get_key( dbp_e );

            dico = dbp_reader_get_dictionary(dbp, BASE_KEY(key));
            tname = dbp_dictionary_name(dico);
            oid = dbp_event_get_event_id(dbp_e);
            node = find_node_by_task_name_and_handle_id(tname, oid);
            
            if( NID != node ) {
                if( KEY_IS_START(key) ) {
                    clear_node_status(node, STATUS_READY | STATUS_ENABLED);
                    set_node_status(node, STATUS_RUNNING);
                } else {
                    assert( KEY_IS_END(key) );
                    clear_node_status(node, STATUS_RUNNING);
                    set_node_status(node, STATUS_DONE);
                    update_neighbors_status(node);
                }
            }

            dbp_e = dbp_iterator_next( allthreads[t] );
        }
    }

    persistentGraphRender(&g, &len);
    addAnimation(g, len, 10);
    free(g);
}

int main(int argc, char *argv[])
{
    dbp_multifile_reader_t *dbp;
    int i, n, e;
    glob_t profiles;
    char *profiles_pattern;
    glob_t dots;
    char *dots_pattern;
    dbp_event_iterator_t **allthreads;
    dbp_event_iterator_t *dbp_i;
    int nbthreads = 0;
    int f, t;
    dbp_file_t *dbp_f;
    dague_time_t mintime = ZERO_TIME, maxtime = ZERO_TIME;
    dbp_thread_t *dbp_t;
    const dbp_event_t *dbp_e;
    float delta;
    char **traced_types;
    int    nb_traced_types;
 
    if( argc != 2 ) {
        fprintf(stderr, "Not the right number of arguments\n");
        usage(argv[0]);
    }
    
    asprintf(&profiles_pattern, "%s.[0-9]*.profile", argv[1]);
    if( glob( profiles_pattern, 0, NULL, &profiles ) != 0 ) {
        fprintf(stderr, "Could not find any %s files\n", profiles_pattern);
        usage(argv[0]);
    }
    asprintf(&dots_pattern, "%s-[0-9]*.dot", argv[1]);
    if( glob( dots_pattern, 0, NULL, &dots ) != 0 ) {
        fprintf(stderr, "Could not find any %s files\n", dots_pattern);
        usage(argv[0]);
    }

    if( dots.gl_pathc != profiles.gl_pathc ||
        dots.gl_pathc == 0 ) {
        fprintf(stderr, 
                "There is %lu files corresponding to the pattern %s, and %lu files corresponding to the pattern %s\n"
                "Cannot handle non-corresponding / empty cases.\n",
                dots.gl_pathc, dots_pattern,
                profiles.gl_pathc, profiles_pattern);
        usage(argv[0]);
    }

    dbp = dbp_reader_open_files(profiles.gl_pathc, profiles.gl_pathv);
    if( NULL == dbp ) {
        fprintf(stderr, "Unable to read all %lu profile files\n",
                profiles.gl_pathc);
        usage(argv[0]);
    }

    fprintf(stderr, "#Read %lu profile files\n", profiles.gl_pathc);

    nb_traced_types = dbp_reader_nb_dictionary_entries( dbp );
    traced_types = (char **)malloc(nb_traced_types * sizeof(char*));
    for(i = 0; i < nb_traced_types; i++) {
        traced_types[i] = strdup( dbp_dictionary_name( dbp_reader_get_dictionary(dbp, i) ) );
    }

    add_key_nodes();

    n = 0;
    for(i = 0; i < (int)dots.gl_pathc; i++)
        n += add_nodes_from_dotfile( dots.gl_pathv[i], i, traced_types, nb_traced_types );
    e = 0;
    for(i = 0; i < (int)dots.gl_pathc; i++) 
        e += add_edges_from_dotfile( dots.gl_pathv[i] );

    fprintf(stderr, "#Read %lu dots files: created a graph of %d nodes, and %d edges\n", 
            dots.gl_pathc, n, e);

    nbthreads = 0;
    for(f = 0; f < dbp_reader_nb_files(dbp); f++) {
        dbp_f = dbp_reader_get_file(dbp, f);
        if( f == 0 ) {
            mintime = dbp_file_get_min_date(dbp_f);
        } else {
            if( time_less(dbp_file_get_min_date(dbp_f), mintime) )
                mintime = dbp_file_get_min_date(dbp_f);
        }
        for(t = 0; t < dbp_file_nb_threads(dbp_f); t++) {
            nbthreads++;
        }
    }

    allthreads = (dbp_event_iterator_t **)malloc( sizeof(dbp_event_iterator_t *) * nbthreads );

    nbthreads = 0;
    maxtime = mintime;
    for(f = 0; f < dbp_reader_nb_files(dbp); f++) {
        dbp_f = dbp_reader_get_file(dbp, f);
        for(t = 0; t < dbp_file_nb_threads(dbp_f); t++) {
            dbp_t = dbp_file_get_thread(dbp_f, t);

            dbp_i = dbp_iterator_new_from_thread(dbp_t);
            while( (dbp_e = dbp_iterator_current(dbp_i)) != NULL ) {
                if( time_less( maxtime, dbp_event_get_timestamp(dbp_e) ) )
                    maxtime = dbp_event_get_timestamp(dbp_e);
                dbp_iterator_next(dbp_i);
            }
            dbp_iterator_delete(dbp_i);
            
            allthreads[nbthreads] = dbp_iterator_new_from_thread(dbp_t);
            dbp_iterator_first(allthreads[nbthreads]);
            nbthreads++;
        }
    }

    fprintf(stderr, "#Sampling %d frames in an execution of %llu %s\n",
            NBFRAMES, diff_time(mintime, maxtime), TIMER_UNIT);
    delta = (float)diff_time(mintime, maxtime) / (float)NBFRAMES;

    graphInit();

    persistentGraphLayoutEntireGraph();

    {
        char *filename;
        char *r;
        unsigned int length;

        persistentGraphRender(&r, &length);
        asprintf(&filename, "%s.gif", argv[1]);
        startAnimation(filename, r, length);
        free(r);
        free(filename);
    }

    for(f = 0; f < NBFRAMES; f++) {
        sample(dbp, allthreads, nbthreads, mintime, (f+1)*delta);
    }
    persistentGraphClose();
    endAnimation();

    return graphFini();
}
