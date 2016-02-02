/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
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

#include "dague/os-spec-timing.h"
#include "dague/profiling.h"
#include "dague/dague_binary_profile.h"
#include "dbpreader.h"

#ifdef DEBUG
#undef DEBUG
#endif

#if defined(DAGUE_DEBUG_MOTORMOUTH)
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

static void dump_one_xml(FILE *tracefile, const dbp_multifile_reader_t *dbp, const dbp_thread_t *th)
{
    int displayed_key, k;
    uint64_t start, end;
    dbp_event_iterator_t *it, *m;
    const dbp_event_t *e, *g;

    fprintf(tracefile,
            "            <THREAD>\n"
            "               <IDENTIFIER><![CDATA[%s]]></IDENTIFIER>\n", dbp_thread_get_hr_id(th) );

    for( k = 0; k < dbp_reader_nb_dictionary_entries(dbp); k++ ) {
        displayed_key = 0;

        it = dbp_iterator_new_from_thread( th );
        while( (e = dbp_iterator_current(it)) != NULL ) {
            if( KEY_IS_START( dbp_event_get_key(e) ) &&
                (BASE_KEY( dbp_event_get_key(e) ) == k) ) {
	      m = dbp_iterator_find_matching_event_all_threads(it, 0);
                if( NULL == m ) {
                    WARNING("   Event of class %s id %"PRIu32":%"PRIu64" at %lu does not have a match anywhere\n",
                             dbp_dictionary_name(dbp_reader_get_dictionary(dbp, BASE_KEY(dbp_event_get_key(e)))),
                             dbp_event_get_handle_id(e), dbp_event_get_event_id(e),
                             dbp_event_get_timestamp(e));
                } else {
                    g = dbp_iterator_current(m);

                    start = dbp_event_get_timestamp( e );
                    end = dbp_event_get_timestamp( g );

                    if( displayed_key == 0 ) {
                        fprintf(tracefile, "               <KEY ID=\"%d\">\n", k);
                        displayed_key = 1;
                    }

                    fprintf(tracefile,
                            "                  <EVENT>\n"
                            "                     <ID>%"PRIu32":%"PRIu64"</ID>\n"
                            "                     <START>%"PRIu64"</START>\n"
                            "                     <END>%"PRIu64"</END>\n",
                            dbp_event_get_handle_id(e), dbp_event_get_event_id( e ),
                            start, end);

                    if( dbp_event_get_flags( e ) & DAGUE_PROFILING_EVENT_HAS_INFO ) {
                        /** TODO fprintf(tracefile, "       <INFO><![CDATA[%s]]></INFO>\n", infostr); */
                    }
                    if( dbp_event_get_flags( g ) & DAGUE_PROFILING_EVENT_HAS_INFO ) {
                        /** TODO fprintf(tracefile, "       <INFO ATEND=\"true\"><![CDATA[%s]]></INFO>\n", infostr); */
                    }
                    fprintf(tracefile, "                  </EVENT>\n");

                    dbp_iterator_delete(m);
                }
            }
            dbp_iterator_next(it);
        }

        if( displayed_key ) {
            fprintf(tracefile, "              </KEY>\n");
        }

        dbp_iterator_delete(it);
    }
    fprintf(tracefile,
            "            </THREAD>\n");
}

static int dump_xml( const char* filename, const dbp_multifile_reader_t *dbp )
{
    int i, ifd, t;
    dbp_file_t *file;
    dbp_dictionary_t *dico;
    FILE* tracefile;

    tracefile = fopen(filename, "w");
    if( NULL == tracefile ) {
        return -1;
    }

    fprintf(tracefile,
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<PROFILING>\n"
            " <IDENTIFIER><![CDATA[%s]]></IDENTIFIER>\n"
            "  <INFOS>\n", dbp_file_hr_id(dbp_reader_get_file(dbp, 0)));
    for(ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        file = dbp_reader_get_file(dbp, ifd);
        for(i = 0; i < dbp_file_nb_infos(file); i++) {
    fprintf(tracefile, "    <INFO NAME=\"%s\"><![CDATA[%s]]></INFO>\n",
                    dbp_info_get_key(dbp_file_get_info(file, i)),
                    dbp_info_get_value(dbp_file_get_info(file, i)));
        }
    }

    fprintf(tracefile,
            "  </INFOS>\n"
            "  <DICTIONARY>\n");

    for(i = 0; i < dbp_reader_nb_dictionary_entries(dbp); i++) {
        dico = dbp_reader_get_dictionary(dbp, i);
        fprintf(tracefile,
                "   <KEY ID=\"%d\">\n"
                "    <NAME>%s</NAME>\n"
                "    <ATTRIBUTES><![CDATA[%s]]></ATTRIBUTES>\n"
                "   </KEY>\n",
                i,
                dbp_dictionary_name(dico),
                dbp_dictionary_attributes(dico));
    }
    fprintf(tracefile, " </DICTIONARY>\n");

    fprintf(tracefile, "   <DISTRIBUTED_PROFILE TIME_UNIT=\""TIMER_UNIT"\" WORLD_SIZE=\"%d\">\n",
            dbp_reader_worldsize(dbp));
    for(ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        file = dbp_reader_get_file(dbp, ifd);

        fprintf(tracefile,
                "      <NODE FILEID=\"%s\" RANK=\"%d\">\n",
                dbp_file_hr_id(file),
                dbp_file_get_rank(file));

        fprintf(tracefile, "         <PROFILES>\n");

        for(t = 0; t < dbp_file_nb_threads(file); t++) {
            dump_one_xml(tracefile, dbp, dbp_file_get_thread(file, t));
        }
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

int main(int argc, char *argv[])
{
    dbp_multifile_reader_t *dbp;

    dbp = dbp_reader_open_files(argc-1, argv+1);

    dump_xml( "out.xml", dbp);

    dbp_reader_close_files(dbp);
    free(dbp);
    dbp = NULL;

    return 0;
}
