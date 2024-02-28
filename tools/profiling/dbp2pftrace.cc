/*
 * Copyright (c) 2022      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <perfetto.h>
#include "parsec/profiling.h"
#include "parsec/parsec_binary_profile.h"
#include "dbpreader.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("executing")
        .SetDescription("executing tasks"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

static std::unique_ptr<perfetto::TracingSession> tracing_session;

void InitializePerfetto()
{
    perfetto::TracingInitArgs args;
    // The backends determine where trace events are recorded. For this example we
    // are going to use the in-process tracing service, which only includes in-app
    // events.
    args.backends = perfetto::kInProcessBackend;

    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
}

void StartTracing(const char *filename)
{
    // The trace config defines which types of data sources are enabled for
    // recording. In this example we just need the "track_event" data source,
    // which corresponds to the TRACE_EVENT trace points.
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(1024);
    auto *ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    cfg.set_write_into_file(true);
    cfg.set_output_path(filename);
    cfg.set_file_write_period_ms(1);

    tracing_session = perfetto::Tracing::NewTrace();
    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();
}

void FlushTracing()
{
    tracing_session->FlushBlocking();
}

void StopTracing()
{
    // Make sure the last event is closed for this example.
    tracing_session->FlushBlocking();

    // Stop tracing and read the trace data.
    tracing_session->StopBlocking();
}

static void convert_infos(const dbp_event_t *e, const dbp_dictionary_t *d, perfetto::EventContext &ctx)
{
    const char *conv = dbp_dictionary_convertor(d);
    const unsigned char *info = reinterpret_cast<const unsigned char *>(dbp_event_get_info(e));

    std::vector<char *> names;

    while( *conv != '\0' ) {
        int pos = 0;
        int fn_start = -1;
        int fn_end = -1;
        int ft_start = -1;
        int ft_end = -1;

        fn_start = pos;
        while( conv[pos] != '\0' && conv[pos] != '{' ) pos++;
        if(conv[pos] == '\0') goto syntax_error;
        fn_end = pos;
        if(fn_end - fn_start == 0) goto syntax_error;

        char *fn = new char[fn_end - fn_start+1];
        names.push_back(fn);
        memcpy(fn, &conv[fn_start], fn_end-fn_start);
        fn[fn_end-fn_start] = '\0';

        pos++;
        ft_start = pos;
        while( conv[pos] != '\0' && conv[pos] != '}' ) pos++;
        if(conv[pos] == '\0') goto syntax_error;
        ft_end = pos;
        if(ft_end - ft_start == 0) goto syntax_error;

        char ft[ft_end - ft_start+1];
        memcpy(ft, &conv[ft_start], ft_end-ft_start);
        ft[ft_end-ft_start] = '\0';

        pos++;
        if(conv[pos] == ';') pos++;

        if(strcmp(ft, "int32_t") == 0) {
            int32_t v = *(int32_t*)info;
            info += sizeof(int32_t);
            ctx.AddDebugAnnotation(fn, v);
        } else if(strcmp(ft, "uint32_t") == 0) {
            int32_t v = *(uint32_t*)info;
            info += sizeof(uint32_t);
            ctx.AddDebugAnnotation(fn, v);
        } else if(strcmp(ft, "int64_t") == 0) {
            int64_t v = *(int64_t*)info;
            info += sizeof(int64_t);
            ctx.AddDebugAnnotation(fn, v);
        } else if(strcmp(ft, "uint64_t") == 0) {
            int64_t v = *(uint64_t*)info;
            info += sizeof(uint64_t);
            ctx.AddDebugAnnotation(fn, v);
        } else {
            std::cerr << "Ignored: value of type " << ft << " for field " << fn << " in event of type " << dbp_dictionary_name(d) << std::endl; 
            goto syntax_error;
        }
        conv += pos;
    }

    for(auto n = names.begin(); n != names.end(); n++) {
        delete[] *n;
    }

    return;

  syntax_error:
    std::cerr << "Malformed convertor ends in " << conv << " for event of type " <<  dbp_dictionary_name(d) << std::endl; 
}

static void convert_one_thread(const dbp_thread_t *th, int thid, const dbp_file_t *file)
{
    std::stringstream ss;
    static int thread_uuid = 1;
    ss << " [rank " << dbp_file_get_rank(file) << " thread " << thid << "] (" << dbp_file_hr_id(file) << ", " << dbp_thread_get_hr_id(th) << ")";
    perfetto::ThreadTrack t_track = perfetto::ThreadTrack::ForThread(thread_uuid++);
    perfetto::protos::gen::TrackDescriptor desc = t_track.Serialize();
    desc.mutable_thread()->set_thread_name(ss.str());
    perfetto::TrackEvent::SetTrackDescriptor(t_track, desc);
    std::size_t nb_traced = 0;

    TRACE_EVENT_BEGIN("executing", nullptr, [&](perfetto::EventContext ctx) {
        ctx.event()->set_timestamp_absolute_us(1);
        ctx.event()->set_name(ss.str());
        ctx.event()->set_track_uuid(t_track.uuid);
    });
    TRACE_EVENT_END("executing", [&](perfetto::EventContext ctx) {
        ctx.event()->set_timestamp_absolute_us(2);
        ctx.event()->set_track_uuid(t_track.uuid);
    });

    dbp_event_iterator_t *it = dbp_iterator_new_from_thread( th );
    dbp_event_iterator_t *m;
    const dbp_event_t *e, *g;
    while( (e = dbp_iterator_current(it)) != NULL ) {
        if( KEY_IS_START(dbp_event_get_key(e)) ) {
            m = dbp_iterator_find_matching_event_all_threads(it);
            if(NULL == m ) {
                std::cerr << "Event of class " << dbp_dictionary_name(dbp_file_get_dictionary(file, BASE_KEY(dbp_event_get_key(e))))
                        << " id " <<  dbp_event_get_taskpool_id(e) << ":" << dbp_event_get_event_id(e) << " at " << dbp_event_get_timestamp(e)
                        << " does not have a match anywhere" << std::endl;
            } else {
                const dbp_dictionary_t *d;
                g = dbp_iterator_current(m);

                auto start = dbp_event_get_timestamp( e );
                auto end = dbp_event_get_timestamp( g );

                d = dbp_file_get_dictionary(file, BASE_KEY(dbp_event_get_key(e)));
                std::stringstream ename(dbp_dictionary_name(d));

                TRACE_EVENT_BEGIN("executing", nullptr, [&](perfetto::EventContext ctx) {
                    ctx.event()->set_timestamp_absolute_us(start);
                    ctx.event()->set_name(ename.str());
                    ctx.event()->set_track_uuid(t_track.uuid);
                    if( dbp_event_get_flags(e) & PARSEC_PROFILING_EVENT_HAS_INFO ) {
                        convert_infos(e, d, ctx);
                    }
                });
                TRACE_EVENT_END("executing", [&](perfetto::EventContext ctx) {
                    ctx.event()->set_timestamp_absolute_us(end);
                    ctx.event()->set_track_uuid(t_track.uuid);
                    if( dbp_event_get_flags(g) & PARSEC_PROFILING_EVENT_HAS_INFO ) {
                        convert_infos(g, d, ctx);
                    }
                });
                nb_traced += 2;
                if(nb_traced > 1024) {
                    FlushTracing();
                    nb_traced = 0;
                }
            }
            dbp_iterator_delete(m);
        }
        dbp_iterator_next(it);
    }
    dbp_iterator_delete(it);
}

static void convert_one_file(const dbp_file_t *file)
{
    std::stringstream ss;
    ss << dbp_file_hr_id(file) << " (" << dbp_file_get_rank(file) << ")";
    // Give a custom name for the traced process.
    perfetto::ProcessTrack p_track = perfetto::ProcessTrack::Current();
    auto desc = perfetto::ProcessTrack::Current().Serialize();
    desc.mutable_process()->set_process_name(ss.str());
    perfetto::TrackEvent::SetTrackDescriptor(p_track, desc);

    for (int t = 0; t < dbp_file_nb_threads(file); t++) {
        const dbp_thread_t *th = dbp_file_get_thread(file, t);
        convert_one_thread(th, t, file);
    }
}

int main(int argc, char *argv[])
{
    int c;
    char *output = nullptr;
    std::vector<char *>inputs;

    while (1) {
        int option_index = 0;
        static struct option long_options[] = {
            {"output",  required_argument, 0,  'o' },
            {"input",   required_argument, 0,  'i' },
            {"help",          no_argument, 0,  'h' },
            {0,         0,                 0,  0 }
        };

        c = getopt_long(argc, argv, "o:i:h", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
        case 'o':
            if(nullptr != output)
                free(output);
            output = strdup(optarg);
            break;
        case 'i':
            inputs.push_back(strdup(optarg));
            break;

        default:
            std::cerr << "Unknown option: '" << c << "'" << std::endl;
            /* fall through */
        case '?':
            /* fall through */
        case 'h':
            std::cerr << "Converts a PaRSEC Binary Profile Format file (.prof file) into a Perfetto Binary Trace (pftrace file)" << std::endl
                      << "Usage:" << std::endl
                      << argv[0] << " [-i input*] -o output [additional inputs] or -h" << std::endl;
            break;
        }
    }
    while (optind < argc) 
        inputs.push_back(strdup(argv[optind++]));

    if( nullptr == output ) {
        std::cerr << "Output file not specified. Call with -o output file" << std::endl;
        return EXIT_FAILURE;
    }

    if( 0 == inputs.size() ) {
        std::cerr << "No input specified. At least one input file must be specified" << std::endl;
        return EXIT_FAILURE;
    }

    dbp_multifile_reader_t *dbp;
    dbp = dbp_reader_open_files(inputs.size(), inputs.data());
    if( NULL == dbp ) {
        std::cerr << "Unable to read some of the files given as input -- bailing out" << std::endl;
        return EXIT_FAILURE;
    }
    for(auto f = inputs.begin(); f != inputs.end(); f++) {
        free(*f);
    }
    inputs.clear();

    InitializePerfetto();

    struct stat st;
    if( stat(output, &st) == 0 ) {
        std::cerr << "Overwriting " << output << std::endl;
        if(unlink(output) != 0) {
            std::cerr << "Unable to delete " << output << " : " << strerror(errno) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    StartTracing(output);

    for(int ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        auto file = dbp_reader_get_file(dbp, ifd);
        convert_one_file(file);
    }
    dbp_reader_close_files(dbp);

    StopTracing();

    return EXIT_SUCCESS;
}
