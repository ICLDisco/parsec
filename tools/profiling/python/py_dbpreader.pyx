####################################
# DBPreader Python interface
# run 'python setup.py build_ext --inplace' to compile
# import py_dbpreader to use
#
# This is verified to work with Python 2.4.3 and above, compiled with Cython 0.16rc0
# However, it is recommended that Python 2.7 or greater is used to build and run.
#
# Be SURE to build this against the same version of Python as you have built Cython itself.
# Contrasting versions will likely lead to odd errors about Unicode functions.

import sys
from libc.stdlib cimport malloc, free
from parsec_profile import * # the pure Python classes

# this is the public Python interface function. call it.
cpdef readProfile(filenames):
    cdef char ** c_filenames = stringListToCStrings(filenames)
    cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)
    profile = multifile_reader(dbp_reader_nb_files(dbp), dbp_reader_nb_dictionary_entries(dbp))
    cdef dbp_file_t * cfile
    cdef dbp_dictionary_t * cdict

    profile.worldsize = dbp_reader_worldsize(dbp)

    # create dictionary first, for later use while making Events
    for key in range(profile.nb_dict_entries):
        cdict = dbp_reader_get_dictionary(dbp, key)
        entry = dbpDictEntry(key, dbp_dictionary_attributes(cdict))
        event_name = dbp_dictionary_name(cdict)
        profile.dictionary[event_name] = entry
        profile.dict_key_to_name[entry.key] = event_name
        profile.event_type_stats[event_name] = EventStats(event_name, key)
        
    # convert c to py
    for ifd in range(profile.nb_files):
        cfile = dbp_reader_get_file(dbp, ifd)
        pfile = dbpFile(profile, dbp_file_hr_id(cfile), dbp_file_get_name(cfile), dbp_file_get_rank(cfile))
        for index in range(dbp_file_nb_infos(cfile)):
            cinfo = dbp_file_get_info(cfile, index)
            key = dbp_info_get_key(cinfo)
            value = dbp_info_get_value(cinfo)
            pfile.infos.append(dbpInfo(key, value))
        for thread_num in range(dbp_file_nb_threads(cfile)):
            new_thr = makeDbpThread(profile, dbp, cfile, thread_num, pfile)
            pfile.threads.append(new_thr)
        profile.files.append(pfile)

    dbp_reader_close_files(dbp) # does nothing as of 2013-04-21
#   dbp_reader_dispose_reader(dbp)
    free(c_filenames)

    return profile

# helper function for readProfile
cdef char** stringListToCStrings(strings):
    cdef char ** c_argv
    bytes_strings = [bytes(x) for x in strings]
    c_argv = <char**>malloc(sizeof(char*) * len(bytes_strings)) 
    if c_argv is NULL:
        raise MemoryError()
    try:
        for idx, s in enumerate(bytes_strings):
            c_argv[idx] = s
    except:
        print("exception caught while converting to c strings")
        free(c_argv)
    return c_argv

# you can't call this. it will be called for you. call readProfile()
cdef makeDbpThread(profile, dbp_multifile_reader_t * dbp, dbp_file_t * cfile, int index, pfile):
    cdef dbp_thread_t * cthread = dbp_file_get_thread(cfile, index)
    cdef dbp_event_iterator_t * it_s = dbp_iterator_new_from_thread(cthread)
    cdef dbp_event_iterator_t * it_e = NULL
    cdef dbp_event_t * event_s = dbp_iterator_current(it_s)
    cdef dbp_event_t * event_e = NULL
    cdef dague_time_t reader_start = dbp_reader_min_date(dbp)
    cdef unsigned long long start = 0
    cdef unsigned long long end = 0
    cdef void * cinfo = NULL
    cdef papi_exec_info_t * cast_exec_info = NULL
    cdef select_info_t * cast_select_info = NULL

    thread = dbpThread(pfile, index)
    if thread.id + 1 > profile.thread_count:
        profile.thread_count = thread.id + 1

    while event_s != NULL:
        if KEY_IS_START( dbp_event_get_key(event_s) ):
            it_e = dbp_iterator_find_matching_event_all_threads(it_s)
            if it_e != NULL:
                event_e = dbp_iterator_current(it_e)
                if event_e != NULL:
                    start = diff_time(reader_start, dbp_event_get_timestamp(event_s))
                    end = diff_time(reader_start, dbp_event_get_timestamp(event_e))
                    event_key = int(dbp_event_get_key(event_s)) / 2 # to match dictionary
                    event_flags = dbp_event_get_flags(event_s)
                    event_handle_id = int(dbp_event_get_handle_id(event_s))
                    event_id = int(dbp_event_get_event_id(event_s))
                    event = dbpEvent(thread,
                                     event_key,
                                     event_flags,
                                     event_handle_id,
                                     event_id,
                                     start, end)
                    stats = profile.event_type_stats[profile.dict_key_to_name[event_key]]
                    stats.count += 1
                    stats.duration += event.duration
                    
                    while len(profile.handle_counts) < (event_handle_id + 1):
                        profile.handle_counts.extend([0])
                    profile.handle_counts[event_handle_id] += 1
                    #####################################
                    # not all events have info
                    # also, not all events have the same info.
                    # so this is where users must add code to translate
                    # their own info objects
                    cinfo = dbp_event_get_info(event_e)
                    if cinfo != NULL:
                        if (profile.dictionary.get('PINS_EXEC', None) and
                            event_key == profile.dictionary['PINS_EXEC'].key):
                            cast_exec_info = <papi_exec_info_t *>cinfo
                            kernel_name = str(cast_exec_info.kernel_name)
                            event.info = dbp_Exec_EventInfo(
                                cast_exec_info.kernel_type,
                                kernel_name,
                                cast_exec_info.vp_id,
                                cast_exec_info.th_id,
                                [cast_exec_info.values[x] for x
                                 in range(cast_exec_info.values_len)])
                            if not stats.papi_stats.get(kernel_name, None):
                                stats.papi_stats[kernel_name] = ExecSelectStats(kernel_name)
                            pstats = stats.papi_stats[kernel_name]
                            pstats.count += 1
                            pstats.duration += event.duration
                            pstats.l1_misses += event.info.values[0]
                            pstats.l2_hits += event.info.values[1]
                            pstats.l2_misses += event.info.values[2]
                            pstats.l2_accesses += event.info.values[3]
                        elif (profile.dictionary.get('PINS_SELECT', None) and
                              event_key == profile.dictionary['PINS_SELECT'].key):
                            cast_select_info = <select_info_t *>cinfo
                            kernel_name = str(cast_exec_info.kernel_name)
                            event.info = dbp_Select_EventInfo(
                                cast_select_info.kernel_type,
                                kernel_name,
                                cast_select_info.vp_id,
                                cast_select_info.th_id,
                                cast_select_info.victim_vp_id,
                                cast_select_info.victim_th_id,
                                cast_select_info.exec_context,
                                [cast_select_info.values[x] for x
                                 in range(cast_select_info.values_len)])
                            if not stats.papi_stats.get(kernel_name, None):
                                stats.papi_stats[kernel_name] = ExecSelectStats(kernel_name)
                            pstats = stats.papi_stats[kernel_name]
                            pstats.count += 1
                            pstats.duration += event.duration
                            pstats.l1_misses += event.info.values[0]
                            pstats.l2_hits += event.info.values[1]
                            pstats.l2_misses += event.info.values[2]
                            pstats.l2_accesses += event.info.values[3]
                        elif (profile.dictionary.get('PINS_SOCKET', None) and
                              event_key == profile.dictionary['PINS_SOCKET'].key):
                            cast_socket_info = <papi_socket_info_t *>cinfo
                            event.info = dbp_Socket_EventInfo(
                                cast_socket_info.vp_id,
                                cast_socket_info.th_id,
                                [cast_socket_info.values[x] for x
                                 in range(cast_socket_info.values_len)])
                            if not stats.papi_stats.get(SocketStats.class_name, None):
                                stats.papi_stats[SocketStats.class_name] = SocketStats()
                            pstats = stats.papi_stats[SocketStats.class_name]
                            pstats.count += 1
                            pstats.duration += event.duration
                            pstats.l3_exc_misses  += event.info.values[0]
                            pstats.l3_shr_misses += event.info.values[1]
                            pstats.l3_mod_misses += event.info.values[2]
                        # elif event.key == profile.dictionary['<SOME OTHER TYPE WITH INFO>'].key:
                            # event.info = <write a function and a Python type to translate>

                    thread.events.append(event)
                dbp_iterator_delete(it_e)
                it_e = NULL
        dbp_iterator_next(it_s)
        event_s = dbp_iterator_current(it_s)
    dbp_iterator_delete(it_s)
    it_s = NULL
    return thread
