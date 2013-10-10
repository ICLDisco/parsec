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
from operator import attrgetter
from libc.stdlib cimport malloc, free
from profiling      import * # the pure Python classes
from profiling_info import * # the pure Python classes representing custom INFO structs

# this is the public Python interface function. call it.
cpdef readProfile(filenames, do_sort=True, sort_key='begin'):
    cdef char ** c_filenames = stringListToCStrings(filenames)
    cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)
    
    nb_files = dbp_reader_nb_files(dbp)
    nb_dict_entries = dbp_reader_nb_dictionary_entries(dbp)
    cdef dbp_file_t * cfile
    cdef dbp_dictionary_t * cdict

    profile = Profile()
    profile.worldsize = dbp_reader_worldsize(dbp) # what does this even do?

    # create dictionary first, for later use while making Events
    for key in range(nb_dict_entries):
        cdict = dbp_reader_get_dictionary(dbp, key)
        event_name = dbp_dictionary_name(cdict)
        profile.type_key_to_name[key] = event_name
        profile.event_types[event_name] = dbpEventType(profile, key,
                                                       dbp_dictionary_attributes(cdict))
    # convert c to py
    for ifd in range(nb_files):
        cfile = dbp_reader_get_file(dbp, ifd)
        rank = dbp_file_get_rank(cfile)
        pfile = dbpFile(profile, dbp_file_hr_id(cfile),
                        dbp_file_get_name(cfile), rank)
        for index in range(dbp_file_nb_infos(cfile)):
            cinfo = dbp_file_get_info(cfile, index)
            key = dbp_info_get_key(cinfo)
            value = dbp_info_get_value(cinfo)
            pfile.infos.append(dbpInfo(key, value))
        for thread_id in range(dbp_file_nb_threads(cfile)):
            new_thr = makeDbpThread(profile, dbp, cfile, thread_id, pfile)
            pfile.threads.append(new_thr)
        profile.files[rank] = pfile

    if do_sort:
        profile.sort(key = attrgetter(sort_key))
        make_duration_stats(profile)
        for key, handle in profile.handles.iteritems():
            handle.sort(key = attrgetter(sort_key))
            make_duration_stats(handle)
        for event_name, event_type in profile.event_types.iteritems():
            event_type.sort(key = attrgetter(sort_key))
            make_duration_stats(event_type)
            
        for rank, f in profile.files.iteritems():
            for thread in f.threads:
                thread.sort(key = attrgetter(sort_key))
                for name, event_type in thread.event_types.iteritems():
                    event_type.sort(key = attrgetter(sort_key))
                    for index, event in enumerate(event_type):
                        if index > 0:
                            if event_type[index - 1].end > event.begin:
                                print('these {} events overlap!'.format(name))
                                print(event_type[index - 1])
                                print(event)
                    make_duration_stats(event_type)
    profile.is_sorted = do_sort
        
    dbp_reader_close_files(dbp) # does nothing as of 2013-04-21
#   dbp_reader_dispose_reader(dbp)
    free(c_filenames)

    # print('start event ids')
    # total = 0
    # d = dict()
    # names = dict()
    # for key, value in profile.begin_ids.iteritems():
    #     total += 1
    #     if len(value) not in d:
    #         d[len(value)] = 0
    #         names[len(value)] = []
    #     d[len(value)] += 1
    #     for event in value:
    #         if profile.type_key_to_name[event.key] not in names[len(value)]:
    #             names[len(value)].append(profile.type_key_to_name[event.key])
    # print(d)
    # print(names)
    # print('end event ids')

    for key, value in profile.errors.iteritems():
        print('event ' + str(key) + ' ' + str(len(value)))
    return profile

cpdef make_duration_stats(container):
    if len(container) == 0:
        container.begin = 0
        container.end = 0
        container.duration = 0
    else:
        container.begin = container[0].begin
        container.end = container[-1].end
        container.duration = container.end - container.begin
    
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

class dbpEvent:
   def __init__(self, parentThread, key, flags, object_id, event_id, start, end):
      self.thread = parentThread
      self.key = key
      self.flags = flags
      self.object_id = object_id
      self.event_id = event_id
      self.start = start
      self.end = end
      self.duration = self.end - self.start
   def __str__(self):
      return 'key %d flags %d tid %d objID %s eventID %d start %d end %d duration %d' % (
              self.key, self.flags, self.thread.id, self.object_id, self.event_id,
              self.start, self.end, self.duration)


    thread = dbpThread(pfile, index)

    while event_s != NULL:
        if KEY_IS_START( dbp_event_get_key(event_s) ):
            it_e = dbp_iterator_find_matching_event_all_threads(it_s, 0)
            if it_e != NULL:
                event_e = dbp_iterator_current(it_e)
                if event_e != NULL:
                    begin = diff_time(reader_begin, dbp_event_get_timestamp(event_s))
                    end = diff_time(reader_begin, dbp_event_get_timestamp(event_e))

                    if thread.begin > begin:
                        thread.begin = begin
                    if thread.end < end:
                        thread.end = end
                    
                    event_key = int(dbp_event_get_key(event_s)) / 2 # to match dictionary
                    event_flags = dbp_event_get_flags(event_s)
                    event_handle_id = int(dbp_event_get_handle_id(event_s))
                    event_id = int(dbp_event_get_event_id(event_s))
                    event = dbpEvent(thread,
                                     event_key,
                                     event_flags,
                                     event_handle_id,
                                     event_id,
                                     begin, end)
                    event_name = profile.type_key_to_name[event_key]
                    
                    if event_handle_id not in profile.handles:
                        profile.handles[event_handle_id] = dbpHandle(profile, event_handle_id)
                    if event_name not in thread.event_types:
                        thread.event_types[event_name] = dbpEventType(
                            profile, event_key, profile.event_types[event_name].attributes)
                    if event_name not in profile.handles[event_handle_id].event_types:
                        profile.handles[event_handle_id].event_types[event_name] = dbpEventType(
                            profile, event_key, profile.event_types[event_name].attributes)

                    global_stats = profile.event_types[event_name].stats
                    global_stats.count += 1
                    global_stats.total_duration += event.duration
                        
                    # debug tests for PINS_L123 split across threads
                    # if dbp_iterator_thread(it_s) != dbp_iterator_thread(it_e):
                    #     print('this event {} started {} and ended {} in different threads.'.format(
                    #         event_name, begin, end))
                    # if event_id not in profile.begin_ids:
                    #     profile.begin_ids[event_id] = []
                    # profile.begin_ids[event_id].append(event)
                    # end_id = dbp_event_get_event_id(event_e)
                    # if end_id not in profile.end_ids:
                    #     profile.end_ids[end_id] = []
                    # profile.end_ids[end_id].append(event)

                    #####################################
                    # not all events have info
                    # also, not all events have the same info.
                    # so this is where users must add code to translate
                    # their own info objects
                    cinfo = dbp_event_get_info(event_e)
                    if cinfo != NULL:
                        if ('PINS_EXEC' in profile.event_types and
                            event_key == profile.event_types['PINS_EXEC'].key):
                            cast_exec_info = <papi_exec_info_t *>cinfo
                            kernel_name = str(cast_exec_info.kernel_name)
                            event.info = dbp_Exec_EventInfo(
                                cast_exec_info.kernel_type,
                                kernel_name,
                                cast_exec_info.vp_id,
                                cast_exec_info.th_id,
                                [cast_exec_info.values[x] for x
                                 in range(cast_exec_info.values_len)])
                            if kernel_name not in global_stats.exec_stats:
                                global_stats.exec_stats[kernel_name] = ExecSelectStats(kernel_name)
                            pstats = global_stats.exec_stats[kernel_name]
                            pstats.count += 1
                            pstats.total_duration += event.duration
                            pstats.l1_misses += event.info.values[0]
                            pstats.l2_hits += event.info.values[1]
                            pstats.l2_misses += event.info.values[2]
                            pstats.l2_accesses += event.info.values[3]
                            # if kernel_name == 'POTRF':
                            #     print('PYDBPR found POTRF ' + str(pstats.count) + ' ' + str(event_key))
                        elif ('PINS_SELECT' in profile.event_types and
                              event_key == profile.event_types['PINS_SELECT'].key):
                            cast_select_info = <select_info_t *>cinfo
                            kernel_name = str(cast_select_info.kernel_name)
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
                            if kernel_name not in global_stats.select_stats:
                                global_stats.select_stats[kernel_name] = ExecSelectStats(kernel_name)
                            pstats = global_stats.select_stats[kernel_name]
                            pstats.count += 1
                            pstats.total_duration += event.duration
                            pstats.l2_misses += event.info.values[1]
                        elif ('PINS_SOCKET' in profile.event_types and
                              event_key == profile.event_types['PINS_SOCKET'].key):
                            cast_socket_info = <papi_socket_info_t *>cinfo
                            event.info = dbp_Socket_EventInfo(
                                cast_socket_info.vp_id,
                                cast_socket_info.th_id,
                                [cast_socket_info.values[x] for x
                                 in range(cast_socket_info.values_len)])
                            if SocketStats.class_name not in global_stats.socket_stats:
                                global_stats.socket_stats[SocketStats.class_name] = SocketStats()
                            pstats = global_stats.socket_stats[SocketStats.class_name]
                            pstats.count += 1
                            pstats.total_duration += event.duration
                            pstats.l3_exc_misses  += event.info.values[0]
                            pstats.l3_shr_misses += event.info.values[1]
                            pstats.l3_mod_misses += event.info.values[2]
                        elif ('PINS_L12_EXEC' in profile.event_types and
                              event_key == profile.event_types['PINS_L12_EXEC'].key):
                            cast_L12_exec_info = <papi_L12_exec_info_t *>cinfo
                            kernel_name = str(cast_L12_exec_info.kernel_name)
                            event.info = dbp_Exec_EventInfo(
                                cast_L12_exec_info.kernel_type,
                                kernel_name,
                                cast_L12_exec_info.vp_id,
                                cast_L12_exec_info.th_id,
                                [cast_L12_exec_info.L1_misses,
                                 cast_L12_exec_info.L2_misses])
                            if kernel_name not in global_stats.exec_stats:
                                global_stats.exec_stats[kernel_name] = ExecSelectStats(kernel_name)
                            pstats = global_stats.exec_stats[kernel_name]
                            pstats.count += 1
                            pstats.total_duration += event.duration
                            pstats.l1_misses += event.info.values[0]
                            pstats.l2_misses += event.info.values[1]
                        elif ('PINS_L12_SELECT' in profile.event_types and
                              event_key == profile.event_types['PINS_L12_SELECT'].key):
                            cast_L12_select_info = <papi_L12_select_info_t *>cinfo
                            kernel_name = str(cast_L12_select_info.kernel_name)
                            event.info = dbp_Select_EventInfo(
                                cast_L12_select_info.kernel_type,
                                kernel_name,
                                cast_L12_select_info.vp_id,
                                cast_L12_select_info.th_id,
                                cast_L12_select_info.victim_vp_id,
                                cast_L12_select_info.victim_th_id,
                                cast_L12_select_info.starvation,
                                cast_L12_select_info.exec_context,
                                [cast_L12_select_info.L1_misses,
                                 cast_L12_select_info.L2_misses])
                            if kernel_name not in global_stats.select_stats:
                                global_stats.select_stats[kernel_name] = ExecSelectStats(kernel_name)
                                global_stats.select_stats[kernel_name].starvation = 0
                            pstats = global_stats.select_stats[kernel_name]
                            pstats.count += 1
                            pstats.total_duration += event.duration
                            pstats.starvation += event.info.starvation
                            pstats.l1_misses += event.info.values[0]
                            pstats.l2_misses += event.info.values[1]
                        elif ('PINS_L123' in profile.event_types and
                              event_key == profile.event_types['PINS_L123'].key):
                            cast_L123_info = <papi_L123_info_t *>cinfo
                            event.info = dbp_Socket_EventInfo(
                                cast_L123_info.vp_id,
                                cast_L123_info.th_id,
                                [cast_L123_info.L1_misses,
                                 cast_L123_info.L2_misses,
                                 cast_L123_info.L3_misses])
                            if SocketStats.class_name not in global_stats.socket_stats:
                                global_stats.socket_stats[SocketStats.class_name] = SocketStats()
                            pstats = global_stats.socket_stats[SocketStats.class_name]
                            pstats.count += 1
                            pstats.total_duration += event.duration
                            pstats.l1_misses += event.info.values[0]
                            pstats.l2_misses += event.info.values[1]
                            pstats.l3_exc_misses  += event.info.values[2]
                            thread.l3_misses = cast_L123_info.L3_misses
                        elif ('PINS_L12_ADD' in profile.event_types and
                              event_key == profile.event_types['PINS_L12_ADD'].key):
                            cast_L12_exec_info = <papi_L12_exec_info_t *>cinfo
                            kernel_name = str(cast_L12_exec_info.kernel_name)
                            event.info = dbp_Exec_EventInfo(
                                cast_L12_exec_info.kernel_type,
                                kernel_name,
                                cast_L12_exec_info.vp_id,
                                cast_L12_exec_info.th_id,
                                [cast_L12_exec_info.L1_misses,
                                 cast_L12_exec_info.L2_misses])
                            if kernel_name not in global_stats.add_stats:
                                global_stats.add_stats[kernel_name] = ExecSelectStats(kernel_name)
                            pstats = global_stats.add_stats[kernel_name]
                            pstats.count += 1
                            pstats.total_duration += event.duration
                            pstats.l1_misses += event.info.values[0]
                            pstats.l2_misses += event.info.values[1]
                        # elif ('<EVENT_TYPE_NAME>' in profile.event_types and
                        #       event.key == profile.event_types['<EVENT_TYPE_NAME>'].key):
                        #   event.info = <write a function and a Python type to translate>
                        else:
                            dont_print = True
                            if not dont_print:
                                print('missed an info for event key ' + event_name )

                    # event constructed. add it everywhere it belongs.
                    thread.event_types[event_name].append(event)
                    thread.append(event)
                    profile.handles[event_handle_id].event_types[event_name].append(event)
                    profile.handles[event_handle_id].append(event)
                    profile.event_types[event_name].append(event)
                    profile.append(event)
                    
                dbp_iterator_delete(it_e)
                it_e = NULL
            else:
                if event_name not in profile.errors:
                    profile.errors[event_name] = []
                profile.errors[event_name].append('event of class {} id {} at {} does not have a match.\n'.format(
                    event_name, event_id, thread.id))
        dbp_iterator_next(it_s)
        event_s = dbp_iterator_current(it_s)
        
    dbp_iterator_delete(it_s)
    it_s = NULL

    thread.duration = thread.end - thread.begin
    if thread.starvation > thread.duration:
        print('something is fishy... {} {}'.format(thread.starvation, thread.duration))
    for event_name, event_type in thread.event_types.iteritems():
        event_type.stats.starvation = 1.0 - float(event_type.stats.total_duration) / thread.duration
        
    return thread
