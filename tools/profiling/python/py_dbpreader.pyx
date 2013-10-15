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

# cython: profile=False
# but could be True if we wanted to use cProfile!
import sys
from operator import attrgetter
from libc.stdlib cimport malloc, free
from profiling      import * # the pure Python classes
# from profiling_info import * # the pure Python classes representing custom INFO structs
import numpy as np
import pandas as pd
#import cProfile, pstats

# this is the public Python interfacea function. call it.
cpdef readProfile(filenames, skip_infos=False):
    cdef char ** c_filenames = stringListToCStrings(filenames)
    cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)
    
    nb_files = dbp_reader_nb_files(dbp)
    nb_dict_entries = dbp_reader_nb_dictionary_entries(dbp)
    cdef dbp_file_t * cfile
    cdef dbp_dictionary_t * cdict

    profile = Profile()
    profile.worldsize = dbp_reader_worldsize(dbp) # what does this even do?

    profile.df = None
    profile.series = []
    profile.infos = []
    profile.event_columns = ['filerank', 'thread', 'key', 'flags', 'handle_id',
                             'id', 'begin', 'end', 'duration', 'unique_id']

    # create dictionary first, for later use while making Events
    for key in range(nb_dict_entries):
        cdict = dbp_reader_get_dictionary(dbp, key)
        event_name = dbp_dictionary_name(cdict)
        profile.type_key_to_name[key] = event_name
        profile.event_types[event_name] = dbpEventType(profile, key,
                                                       dbp_dictionary_attributes(cdict))
    # convert c to py
    # pr = cProfile.Profile()
    # pr.enable()
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
        print('First, we must iterate through the threads.')
        with Timer() as t:
            for thread_id in range(dbp_file_nb_threads(cfile)):
                thread = makeDbpThread(profile, dbp, cfile, thread_id, pfile, skip_infos)
                pfile.threads.append(thread)
        print('This takes ' + str(t.interval) + ' seconds, which is about')
        print(str(t.interval/len(pfile.threads)) + ' seconds per thread.')
        profile.files[rank] = pfile
    # pr.disable()
    # ps = pstats.Stats(pr, stream=sys.stdout)
    # ps.print_stats()

    print('Then we construct the DataFrames....')
    with Timer() as t:
        profile.df = pd.DataFrame(profile.series, columns=profile.event_columns)
#        profile.df = pd.DataFrame.from_records(profile.series)
    print('   main dataframe time: ' + str(t.interval))
    with Timer() as t:
        infos = pd.DataFrame.from_records(profile.infos)
    print('   infos dataframe time: ' + str(t.interval))
    with Timer() as t:
        profile.df = profile.df.merge(infos, on='unique_id', how='outer')
    print('   join/merge time: ' + str(t.interval))
    
    profile.series = None

    # with Timer() as t:
    #     profile.info_df = pd.DataFrame.from_records(profile.infos)
    # print('infos dataframe time: ' + str(t.interval))
    profile.infos = None
    
    dbp_reader_close_files(dbp) # does nothing as of 2013-04-21
#   dbp_reader_dispose_reader(dbp)
    free(c_filenames)

    for key, value in profile.errors.iteritems():
        print('event ' + str(key) + ' ' + str(len(value)))
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
cdef makeDbpThread(profile, dbp_multifile_reader_t * dbp, dbp_file_t * cfile, int index, pfile, skip_infos):
    cdef dbp_thread_t * cthread = dbp_file_get_thread(cfile, index)
    cdef dbp_event_iterator_t * it_s = dbp_iterator_new_from_thread(cthread)
    cdef dbp_event_iterator_t * it_e = NULL
    cdef const dbp_event_t * event_s = dbp_iterator_current(it_s)
    cdef const dbp_event_t * event_e = NULL
    cdef dague_time_t reader_begin = dbp_reader_min_date(dbp)
    cdef uint64_t begin = 0
    cdef uint64_t end = 0
    cdef void * cinfo = NULL
    cdef papi_exec_info_t * cast_exec_info = NULL
    cdef select_info_t * cast_select_info = NULL
    cdef papi_L123_info_t * cast_L123_info = NULL
    cdef papi_L12_select_info_t * cast_L12_select_info = NULL
    cdef papi_L12_exec_info_t * cast_L12_exec_info = NULL
    cdef long long int * cast_lld_ptr = NULL
    cdef bytes kernel_name

    thread_id = index
    thread = dbpThread(pfile, index)

    while event_s != NULL:
        if KEY_IS_START( dbp_event_get_key(event_s) ):
            it_e = dbp_iterator_find_matching_event_all_threads(it_s, 0)
            if it_e != NULL:
                event_e = dbp_iterator_current(it_e)
                if event_e != NULL:
                    begin = diff_time(reader_begin, dbp_event_get_timestamp(event_s))
                    end = diff_time(reader_begin, dbp_event_get_timestamp(event_e))

                    event_key = int(dbp_event_get_key(event_s)) / 2 # to match dictionary

                    # if end < begin and event_key == profile.event_types['PINS_L12_SELECT'].key:
                    #     dbp_iterator_delete(it_e)
                    #     it_e = NULL
                    #     dbp_iterator_next(it_s)
                    #     event_s = dbp_iterator_current(it_s)
                    #     continue

                    event_flags = dbp_event_get_flags(event_s)
                    event_handle_id = int(dbp_event_get_handle_id(event_s))
                    event_id = int(dbp_event_get_event_id(event_s))
                    duration = end - begin
                    event_name = profile.type_key_to_name[event_key]

                    #####################################
                    # not all events have info
                    # also, not all events have the same info.
                    # so this is where users must add code to translate
                    # their own info objects
                    event_info = None
                    unique_id = len(profile.series)
                    
                    if not skip_infos:
                        cinfo = dbp_event_get_info(event_e)
                        if cinfo != NULL:
                            if ('PINS_EXEC' in profile.event_types and
                                event_key == profile.event_types['PINS_EXEC'].key):
                                cast_exec_info = <papi_exec_info_t *>cinfo
                                kernel_name = str(cast_exec_info.kernel_name)
                                event_info = {
                                    'kernel_type':
                                    cast_exec_info.kernel_type,
                                    'kernel_name':
                                    kernel_name,
                                    'vp_id':
                                    cast_exec_info.vp_id,
                                    'thread_id':
                                    cast_exec_info.th_id,
                                    'exec_info':
                                    [cast_exec_info.values[x] for x
                                     in range(cast_exec_info.values_len)]}
                            elif ('PINS_SELECT' in profile.event_types and
                                  event_key == profile.event_types['PINS_SELECT'].key):
                                cast_select_info = <select_info_t *>cinfo
                                kernel_name = str(cast_select_info.kernel_name)
                                event_info = {
                                    'kernel_type':
                                    cast_select_info.kernel_type,
                                    'kernel_name':
                                    kernel_name,
                                    'vp_id':
                                    cast_select_info.vp_id,
                                    'th_id':
                                    cast_select_info.th_id,
                                    'victim_vp_id':
                                    cast_select_info.victim_vp_id,
                                    'victim_thread_id':
                                    cast_select_info.victim_th_id,
                                    'exec_context':
                                    cast_select_info.exec_context,
                                    'values':
                                    [cast_select_info.values[x] for x
                                     in range(cast_select_info.values_len)]}
                            elif ('PINS_SOCKET' in profile.event_types and
                                  event_key == profile.event_types['PINS_SOCKET'].key):
                                cast_socket_info = <papi_socket_info_t *>cinfo
                                event_info = [
                                    cast_socket_info.vp_id,
                                    cast_socket_info.th_id,
                                    [cast_socket_info.values[x] for x
                                     in range(cast_socket_info.values_len)]]
                            elif ('PINS_L12_EXEC' in profile.event_types and
                                  event_key == profile.event_types['PINS_L12_EXEC'].key):
                                cast_L12_exec_info = <papi_L12_exec_info_t *>cinfo
                                kernel_name = cast_L12_exec_info.kernel_name[:12]
                                event_info = {
                                    'unique_id':
                                    unique_id,
                                    'kernel_type':
                                    cast_L12_exec_info.kernel_type,
                                    'kernel_name':
                                    kernel_name,
                                    'vp_id':
                                    cast_L12_exec_info.vp_id,
                                    'thread_id':
                                    cast_L12_exec_info.th_id,
                                    'PAPI_L1':
                                    cast_L12_exec_info.L1_misses,
                                    'PAPI_L2':
                                     cast_L12_exec_info.L2_misses
                                }
                            elif ('PINS_L12_SELECT' in profile.event_types and
                                  event_key == profile.event_types['PINS_L12_SELECT'].key):
                                cast_L12_select_info = <papi_L12_select_info_t *>cinfo
                                kernel_name = cast_L12_select_info.kernel_name[:KERNEL_NAME_SIZE]
                                event_info = {
                                    'unique_id':
                                    unique_id,
                                    'kernel_type':
                                    cast_L12_select_info.kernel_type,
                                    'kernel_name':
                                    kernel_name,
                                    'vp_id':
                                    cast_L12_select_info.vp_id,
                                    'thread_id':
                                    cast_L12_select_info.th_id,
                                    'victim_vp_id':
                                    cast_L12_select_info.victim_vp_id,
                                    'victim_thread_id':
                                    cast_L12_select_info.victim_th_id,
                                    'starvation':
                                    cast_L12_select_info.starvation,
                                    'exec_context':
                                    cast_L12_select_info.exec_context,
                                    'PAPI_L1':
                                    cast_L12_select_info.L1_misses,
                                    'PAPI_L2':
                                     cast_L12_select_info.L2_misses
                                }

                            elif ('PINS_L123' in profile.event_types and
                                  event_key == profile.event_types['PINS_L123'].key):
                                cast_L123_info = <papi_L123_info_t *>cinfo
                                event_info = {
                                    'unique_id': 
                                    unique_id,
                                    'vp_id':
                                    cast_L123_info.vp_id,
                                    'thread_id':
                                    cast_L123_info.th_id,
                                    'PAPI_L1':
                                    cast_L123_info.L1_misses,
                                    'PAPI_L2':
                                    cast_L123_info.L2_misses,
                                    'PAPI_L3':
                                    cast_L123_info.L3_misses
                                }
                            elif ('PINS_L12_ADD' in profile.event_types and
                                  event_key == profile.event_types['PINS_L12_ADD'].key):
                                cast_L12_exec_info = <papi_L12_exec_info_t *>cinfo
                                kernel_name = cast_L12_exec_info.kernel_name[:12]
                                event_info = {
                                    'unique_id':
                                    unique_id,
                                    'kernel_type':
                                    cast_L12_exec_info.kernel_type,
                                    'kernel_name':
                                    kernel_name,
                                    'vp_id':
                                    cast_L12_exec_info.vp_id,
                                    'thread_id':
                                    cast_L12_exec_info.th_id,
                                    'PAPI_L1':
                                    cast_L12_exec_info.L1_misses,
                                    'PAPI_L2':
                                    cast_L12_exec_info.L2_misses
                                }
                            # elif ('<EVENT_TYPE_NAME>' in profile.event_types and
                            #       event.key == profile.event_types['<EVENT_TYPE_NAME>'].key):
                            #   event_info = <write a function and a Python type to translate>
                            else:
                                dont_print = True
                                if not dont_print:
                                    print('missed an info for event key ' + event_name )

                    if event_info:
                        profile.infos.append(event_info)
#                        thing.update(event_info)
                    profile.series.append([
                                           thread.file.rank, thread.id, event_key, event_flags,
                                           event_handle_id, event_id,
                                           begin, end, duration, unique_id]) #event_info])
                    ['filerank', 'thread', 'key', 'flags', 'handle_id',
                     'id', 'begin', 'end', 'duration', 'unique_id']
                    # thing = {'filerank':thread.file.rank,
                    #                        'thread':thread.id,
                    #                        'key':event_key,
                    #                        'flags':event_flags,
                    #                        'handle_id':event_handle_id,
                    #                        'id':event_id,
                    #                        'begin':begin,
                    #                        'end':end,
                    #                        'duration':duration,
                    #                        }
                    # profile.series.append(thing)
                    
                dbp_iterator_delete(it_e)
                it_e = NULL
            else:
                if event_name not in profile.errors:
                    profile.errors[event_name] = []
                profile.errors[event_name].append('event of class {} id {} at {} does not have a match.\n'.format(
                    event_name, event_id, thread_id))
        dbp_iterator_next(it_s)
        event_s = dbp_iterator_current(it_s)
        
    dbp_iterator_delete(it_s)
    it_s = NULL

    return thread
