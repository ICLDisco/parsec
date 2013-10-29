####################################
# DBPreader Python interface
# run 'python setup.py build_ext --inplace' to compile
# import parsec_binprof
#
# Cython 0.18+ required. 
# pandas 0.12+ (and numpy, etc.) required.
# Python 2.7.3 recommended.
#
# Be SURE to build this against the same version of Python as you have built Cython itself.
# Contrasting versions will likely lead to odd errors about Unicode functions.

# cython: profile=False
# ...but could be True if we wanted to # import cProfile, pstats
from __future__ import print_function
import sys
from operator import attrgetter, itemgetter
from libc.stdlib cimport malloc, free
from parsec_profiling import * # the pure Python classes
import pandas as pd
import time
import re
import shutil

# reads an entire profile into a set of pandas DataFrames
cpdef read(filenames, report_progress=False, info_only=False):
    cdef dbp_file_t * cfile
    cdef dbp_dictionary_t * cdict
    cdef char ** c_filenames = string_list_to_c_strings(filenames)
    cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)

    nb_dict_entries = dbp_reader_nb_dictionary_entries(dbp)
    nb_files = dbp_reader_nb_files(dbp)
    worldsize = dbp_reader_worldsize(dbp)
    last_error = dbp_reader_last_error(dbp)

    # create event dictionaries first, for later use while reading events
    builder = ProfileBuilder()
    for key in range(nb_dict_entries):
        cdict = dbp_reader_get_dictionary(dbp, key)
        event_name = dbp_dictionary_name(cdict)
        builder.type_names[key] = event_name
        builder.event_types[event_name] = {'key':key, 'attributes':str(dbp_dictionary_attributes(cdict))}
    builder.type_names[-1] = '' # this is the default, for kernels without names

    # start with our nodes in the correct order
    node_order = dict()
    for i in range(nb_files):
        cfile = dbp_reader_get_file(dbp, i)
        node_id = dbp_file_get_rank(cfile)
        node_order[node_id] = i

    total_threads = 0
    with Timer() as t:
        # read the file for each node
        for node_id in sorted(node_order.keys()):
            cfile = dbp_reader_get_file(dbp, node_order[node_id])
            node_dct = {'exe':dbp_file_hr_id(cfile),
                        'filename':dbp_file_get_name(cfile),
                        'id':node_id}
            for index in range(dbp_file_nb_infos(cfile)):
                cinfo = dbp_file_get_info(cfile, index)
                key = dbp_info_get_key(cinfo)
                value = dbp_info_get_value(cinfo)
                add_kv(node_dct, key, value)
            builder.nodes.append(node_dct)
    
            # now read threads of this node
            num_threads = dbp_file_nb_threads(cfile)
            total_threads += num_threads
            builder.unordered_threads = dict()
            for thread_num in range(num_threads):
                construct_thread(builder, dbp, cfile, node_id, thread_num, info_only)
                cond_print('.', report_progress, end='')
                sys.stdout.flush()
            # sort threads
            for thread_num in range(num_threads):
                builder.threads.append(builder.unordered_threads[thread_num])
    # report progress
    cond_print('\nParsing the PBP files took ' + str(t.interval) + ' seconds, ' , 
               report_progress, end='')
    cond_print('which is ' + str(t.interval/total_threads) 
               + ' seconds per thread.', report_progress)

    # now, some voodoo to add shared file information to overall profile info
    # e.g., PARAM_N, PARAM_MB, etc.
    # basically, any key that has the same value in all nodes should
    # go straight into the top-level 'information' dictionary, since it is global
    builder.information.update(builder.nodes[0])
    for node in builder.nodes:
        for key, value in node.iteritems():
            if key in builder.information.keys():
                if builder.information[key] != node[key] and node[key] != 0:
                    del builder.information[key]
    builder.information['nb_nodes'] = nb_files
    builder.information['worldsize'] = worldsize
    builder.information['last_error'] = last_error

    cond_print('Then we construct the main DataFrames....', report_progress)
    if len(builder.events) > 0:
        with Timer() as t:
            events = pd.DataFrame(builder.events, columns=ParsecProfile.basic_event_columns)
        cond_print('   events dataframe construction time: ' + str(t.interval), report_progress)
    else: # b/c pd.DataFrame chokes on an empty list
        events = pd.DataFrame()
        cond_print('   No events were found.', report_progress)
    with Timer() as t:
        infos = pd.DataFrame.from_records(builder.infos)
    cond_print('   infos dataframe construction time: ' + str(t.interval), report_progress)
    if len(infos) > 0: # merge also chokes on an empty dataframe
        cond_print('Next, we merge them by their unique id.', report_progress)
        with Timer() as t:
            events = events.merge(infos, on='unique_id', how='outer')
        cond_print('   join/merge time: ' + str(t.interval), report_progress)

    with Timer() as t:
        information = pd.Series(builder.information)
        event_types = pd.DataFrame.from_records(builder.event_types)
        type_names = pd.Series(builder.type_names)
        nodes = pd.DataFrame.from_records(builder.nodes)
        threads = pd.DataFrame.from_records(builder.threads)
        if len(builder.errors) > 0:
            errors = pd.DataFrame(builder.errors,
                                  columns=ParsecProfile.basic_event_columns + ['error_msg'])
        else:
            errors = pd.DataFrame()
    cond_print('Constructed additional structures in {} seconds.'.format(t.interval), 
               report_progress)

    profile = ParsecProfile(events, event_types, type_names, 
                            nodes, threads, errors, information)

    dbp_reader_close_files(dbp) # does nothing as of 2013-04-21
#   dbp_reader_dispose_reader(dbp)
    free(c_filenames)

    return profile

# just a shortcut for passing info_only
cpdef get_info(filenames):
    return read(filenames, info_only=True)

cpdef convert(filenames, outfilename, unlink=True, table=False, append=False, report_progress=False):
    profile = read(filenames, report_progress=report_progress)
    store = profile.to_hdf(outfilename, table=table, append=append)
    store.close()
    if unlink:
        for filename in filenames:
            shutil.unlink(filename)

# This function helps support duplicate keys in the info dictionaries 
# by appending the extra values to a list stored at '<key>_list' in the dictionary.
# It also attempts to store values as numbers instead of strings, if possible.
cpdef add_kv(dct, key, value):
    try:    # try to convert value to its number type
        value = float(value) if '.' in value else int(value)
    except: # if this fails, it's a string, and that's fine too
        pass
    if key not in dct:
        # the first value we find is generally the
        # last value added to the profile, so we 'prefer' it
        # by putting it directly in the dictionary.
        dct[key] = value 
    else:
        list_k = key + '_list'
        if list_k in dct:
            if isinstance(dct[list_k], list):
                dct[list_k].append(value)
            else:
                print('ignoring secondary value of ' + 
                      str(value) + ' for ' + str(key))
        else:
            dct[list_k] = [value]

# helper function for readProfile
cdef char** string_list_to_c_strings(strings):
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

thread_id_in_descrip = re.compile('.*thread\s+(\d+).*', re.IGNORECASE)
vp_id_in_descrip = re.compile('.*VP\s+(\d+).*', re.IGNORECASE)

# you can't call this. it will be called for you. call readProfile()
cdef construct_thread(builder, dbp_multifile_reader_t * dbp, dbp_file_t * cfile, 
                   int node_id, int thread_num, int info_only):
    cdef dbp_thread_t * cthread = dbp_file_get_thread(cfile, thread_num)
    cdef dbp_event_iterator_t * it_s = dbp_iterator_new_from_thread(cthread)
    cdef dbp_event_iterator_t * it_e = NULL
    cdef const dbp_event_t * event_s = dbp_iterator_current(it_s)
    cdef const dbp_event_t * event_e = NULL
    cdef dbp_info_t * th_info = NULL
    cdef uint64_t begin = 0
    cdef uint64_t end = 0
    cdef void * cinfo = NULL
    cdef papi_exec_info_t * cast_exec_info = NULL
    cdef select_info_t * cast_select_info = NULL
    cdef papi_L123_info_t * cast_L123_info = NULL
    cdef papi_L12_select_info_t * cast_L12_select_info = NULL
    cdef papi_L12_exec_info_t * cast_L12_exec_info = NULL
    cdef long long int * cast_lld_ptr = NULL

    thread_descrip = dbp_thread_get_hr_id(cthread)
    thread = {'node_id': node_id, 'description':thread_descrip}

    for i in range(dbp_thread_nb_infos(cthread)):
        th_info = dbp_thread_get_info(cthread, i)
        key = dbp_info_get_key(th_info);
        value = dbp_info_get_value(th_info);
        add_kv(thread, key, value)

    if not 'id' in thread:
        thread_id = thread_num
    else:
        thread_id = thread['id']
    builder.unordered_threads[thread_id] = thread

    while event_s != NULL and not info_only:
        event_key = dbp_event_get_key(event_s) / 2 # to match dictionary
        event_name = builder.type_names[event_key]
        begin = dbp_event_get_timestamp(event_s)
        event_flags = dbp_event_get_flags(event_s)
        handle_id = dbp_event_get_handle_id(event_s)
        event_id = dbp_event_get_event_id(event_s)
        unique_id = len(builder.events)

        if KEY_IS_START( dbp_event_get_key(event_s) ):
            it_e = dbp_iterator_find_matching_event_all_threads(it_s, 0)
            if it_e == NULL:
                event = [node_id, thread_id, handle_id, event_key, 
                         begin, None, 0, event_flags, unique_id, event_id]
                error_msg = 'event of class {} id {} at {} does not have a match.\n'.format(
                    event_name, event_id, thread_id)
                builder.errors.append(event + [error_msg])
            else:
                event_e = dbp_iterator_current(it_e)
                if event_e != NULL:
                    end = dbp_event_get_timestamp(event_e)
                    duration = end - begin

                    event = [node_id, thread_id, handle_id, event_key, 
                             begin, end, duration, event_flags, unique_id, event_id]

                    if end < begin:
                        dbp_iterator_delete(it_e)
                        it_e = NULL
                        dbp_iterator_next(it_s)
                        event_s = dbp_iterator_current(it_s)
                        error_msg = 'event of class {} id {} at {} has a negative duration.\n'.format(
                            event_name, event_id, thread_id)
                        builder.errors.append(event + [error_msg])
                        continue

                    builder.events.append(event)

                    #####################################
                    # not all events have info
                    # also, not all events have the same info.
                    # so this is where users must add code to translate
                    # their own info objects
                    event_info = None
                    cinfo = dbp_event_get_info(event_e)
                    if cinfo != NULL:
                        if ('PINS_EXEC' in builder.event_types and
                            event_key == builder.event_types.PINS_EXEC['key']):
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
                        elif ('PINS_SELECT' in builder.event_types and
                              event_key == builder.event_types.PINS_SELECT['key']):
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
                        elif ('PINS_SOCKET' in builder.event_types and
                              event_key == builder.event_types['PINS_SOCKET']['key']):
                            cast_socket_info = <papi_socket_info_t *>cinfo
                            event_info = [
                                cast_socket_info.vp_id,
                                cast_socket_info.th_id,
                                [cast_socket_info.values[x] for x
                                 in range(cast_socket_info.values_len)]]
                        # START NEW, IN-USE INFOS
                        elif ('PINS_L12_EXEC' in builder.event_types and
                              event_key == builder.event_types['PINS_L12_EXEC']['key']):
                            cast_L12_exec_info = <papi_L12_exec_info_t *>cinfo
                            # kernel_name = cast_L12_exec_info.kernel_name[:12]
                            event_info = {
                                'unique_id':
                                unique_id,
                                'kernel_type':
                                cast_L12_exec_info.kernel_type,
                                'PAPI_L1':
                                cast_L12_exec_info.L1_misses,
                                'PAPI_L2':
                                 cast_L12_exec_info.L2_misses
                            }
                        elif ('PINS_L12_SELECT' in builder.event_types and
                              event_key == builder.event_types['PINS_L12_SELECT']['key']):
                            cast_L12_select_info = <papi_L12_select_info_t *>cinfo
                            event_info = {
                                'unique_id':
                                unique_id,
                                'kernel_type':
                                cast_L12_select_info.kernel_type,
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
                            # kernel_name_test = builder.test_df[event_info['kernel_type']]
                            # if kernel_name_test == '':
                            #     print(kernel_name_test + str(event_info['kernel_type']))

                        elif ('PINS_L123' in builder.event_types and
                              event_key == builder.event_types['PINS_L123']['key']):
                            cast_L123_info = <papi_L123_info_t *>cinfo
                            event_info = {
                                'unique_id':
                                unique_id,
                                'PAPI_L1':
                                cast_L123_info.L1_misses,
                                'PAPI_L2':
                                cast_L123_info.L2_misses,
                                'PAPI_L3':
                                cast_L123_info.L3_misses
                            }
                        elif ('PINS_L12_ADD' in builder.event_types and
                              event_key == builder.event_types['PINS_L12_ADD']['key']):
                            cast_L12_exec_info = <papi_L12_exec_info_t *>cinfo
                            event_info = {
                                'unique_id':
                                unique_id,
                                'kernel_type':
                                cast_L12_exec_info.kernel_type,
                                'PAPI_L1':
                                cast_L12_exec_info.L1_misses,
                                'PAPI_L2':
                                cast_L12_exec_info.L2_misses
                            }
                        # elif ('<EVENT_TYPE_NAME>' in builder.event_types and
                        #       event['key'] == builder.event_types['<EVENT_TYPE_NAME>']['key']):
                        #   event_info = <write a function and a Python type to translate>
                        else:
                            dont_print = True
                            if not dont_print:
                                print('missed an info for event key ' + event_name )

                    if event_info:
                        builder.infos.append(event_info)

                dbp_iterator_delete(it_e)
                it_e = NULL
        dbp_iterator_next(it_s)
        event_s = dbp_iterator_current(it_s)

    dbp_iterator_delete(it_s)
    it_s = NULL

# NOTE:
# this breaks Cython, so don't do it
# index = -1
# print('index is ' + str(index))
# print(builder.test_df[index])

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

# private utility class
class ProfileBuilder(object):
    def __init__(self):
        self.events = list()
        self.infos = list()
        self.event_types = dict()
        self.type_names = dict()
        self.nodes = list()
        self.threads = list()
        self.errors = list()
        self.information = dict()

def cond_print(string, cond, **kwargs):
    if cond:
        print(string, **kwargs)

