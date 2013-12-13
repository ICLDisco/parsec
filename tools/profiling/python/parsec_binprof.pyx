""" DBPreader Python interface

run 'python setup.py build_ext --inplace' to compile
The preferred nomenclature for the Python Binary Profile is "PBP",
therefore it is recommended to do "import parsec_binprof as pbp"
in Python programs using this module.

REQUIREMENTS:
# Cython 0.18+ required.
# pandas 0.12+ (and numpy, etc.) required.
# Python 2.7.3 recommended.

BUILD NOTES:
# Be SURE to build this against the same version of Python as you have built Cython itself.
# Contrasting versions will likely lead to odd errors about Unicode functions.
"""

# cython: profile=False
# ...but could be True if we wanted to # import cProfile, pstats
from __future__ import print_function
import sys, os
from operator import attrgetter, itemgetter
from libc.stdlib cimport malloc, free
from parsec_profiling import * # the pure Python classes
import pandas as pd
import time
import re
from multiprocessing import Process, Pipe
import multiprocessing

# 'include' will eventually be deprecated by Cython, but I still prefer it.
include "pbp_info_parser.pxi"

multiprocess_io_cap = 9 # this seems to be a good default on ICL machines
pbp_core = '.prof-'

# reads an entire profile into a set of pandas DataFrames
# filenames ought to be a list of strings, or comparable type.
cpdef read(filenames, report_progress=False, skeleton_only=False, multiprocess=True):
    cdef dbp_file_t * cfile
    cdef dbp_dictionary_t * cdict
    if isinstance(filenames, basestring): # if the user passed a single string
        filenames = [filenames]
    cdef char ** c_filenames = string_list_to_c_strings(filenames)
    cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)

    # determine amount of multiprocessing
    if isinstance(multiprocess, bool):
        if multiprocess:
            multiprocess = multiprocessing.cpu_count()
            if multiprocess > multiprocess_io_cap:
                multiprocess = multiprocess_io_cap
        else:
            multiprocess = 1
    elif multiprocess < 1:
        multiprocess = 1

    nb_dict_entries = dbp_reader_nb_dictionary_entries(dbp)
    nb_files = dbp_reader_nb_files(dbp)
    worldsize = dbp_reader_worldsize(dbp)
    last_error = dbp_reader_last_error(dbp)

    # create event dictionaries first, for later use while reading events
    builder = ProfileBuilder()
    for event_type in range(nb_dict_entries):
        cdict = dbp_reader_get_dictionary(dbp, event_type)
        event_name = dbp_dictionary_name(cdict)
        builder.event_names[event_type] = event_name
        builder.event_types[event_name] = event_type
        builder.event_attributes[event_type] = str(dbp_dictionary_attributes(cdict))
    builder.event_names[-1] = '' # this is the default, for kernels without names

    # start with our nodes in the correct order
    builder.node_order = dict()
    for i in range(nb_files):
        cfile = dbp_reader_get_file(dbp, i)
        node_id = dbp_file_get_rank(cfile)
        builder.node_order[node_id] = i

    # read the file for each node
    node_threads = []
    for node_id in sorted(builder.node_order.keys()):
        cfile = dbp_reader_get_file(dbp, builder.node_order[node_id])
        node_dct = {'exe':dbp_file_hr_id(cfile),
                    'filename':dbp_file_get_name(cfile),
                    'id':node_id,
                    'error':dbp_file_error(cfile)}
        for index in range(dbp_file_nb_infos(cfile)):
            cinfo = dbp_file_get_info(cfile, index)
            key = dbp_info_get_key(cinfo)
            value = dbp_info_get_value(cinfo)
            add_kv(node_dct, key, value)
        builder.nodes.append(node_dct)
        # record threads for this node
        builder.unordered_threads_by_node[node_id] = dict()
        num_threads = dbp_file_nb_threads(cfile)
        node_threads += [(node_id, thread_num) for thread_num in range(num_threads)]

    # now split our work by the number of worker processes we're using
    if len(node_threads) < multiprocess:
        multiprocess = len(node_threads)
    node_thread_chunks = chunk(node_threads, multiprocess)
    process_pipes = list()
    processes = list()
    
    with Timer() as t:
        for nt_chunk in node_thread_chunks:
            my_end, their_end = Pipe()
            process_pipes.append(my_end)
            p = Process(target=construct_thread_in_process, args=
                        (their_end, builder, filenames, 
                         nt_chunk, skeleton_only, report_progress))
            processes.append(p)
            p.start()
        while process_pipes:
            any_read = False
            for pipe in process_pipes:
                try:
                    if not pipe.poll():
                        continue
                    any_read = True
                    events, errors, threads = pipe.recv()
                    for node_id, thread in threads.iteritems():
                        builder.unordered_threads_by_node[node_id].update(thread)
                    builder.events.append(events)
                    builder.errors.append(errors)
                    cond_print('<', report_progress, end='') # print comms progress
                    sys.stdout.flush()
                    process_pipes.remove(pipe)
                except EOFError:
                    process_pipes.remove(pipe)
            if not any_read:
                time.sleep(0.05) # tiny sleep so as not to hog CPU
        for p in processes:
            p.join() # cleanup spawned processes
    # report progress
    cond_print('\nParsing the PBP files took ' + str(t.interval) + ' seconds, ' ,
               report_progress, end='')
    cond_print('which is ' + str(t.interval/len(node_threads))
               + ' seconds per thread.', report_progress)

    # sort threads
    for node_id in sorted(builder.unordered_threads_by_node.keys()):
        for thread_num in sorted(builder.unordered_threads_by_node[node_id].keys()):
            builder.threads.append(builder.unordered_threads_by_node[node_id][thread_num])

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

    cond_print('Then we concatenate the event DataFrames....', report_progress)
    with Timer() as t:
        events = pd.concat(builder.events)
    cond_print('   events DataFrame concatenation time: ' + str(t.interval), report_progress)

    with Timer() as t:
        information = pd.Series(builder.information)
        event_types = pd.Series(builder.event_types)
        event_names = pd.Series(builder.event_names)
        event_attributes = pd.Series(builder.event_attributes)
        nodes = pd.DataFrame.from_records(builder.nodes)
        threads = pd.DataFrame.from_records(builder.threads)
        errors = pd.concat(builder.errors)
    cond_print('Constructed additional structures in {} seconds.'.format(t.interval),
               report_progress)

    profile = ParsecProfile(events, event_types, event_names, event_attributes,
                            nodes, threads, information, errors)

    dbp_reader_close_files(dbp) # does nothing as of 2013-04-21
#   dbp_reader_dispose_reader(dbp)
    free(c_filenames)

    return profile

# just a shortcut for passing skeleton_only
cpdef get_info(filenames):
    return read(filenames, skeleton_only=True)

# returns the output filename in a list, not the profile itself.
cpdef convert(filenames, outfilename=None, unlink=True, multiprocess=True,
              force_reconvert=False, validate_existing=False,
              table=False, append=False, report_progress=False):
    if outfilename == None:
        outfilename = filenames[0].replace('.prof-', '.h5-')
    if os.path.exists(outfilename):
        if not force_reconvert:
            try:
                if validate_existing:
                    ParsecProfile.from_hdf(outfilename, skeleton_only=True)
                cond_print(
                    'P3 {} already exists. '.format(
                        os.path.basename(outfilename)) +
                    'Conversion not forced.', report_progress)
                return outfilename # file already exists
            except:
                cond_print(
                    'P3 {} already exists, but cannot be validated. '.format(
                        os.path.basename(outfilename)) +
                    'Conversion will proceed.', report_progress)
                pass # something went wrong, so try conversion anyway
    # convert
    cond_print('Converting {}'.format(filenames), report_progress)
    profile = read(filenames, report_progress=report_progress, multiprocess=multiprocess)
    # write file
    with Timer() as t:
        profile.to_hdf(outfilename, table=table, append=append)
    cond_print('Wrote profile to HDF5 format 1in {} seconds.'.format(t.interval), report_progress)
    if unlink:
        for filename in filenames:
            cond_print('Unlinking {} after conversion'.format(filename), report_progress)
            os.unlink(filename)
    return outfilename


# This function helps support duplicate keys in the info dictionaries
# by appending the extra values to a list stored at '<key>_list' in the dictionary.
# It also attempts to store values as numbers instead of strings, if possible.
cpdef add_kv(dct, key, value, append_if_present=True):
    try:    # try to convert value to its number type
        value = float(value) if '.' in value else int(value)
    except: # if this fails, it's a string, and that's fine too
        pass
    if key not in dct:
        # the first value we find is generally the
        # last value added to the profile, so we 'prefer' it
        # by putting it directly in the dictionary.
        dct[key] = value
    elif append_if_present:
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


cpdef construct_thread_in_process(pipe, builder, filenames, node_threads, 
                                  skeleton_only, report_progress):
    cdef dbp_file_t * cfile
    cdef char ** c_filenames = string_list_to_c_strings(filenames)
    cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)
    
    for node_id, thread_num in node_threads: # should be list of tuples
        cfile = dbp_reader_get_file(dbp, builder.node_order[node_id])
        construct_thread(builder, dbp, cfile, node_id, thread_num, skeleton_only)
        cond_print('.', report_progress, end='')
        sys.stdout.flush()
    
    # now we must send the constructed objects back to the spawning process
    if len(builder.events) > 0:
        builder.events = pd.DataFrame.from_records(builder.events)
    else:
        builder.events = pd.DataFrame()
    if len(builder.errors) > 0:
        builder.errors = pd.DataFrame.from_records(builder.errors)
    else:
        builder.errors = pd.DataFrame()
    pipe.send((builder.events, builder.errors, builder.unordered_threads_by_node))


thread_id_in_descrip = re.compile('.*thread\s+(\d+).*', re.IGNORECASE)
vp_id_in_descrip = re.compile('.*VP\s+(\d+).*', re.IGNORECASE)

cdef construct_thread(builder, dbp_multifile_reader_t * dbp, dbp_file_t * cfile,
                      int node_id, int thread_num, int skeleton_only):
    """Converts all events using the C interface into Python dicts

    Also creates a 'thread' dict describing the very basic information
    about the thread as seen by PaRSEC. Hopefully the information
    we store about the thread will continue to improve in the future.
    """
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
    cdef papi_core_socket_info_t * cast_core_socket_info = NULL
    cdef papi_core_select_info_t * cast_core_select_info = NULL
    cdef papi_core_exec_info_t * cast_core_exec_info = NULL

    th_begin = sys.maxint
    th_end = 0
    thread_descrip = dbp_thread_get_hr_id(cthread)
    thread = {'node_id': node_id, 'description': thread_descrip}

    for i in range(dbp_thread_nb_infos(cthread)):
        th_info = dbp_thread_get_info(cthread, i)
        key = dbp_info_get_key(th_info);
        value = dbp_info_get_value(th_info);
        add_kv(thread, key, value)

    # sanity check events
    try:
        th_duration = thread['end'] - thread['begin']
    except:
        th_duration = sys.maxint

    if not 'id' in thread:
        thread_id = thread_num
    else:
        thread_id = thread['id']
    builder.unordered_threads_by_node[node_id][thread_id] = thread

    while event_s != NULL and not skeleton_only:
        event_type = dbp_event_get_key(event_s) / 2 # to match dictionary
        event_name = builder.event_names[event_type]
        begin = dbp_event_get_timestamp(event_s)
        event_flags = dbp_event_get_flags(event_s)
        handle_id = dbp_event_get_handle_id(event_s)
        event_id = dbp_event_get_event_id(event_s)
        if begin < th_begin:
            th_begin = begin

        if KEY_IS_START( dbp_event_get_key(event_s) ):
            it_e = dbp_iterator_find_matching_event_all_threads(it_s, 0)
            if it_e != NULL:

                event_e = dbp_iterator_current(it_e)

                if event_e != NULL:
                    end = dbp_event_get_timestamp(event_e)
                    # 'end' and 'begin' are unsigned, so subtraction is invalid if they are
                    if end > begin:
                        duration = end - begin
                    else:
                        duration = -1
                    event = {'node_id':node_id, 'thread_id':thread_id, 'handle_id':handle_id, 
                             'type':event_type, 'begin':begin, 'end':end, 'duration':duration, 
                             'flags':event_flags, 'id':event_id}

                    if duration >= 0 and duration <= th_duration:
                        # VALID EVENT FOUND
                        builder.events.append(event)
                        cinfo = dbp_event_get_info(event_e)
                        if cinfo != NULL:
                            event_info = parse_info(builder, event_type, cinfo)
                            if event_info:
                                event.update(event_info)
                        if th_end < end:
                            th_end = end
                    else: # the event is 'not sane'
                        error_msg = ('event of class {} id {} at {}'.format(
                            event_name, event_id, thread_id) +
                                     ' has a unreasonable duration.\n')
                        event.update({'error_msg':error_msg})
                        builder.errors.append(event)

                dbp_iterator_delete(it_e)
                it_e = NULL

            else: # the event is not complete
                error_msg = 'event of class {} id {} at {} does not have a match.\n'.format(
                    event_name, event_id, thread_id)
                error = {'node_id':node_id, 'thread_id':thread_id, 'handle_id':handle_id, 
                         'type':event_type, 'begin':begin, 'end':0, 'duration':0, 
                         'flags':event_flags, 'id':event_id, 'error_msg': error_msg}
                builder.errors.append(error)

        dbp_iterator_next(it_s)
        event_s = dbp_iterator_current(it_s)

    dbp_iterator_delete(it_s)
    it_s = NULL

    add_kv(thread, 'begin', th_begin, append_if_present=False)
    add_kv(thread, 'end', th_end, append_if_present=False)
    add_kv(thread, 'duration', thread['end'] - thread['begin'], append_if_present=False)
    # END construct_thread



class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

# private utility class
class ProfileBuilder(object):
    def __init__(self):
        self.events = list()
        self.infos = list()
        self.event_types = dict()
        self.event_names = dict()
        self.event_attributes = dict()
        self.nodes = list()
        self.unordered_threads_by_node = dict()
        self.threads = list()
        self.errors = list()
        self.information = dict()

def cond_print(string, cond, **kwargs):
    if cond:
        print(string, **kwargs)

# NOTE:
# this breaks Cython, so don't do it
# index = -1
# print('index is ' + str(index))
# print(builder.test_df[index])

def chunk(xs, n):
    ys = list(xs)
    ylen = len(ys)
    size = int(ylen / n)
    chunks = [ys[0+size*i : size*(i+1)] for i in xrange(n)]
    leftover = ylen - size*n
    edge = size*n
    for i in xrange(leftover):
        chunks[i%n].append(ys[edge+i])
    return chunks
