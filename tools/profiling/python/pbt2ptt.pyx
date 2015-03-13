""" DBPreader Python interface

run 'python setup.py build_ext --inplace' to compile
The preferred nomenclature for the Python Binary Trace is "PBT",

REQUIREMENTS:
# Cython 0.19+ required.
# pandas 0.13+ (and numpy, etc.) required.
# Python 2.7.3 recommended.

BUILD NOTES:
# Be SURE to build this against the same version of Python as you have built Cython itself.
# Contrasting versions will likely lead to odd errors about Unicode functions.

TERMINOLOGY NOTES:
# The phrase "skeleton_only" refers to loading only the metadata of a trace;
# i.e., everything but the events. This can significantly improve load times,
# especially in traces with many thousands of events, and is therefore useful
# when your scripts are only interested in comparing the basic trace information
# between multiple traces.
"""

# cython: trace=False
# ...but could be True if we wanted to # import cProfile, pstats
from __future__ import print_function

import sys
import os
import time
import re
from operator import attrgetter, itemgetter
from libc.stdlib cimport malloc, free
from multiprocessing import Process, Pipe
import multiprocessing

import pandas as pd

from parsec_trace_tables import * # the pure Python classes
from common_utils import *

multiprocess_io_cap = 9 # this seems to be a good default on ICL machines
microsleep = 0.05

cpdef read(filenames, report_progress=False, skeleton_only=False, multiprocess=False,
           add_info=dict()):
    """ Given binary trace filenames, returns a PaRSEC Trace Table (PTT) object

    Defaults in parentheses

    filenames should be a list-like of strings.

    report_progress (False) turns on stdout printing of trace load progress, useful
    for command line scripts.

    skeleton_only (False) will load everything but the events.

    multiprocess (True) specifies the use of multiple I/O and CPU threads to use during the load.
    An integer number may be specified instead of True.
    Setting skeleton_only will also set multiprocess == 1.

    add_info ({}) -- a dictionary to merge with the node information from the trace.
    This is useful in situations where the caller may have high level information
    about the trace that PaRSEC and the trace itself does not or cannot have,
    and where the caller wishes to embed that information at the time of PTT generation.
    """
    cdef dbp_file_t * cfile
    cdef dbp_dictionary_t * cdict
    if isinstance(filenames, basestring): # if the user passed a single string instead of a list
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
    if skeleton_only:
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
        builder.event_convertors[event_type] = None
        if str("PINS_EXEC") == event_name:
            event_conv = 'kernel_type{int32_t}:value1{int64_t}:value2{int64_t}:value3{int64_t}:'
        else:
            event_conv = dbp_dictionary_convertor(cdict)
        if 0 != len(event_conv):
            builder.event_convertors[event_type] = ExtendedEvent(builder.event_names[event_type],
                                                                 event_conv, dbp_dictionary_keylen(cdict))
            
    builder.event_names[-1] = '' # this is the default, for kernels without names

    # start with our nodes in the correct order
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
        try:
            node_dct['exe_abspath'] = os.path.abspath(
                node_dct['cwd'] + os.sep + node_dct['exe'])
            node_dct['exe'] = os.path.basename(node_dct['exe_abspath'])
        except KeyError as ke:
            pass

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
            something_was_read = False
            for pipe in process_pipes:
                try:
                    if not pipe.poll():
                        continue
                    something_was_read = True
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
            if not something_was_read:
                time.sleep(microsleep) # tiny sleep so as not to hog CPU
        for p in processes:
            p.join() # cleanup spawned processes
    # report progress
    cond_print('\nParsing the PBT files took ' + str(t.interval) + ' seconds' ,
               report_progress, end='')
    if len(node_threads) > 0:
        cond_print(', which is ' + str(t.interval/len(node_threads))
                   + ' seconds per thread.', report_progress)
    else:
        cond_print('\n', report_progress)

    # sort threads
    for node_id in sorted(builder.unordered_threads_by_node.keys()):
        for thread_num in sorted(builder.unordered_threads_by_node[node_id].keys()):
            builder.threads.append(builder.unordered_threads_by_node[node_id][thread_num])

    # now, some voodoo to add shared file information to overall trace info
    # e.g., PARAM_N, PARAM_MB, exe, SYNC_TIME_ELAPSED, etc.
    # basically, any key that has the same value in *all nodes* should
    # go straight into the top-level 'information' dictionary, since it is global
    if len(builder.nodes) > 0:
        builder.information.update(builder.nodes[0]) # start with all infos from node 0
    else:
        cond_print('No nodes were found in the trace.', report_progress)
    for node in builder.nodes:
        for key, value in node.iteritems(): # now remove, one by one, non-matching infos
            if key in builder.information.keys():
                if builder.information[key] != node[key] and node[key] != 0:
                    del builder.information[key]
    builder.information['nb_nodes'] = nb_files
    builder.information['worldsize'] = worldsize
    builder.information['last_error'] = last_error
    # allow the caller (who may know something extra about the run)
    # to specify additional trace information
    if add_info:
        for key, val in add_info.iteritems():
            add_kv(builder.information, key, val)

    if len(builder.events) > 0:
        cond_print('Then we concatenate the event DataFrames....', report_progress)
        with Timer() as t:
            events = pd.concat(builder.events)
        cond_print('   events DataFrame concatenation time: ' + str(t.interval), report_progress)
    else:
        events = pd.DataFrame()
        cond_print('No events were found in the trace.', report_progress)

    with Timer() as t:
        information = pd.Series(builder.information)
        event_types = pd.Series(builder.event_types)
        event_names = pd.Series(builder.event_names)
        event_attributes = pd.Series(builder.event_attributes)
        event_convertors = pd.Series(builder.event_convertors)
        nodes = pd.DataFrame.from_records(builder.nodes)
        threads = pd.DataFrame.from_records(builder.threads)
        if len(builder.errors) > 0:
            errors = pd.concat(builder.errors)
        else:
            errors = pd.DataFrame()
    cond_print('Constructed additional structures in {} seconds.'.format(t.interval),
               report_progress)

    trace = ParsecTraceTables(events, event_types, event_names, event_attributes, event_convertors,
                              nodes, threads, information, errors)

    dbp_reader_close_files(dbp) # does nothing as of 2013-04-21
#   dbp_reader_dispose_reader(dbp)
    free(c_filenames)

    return trace


cpdef convert(filenames, out=None, unlink=False, multiprocess=True,
              force_reconvert=False, validate_existing=False,
              table=False, append=False, report_progress=False,
              add_info=dict(), compress=('blosc', 0), skeleton_only=False):
    ''' Given [filenames] that comprise a single binary trace, returns the filename of the converted trace.

    filenames -- a list-like of strings

    out (None) -- allows manual specification of an output filename.

    unlink (False) -- will unlink the binary trace after successful conversion.

    skeleton_only, multiprocess, report_progress, add_info -- see docs for "read()"

    force_reconvert (False) -- causes conversion to happen even if a converted trace is determined to already exist.
    This is usually determined by a simple match of the default generated output filename.
    This (hopefully) simplifies significantly the mental/organiziational workload of the user,
    as traces can simply be "re-converted" each time from the binary trace without performing the
    actual conversion, while still allowing for a manual override in cases where that is necessary.

    validate_existing (False) -- requires a successful load of an existing converted trace, instead of just a filename match.

    table (False) -- allows PyTables 'tabular' storage of the PTT, which is slower but more flexible,
    especially in cases of extremely large traces, where the tabular format allows for database-style partial loads.
    See pandas & PyTables docs for more details.

    append (False) -- related to table. See pandas & PyTables docs.

    compress (('blosc', 0)) -- takes a tuple of compression algorithm and "level" from 0-9.
    Level 0 means no compression.
    '''
    if skeleton_only:
        compress=('blosc', 5)
    if len(filenames) < 1:
        cond_print('No filenames supplied for conversion!', report_progress)
        return None
    if len(filenames) == 1 and not force_reconvert:
        if is_ptt(filenames[0]):
            cond_print('File {} is already a PTT. Not converting.'.format(filenames[0]),
                       report_progress)
            return filenames[0]

    # check for existing .h5 (try not to re-convert unnecessarily)
    existing_h5s = find_h5_conflicts(filenames)
    if existing_h5s and not force_reconvert:
        cond_print('potential pre-existing PTTs: ' + str(existing_h5s), report_progress)
        existing_h5 = existing_h5s[0]
        # do skeleton read to check more carefully
        try:
            if validate_existing:
                from_hdf(existing_h5, skeleton_only=True)
            cond_print(
                'PTT {} already exists. '.format(
                    existing_h5) +
                'Conversion not forced.', report_progress)
            return existing_h5 # file already exists
        except:
            cond_print(
                'Possibly pre-existant PTT {} already exists, but cannot be validated. '.format(
                    existing_h5) +
                'Conversion will proceed.', report_progress)
            pass # something went wrong, so try conversion anyway

    # convert
    cond_print('Converting {}'.format(filenames), report_progress)
    trace = read(filenames, report_progress=report_progress,
                 multiprocess=multiprocess, add_info=add_info, skeleton_only=skeleton_only)

    if out == None: # create out filename
        out_dir = os.path.dirname(filenames[0])
        # if all the files exist in the same directory, output the conversion there
        for filename in filenames[1:]:
            if out_dir != os.path.dirname(filename):
                out_dir = None
                break
        try:
            rank_zero_filename = os.path.basename(trace.nodes.iloc[0]['filename'])
            # if we don't already have an out_dir
            if not out_dir:
                for filename in filenames:
                    if os.path.basename(filename) == rank_zero_filename:
                        out_dir = os.path.dirname(filename)
                        if not out_dir:
                            out_dir = '.'
                        break
            match = dot_prof_regex.match(rank_zero_filename)
            if match:
                infos = default_descriptors[:] + ['start_time']
                infos.remove('exe') # this is already in match.group(1)
                out = (match.group(1).strip('_') + '-' + trace.name(infos=infos) +
                       '-' + match.group(4) + '.h5')
            else:
                out = get_basic_ptt_name(rank_zero_filename)
        except Exception as e:
            print(e)
            out = get_basic_ptt_name(os.path.basename(filenames[0]))
        if not out_dir:
            out_dir = '.'
        out = out_dir + os.sep + out

    # write file
    with Timer() as t:
        trace.to_hdf(out, table=table, append=append,
                     complevel=compress[1], complib=compress[0])
    cond_print('Wrote trace to HDF5 format in {} seconds.'.format(t.interval), report_progress)
    if unlink:
        for filename in filenames:
            cond_print('Unlinking {} after conversion'.format(filename), report_progress)
            os.unlink(filename)
    return out


cpdef add_kv(dct, key, value, append_if_present=True):
    ''' Adds value for key to dict; converts strings to numbers if possible.

    If the key is already present, replaces the current value with the new value,
    and appends all old and new values in a list at key == <key>_list.
    '''
    try:    # try to convert value to its number type
        value = float(value) if '.' in value else int(value)
    except: # if this fails, it's a string, and that's fine too
        pass
    if key not in dct:
        # the first value we find is generally the
        # last value added to the trace, so we 'prefer' it
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


cdef char** string_list_to_c_strings(strings):
    ''' Converts a list of Python strings to C-style strings '''
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
    ''' Target function for the map/reduce threading functionality '''
    cdef dbp_file_t * cfile
    cdef char ** c_filenames = string_list_to_c_strings(filenames)
    # note that this requires re-opening the binary files in each thread
    cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)

    # node_threads is our thread-specific input data
    for node_id, thread_num in node_threads: # should be list of tuples
        cfile = dbp_reader_get_file(dbp, builder.node_order[node_id])
        construct_thread(builder, skeleton_only, dbp, cfile, node_id, thread_num)
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

cdef construct_thread(builder, skeleton_only, dbp_multifile_reader_t * dbp, dbp_file_t * cfile,
                      int node_id, int thread_num):
    """Converts all events using the C interface into a list of Python dicts

    Also creates a 'thread' dict describing the very basic information
    about the thread as seen by PaRSEC.

    Hopefully the information we store about the thread will continue to improve in the future.
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

    th_begin = sys.maxint
    th_end = 0
    thread_descrip = dbp_thread_get_hr_id(cthread)
    thread = {'node_id': node_id, 'thread_id': thread_num, 'description': thread_descrip}

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

    builder.unordered_threads_by_node[node_id][thread_num] = thread
    while event_s != NULL and not skeleton_only:
        event_type = dbp_event_get_key(event_s) / 2 # to match dictionary
        event_name = builder.event_names[event_type]
        begin = dbp_event_get_timestamp(event_s)
        event_flags = dbp_event_get_flags(event_s)
        handle_id = dbp_event_get_handle_id(event_s)
        event_id = dbp_event_get_event_id(event_s)
        if begin < th_begin:
            th_begin = begin

        # this would be a good place for a test for 'singleton' events.
        if KEY_IS_START( dbp_event_get_key(event_s) ):
            it_e = dbp_iterator_find_matching_event_all_threads(it_s, 0)
            if it_e != NULL:

                event_e = dbp_iterator_current(it_e)

                if event_e != NULL:
                    end = dbp_event_get_timestamp(event_e)
                    # 'end' and 'begin' are unsigned, so subtraction is invalid if they are
                    if end >= begin:
                        duration = end - begin
                    else:
                        duration = -1
                    event = {'node_id':node_id, 'thread_id':thread_num, 'handle_id':handle_id,
                             'type':event_type, 'begin':begin, 'end':end, 'duration':duration,
                             'flags':event_flags, 'id':event_id}

                    if duration >= 0 and duration <= th_duration:
                        # VALID EVENT FOUND
                        builder.events.append(event)
                        cinfo = dbp_event_get_info(event_e)
                        if cinfo != NULL:
                            if None != builder.event_convertors[event_type]:
                                event_info = parse_info(builder, event_type, <char*>cinfo)
                                if None != event_info:
                                    print(event_info)
                                    event.update(event_info)
                        if th_end < end:
                            th_end = end
                    else: # the event is 'not sane'
                        error_msg = ('event of class {} id {} at {}'.format(
                            event_name, event_id, thread_num) +
                                     ' has a unreasonable duration.\n')
                        event.update({'error_msg':error_msg})
                        # we still store error events, in the same format as a normal event
                        # we simply add an error message column, and put them in a different table.
                        # Users who wish to use these events can simply merge them with the events table.
                        builder.errors.append(event)

                dbp_iterator_delete(it_e)
                it_e = NULL

            else: # the event is not complete
                # this will change once singleton events are enabled.
                error_msg = 'event of class {} id {} at {} does not have a match.\n'.format(
                    event_name, event_id, thread_num)
                error = {'node_id':node_id, 'thread_id':thread_num, 'handle_id':handle_id,
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


# private utility class
class ProfileBuilder(object):
    def __init__(self):
        self.events = list()
        self.infos = list()
        self.nodes = list()
        self.errors = list()
        self.threads = list()
        self.event_types = dict()
        self.event_names = dict()
        self.information = dict()
        self.event_attributes = dict()
        self.event_convertors = dict()
        self.unordered_threads_by_node = dict()
        self.node_order = dict()


# NOTE:
# negative indexing breaks Cython, so don't do it.
# index = -1
# print('index is ' + str(index))
# print(builder.test_df[index])

def chunk(xs, n):
    ''' Splits a list of Xs into n roughly equally-sized lists.

    Useful for the naive Python map-reduce operation.
    '''
    ys = list(xs)
    ylen = len(ys)
    if ylen < 1:
        return []
    size = int(ylen / n)
    chunks = [ys[0+size*i : size*(i+1)] for i in xrange(n)]
    leftover = ylen - size*n
    edge = size*n
    for i in xrange(leftover):
        chunks[i%n].append(ys[edge+i])
    return chunks

from collections import namedtuple
import struct

cdef class ExtendedEvent:
    cdef object ev_struct
    cdef object aev
    cdef char* fmt
    cdef int event_len

    def __init__(self, event_name, event_conv, event_len):
        fmt = '@'
        tup = []
        for ev in str.split(event_conv, ':'):
            if 0 == len(ev):
                continue
            ev_list = str.split(ev, '{', 2)
            if len(ev_list) > 1:
                [ev_name, ev_type] = ev_list[:2]
                ev_type = ev_type.replace('}', '')
            else:
                ev_name = ev_list[0] if len(ev_list) == 1 else ''
                ev_type = ''
            if 0 == len(ev_name):
                continue
            ev_name = ev_name.replace(' ', '_')
            tup.append(ev_name)
            if ev_type == 'int32_t' or ev_type == 'int':
                fmt += 'i'
            elif ev_type == 'int64_t':
                fmt += 'l'
            elif ev_type == 'double':
                fmt += 'd'
            elif ev_type == 'float':
                fmt += 'f'
        self.aev = namedtuple(event_name, tup)
        #print('event[{0} = {1} fmt \'{2}\''.format(event_name, tup, fmt))
        self.ev_struct = struct.Struct(fmt)
        if event_len != len(self):
            print('Expected length differs from provided length for {0} extended event ({1} != {2})'.format(event_name, len(self), event_len))
            event_len = event_len if event_len < len(self) else len(self)
        self.event_len = event_len
    def __len__(self):
        return self.ev_struct.size
    def unpack(self, pybs):
        return self.aev._make(self.ev_struct.unpack(pybs))

# add parsing clauses to this function to get infos.
cdef parse_info(builder, event_type, char * cinfo):
    cdef bytes pybs;

    if None == builder.event_convertors[event_type]:
       return None
    event_info = None

    try:
        pybs = cinfo[:len(builder.event_convertors[event_type])]
        return builder.event_convertors[event_type].unpack(pybs)._asdict()
    except Exception as e:
        print(e)
        return None
