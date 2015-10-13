#!/usr/bin/env python

from __future__ import print_function
import os
import parsec_trace_tables as ptt
import pbt2ptt
import numpy as np
import time
import pandas

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def safe_unlink(files):
    for ufile in files:
        try:
            os.unlink(ufile) # no need to have them hanging around anymore
        except OSError:
            print('the file {} has apparently vanished.'.format(ufile))

def do_demo(filenames, translate=False):
    with Timer() as main:
        trace = None

        with Timer() as t:
            if len(filenames) == 1 and (ptt.is_ptt(filenames[0])):
                print('First, we load the HDFed trace...')
            else:
                print('First, we read the binary trace and convert it to the ParSEC Trace Tables format.')
                filenames[0] = pbt2ptt.convert(filenames, report_progress=True)
                print('Then, we read the HDFed trace...')
            trace = ptt.from_hdf(filenames[0])

        print('The load took {} seconds.'.format(t.interval))
        print('')
        print('First, let\'s print some basic information about the run.\n')

        print('Most PaRSEC traces are traces of testing executables, and')
        print('these runs tend to have some basic linear algebra attributes, such as matrix size.')
        print('If the trace contains these sorts of attributes, they will print below:\n')
        try:
            print('N: {} M: {} NB: {} MB: {} gflops: {} time elapsed: {} scheduler: {}\n'.format(
                trace.N, trace.M, trace.NB, trace.MB, trace.gflops, trace.time_elapsed, trace.sched))
        except AttributeError as e:
            print(e)
            print('It appears that one or more of the basic attributes was not present,')
            print('so we\'ll just move on.\n')

        print('The bulk of the trace information is stored in a data structure called a DataFrame.')
        print('A DataFrame is a large matrix/table with labeled columns.\n')
        print('One of our trace\'s DataFrames contains all of the "events".')
        print('Each event in our trace is one row in the events DataFrame,')
        print('and some events have different pieces of information than others.\n')

        print('The columns of the DataFrame (or data labels) and their datatypes are:')
        print(trace.events.dtypes)

        print('')
        print('Now, we can print some statistics about the *shared* columns of the events.')
        print('###################################################\n')
        # Note trace.events.loc[:,'begin':] returns the information from columns 'begin' to the last column
        with Timer() as t:
            print(trace.events.loc[:,'begin':])
        print('There are ' + str(len(trace.events)) + ' events in this trace', end=' ')
        print('and they took {} seconds to describe.'.format(t.interval))
        print('###################################################\n\n')
        print('')

        print('')
        user_columns = []
        for column_name in trace.events.columns.values:
            if column_name not in trace.events.loc[:,'begin':]:
                user_columns.append(column_name)
        print('Here are some statistics on the unique, non-shared columns:')
        print('###################################################\n')
        with Timer() as t:
            print(trace.events[user_columns].describe())
        print('There are ' + str(len(trace.events)) + ' events in this trace', end=' ')
        print('and they took {} seconds to describe.'.format(t.interval))
        print('###################################################\n\n')
        print('')

        # Set this to a number to specify the compression level (e.x. 1)
        clevel = None
        if clevel:
            print('Compression Test:')
            print('###################################################\n')

            print('Testing re-store of events as a compressed HDF5')
            with Timer() as t:
                trace.events.to_hdf('test_compressed_events.hdf5', 'events', complevel=clevel, complib='blosc')
            print('took {} to write only the events to HDF5, compression level {}\n'.format(t.interval,clevel))
            print('')

            print('Testing re-store as a Table HDF5')
            with Timer() as t:
                trace.to_hdf('test_table.hdf5', table=True, append=False)
            print('took {} to write the HDF5 table\n'.format(t.interval))
            print('')

            print('Testing re-store as Storer HDF5')
            with Timer() as t:
                trace.to_hdf('test_storer.hdf5', table=False, append=False)
            print('took {} to write the HDF5 storer\n'.format(t.interval))
            print('')

            print('Testing re-store as HDF5_Store')
            with Timer() as t:
                new_store = pandas.HDFStore('test_events.hdf5_store', 'w', complevel=clevel, complib='blosc')
                new_store.put('events', trace.events, table=False)
            print('took {} to PUT only the events to HDF5, compression level {}\n'.format(t.interval,clevel))
            print('')

            print('Testing read from compressed HDF5')
            with Timer() as t:
                trace.events = pandas.read_hdf('test_compressed_events.hdf5', 'events')
                print(trace.events[trace.events.loc[:,'begin':]].describe())
            print('There are ' + str(len(trace.events)) + ' events in this trace',
                  'and they took {} seconds to read & describe.'.format(t.interval))
            print('')

            print('Testing read from Table HDF5')
            with Timer() as t:
                trace = ptt.from_hdf('test_table.hdf5')
            print('There are ' + str(len(trace.events)) + ' events in this trace',
                  'and they took {} seconds to read'.format(t.interval))
            print('')

            print ('Testing write to CSV (with compression)...')
            with Timer() as t:
                trace.events.to_csv('test.csv', complevel=clevel, complib='blosc' )
            print('took {} to write to csv, clevel {}'.format(t.interval, clevel))
            print('###################################################\n\n')

        print('Now, we will select only the PINS_PAPI* events via a simple operation (only the first will be printed).')
        print('###################################################\n')
        for event_name in trace.event_types.keys():
            if event_name.startswith('PINS_PAPI'):
                print('\nFound: ' + event_name)
                print('---------------------------------------------------\n')
                onlyexec = trace.events[:][ trace.events['type'] == trace.event_types[event_name] ]
                print(onlyexec.describe())
                break # Removing this break will print all of the PINS_PAPI* events' descriptions
        print('###################################################\n\n')
        onlyexec = trace.events[:][ trace.events['type'] == trace.event_types[event_name] ]

        print('')
        print('Now, we will select only the {} events from thread 0.'.format(event_name))
        print('We will also pick only certain pieces of the statistics to show, using the same')
        print('syntax that is used to pick rows out of any regular DataFrame.\n')
        onlyexec = onlyexec[:][onlyexec.stream_id == 0]

        if len(onlyexec) == 0:
            print('Unfortunately the {} event doesn\'t have any events for thread 0,'.format(event_name))
            print('so the following outputs will be rather dull...\n')

        print('Again, our view of the dataframe has changed:')
        print('###################################################\n')
        print(onlyexec.describe()[:]['count':'std'])
        print('###################################################\n\n')
        print('')
        print('It is also possible to perform both operations in one query, like so:')
        onlyexec = trace.events[:][(trace.events['type'] == trace.event_types[event_name]) &
                                     (trace.events.stream_id == 0) ]
        print('Note that the description is the same as for the previous subset.')
        print('###################################################\n')
        print(onlyexec.describe()[:]['count':'std'])
        print('###################################################\n\n')
        print('')

        print('Now, a simple sort of {} events from thread 0 by duration, in ascending order.'.format(event_name))
        with Timer() as t:
            onlyexec['duration'] = pandas.Series(onlyexec['end'] - onlyexec['begin'])
            srted = onlyexec.sort_index(by=['duration'], ascending=[True])
        print('That sort only took ' + str(t.interval) + ' seconds.')
        print('Here is the sorted list:')
        print('###################################################\n')
        print(srted)
        print('###################################################\n\n')

    print('Overall demo took {:.2f} seconds'.format(main.interval))

if __name__ == '__main__':
    import sys

    do_demo(sys.argv[1:], translate=False)
