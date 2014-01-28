#!/usr/bin/env python

from __future__ import print_function
import os
import parsec_trace_tables as ptt
import pbt2ptt
import papi_core_utils
import numpy as np
import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

# profiling stuff
# import pstats, cProfile
# import pyximport
# pyximport.install()

def safe_unlink(files):
    for ufile in files:
        try:
            os.unlink(ufile) # no need to have them hanging around anymore
        except OSError:
            print('the file {} has apparently vanished.'.format(ufile))

def do_demo(filenames, translate=False):
    with Timer() as main:
        profile = None

        with Timer() as t:
            if len(filenames) == 1 and ('.h5-' in filenames[0] or 'hdf5' in filenames[0]):
                print('First, we load the HDFed profile...')
            else:
                print('First, we read the binary profile and convert it to pandas format.')
                filenames[0] = pbt2ptt.convert(filenames, report_progress=True, unlink=False,
                                           multiprocess=True)
                print('Then, we read the HDFed profile...')
            profile = ptt.ParsecTraceTables.from_hdf(filenames[0])

        print('The load took {} seconds.'.format(t.interval))
        print('')
        print('First, let\'s print some basic information about the run.\n')

        print('Most PaRSEC profiles are profiles of testing executables, and')
        print('these runs tend to have some basic linear algebra attributes, such as matrix size.')
        print('If the profile contains these sorts of attributes, they will print below:\n')
        try:
            print('N: {} M: {} NB: {} MB: {} gflops: {} time elapsed: {} scheduler: {}\n'.format(
                profile.N, profile.M, profile.NB, profile.MB, profile.gflops, profile.time_elapsed, profile.sched))
        except AttributeError as e:
            print(e)
            print('It appears that one or more of the basic attributes was not present,')
            print('so we\'ll just move on.\n')

        print('The bulk of the profile information is stored in a data structure called a DataFrame.')
        print('A DataFrame is a large matrix/table with labeled columns.\n')
        print('One of our profile\'s DataFrames contains all of the "events".')
        print('Each event in our profile is one row in the events DataFrame,')
        print('and some events have different pieces of information than others.\n')

        print('The columns of the DataFrame (or data labels) and their datatypes are:')
        print(profile.events.dtypes)

        print('')
        print('Now, we can print some statistics about the *shared* columns of the events.')
        with Timer() as t:
            print(profile.events[profile.basic_columns].describe())
        print('There are ' + str(len(profile.events)) + ' events in this profile', end=' ')
        print('and they took {} seconds to describe.'.format(t.interval))

        print('')

        clevel = None
        if clevel:
            # print('Testing re-store as compressed HDF5')
            # with Timer() as t:
            #     profile.events.to_hdf('test_events.hdf5', 'events', complevel=clevel, complib='blosc')
            # print('took {} to write only the events to HDF5, compression level {}\n'.format(t.interval,clevel))
            print('Testing re-store as table HDF5')
            with Timer() as t:
                profile.to_hdf('test_table.hdf5', table=True, append=False)
            print('took {} to write the HDF5 table\n'.format(t.interval))
            print('Testing re-store as Storer HDF5')
            with Timer() as t:
                profile.to_hdf('test_storer.hdf5', table=False, append=False)
            print('took {} to write the HDF5 storer\n'.format(t.interval))
            # with Timer() as t:
            #     new_store = pd.HDFStore('test_events.hdf5_store', 'w', complevel=clevel, complib='blosc')
            #     new_store.put('events', profile.events, table=False)
            # print('took {} to PUT only the events to HDF5, compression level {}\n'.format(t.interval,clevel))
            # with Timer() as t:
            #     profile.events = pd.read_hdf('test_events.hdf5', 'events')
            #     print(profile.events[profile.basic_columns].describe())
            # print('There are ' + str(len(profile.events)) + ' events in this profile',
            #       'and they took {} seconds to read & describe.'.format(t.interval))
            with Timer() as t:
                profile = ParsecTraceTables.from_hdf('test_table.hdf5')
            print('There are ' + str(len(profile.events)) + ' events in this profile',
                  'and they took {} seconds to read'.format(t.interval))
            with Timer() as t:
                print(profile.events[profile.basic_columns].describe())
            print(' and {} seconds to describe.'.format(t.interval))
            # with Timer() as t:
            #     profile.to_hdf('test.hdf5', complevel=clevel)
            # print('took {} to write entire file to HDF5, compression level {}'.format(t.interval,clevel))

            # print ('Testing write to CSV...')
            # with Timer() as t:
            #     profile.events.to_csv('test.csv', complevel=clevel, complib='blosc' )
            # print('took {} to write to csv, clevel {}'.format(t.interval, clevel))

        print('Now, we will select only the PAPI_CORE_EXEC* events via a simple operation.')
        for event_name in profile.event_types.keys():
            if event_name.startswith('PAPI_CORE_EXEC'):
                break
        onlyexec = profile.events[:][ profile.events['type'] == profile.event_types[event_name] ]
        print('Notice how the description of this subset is very different:')
        print(onlyexec[profile.basic_columns].describe())
        print('')
        print('Now, we will select only the exec events from thread 7.')
        print('We will also pick only certain pieces of the statistics to show, using the same')
        print('syntax that is used to pick rows out of any regular DataFrame.\n')
        onlyexec = onlyexec[:][onlyexec.thread_id == 7]
        print('Again, our view of the dataframe has changed:')
        print(onlyexec[profile.basic_columns].describe()[:]['count':'std'])
        print('')
        print('It is also possible to perform both operations in one query, like so:')
        onlyexec = profile.events[:][(profile.events['type'] == profile.event_types[event_name]) &
                                     (profile.events.thread_id == 7) ]
        print('Note that the description is the same as for the previous subset.')
        print(onlyexec[profile.basic_columns].describe()[:]['count':'std'])
        print('')
        print('Now, a simple sort of EXEC events from thread 7 by duration, in ascending order.')
        with Timer() as t:
            srted = onlyexec.sort_index(by=['duration'], ascending=[True])
        print('That sort only took ' + str(t.interval) + ' seconds.')

        pce_vals = papi_core_utils.PAPICoreEventValueLabelGetter()

        print('To show that we\'ve sorted the events, we print the first five,')
        print('middle five, and last five events in the dataframe:')
        print(srted.loc[:,['duration', 'begin', 'end', 'id'] +
                        pce_vals[event_name] ].iloc[:5])
        print('')
        print(srted.loc[:,['duration', 'begin', 'end', 'id'] +
                        pce_vals[event_name] ].iloc[len(srted)/2-3:len(srted)/2+2])
        print('')
        print(srted.loc[:,['duration', 'begin', 'end', 'id'] +
                        pce_vals[event_name] ].iloc[-5:])
        print('')
        print('Up until now, we have only been',
              'looking at certain columns of the DataFrame.')
        print('But now we will show that some of these events',
              'also have profiling info embedded into them.\n')

        print('For the sorted EXEC events from thread 7,',
              'the following profiling info data are available:\n')

        print(srted)

        print(srted[ pce_vals[event_name] + ['kernel_type', 'thread_id']
                 ].describe().loc['mean':'std',:])
        print('')

        print('We can select events by index and access their data piece by piece if we want.')
        print('First, we cut these events down to only those with the kernel name "SYRK"\n')
        srted = srted[:][srted.kernel_type == profile.event_types['SYRK']]
        pce_names = pce_vals[event_name]
        print('Now we print L1 and L2 misses for this second item in this set of events:\n')
        print('sorted SYRK execs, index 1, kernel type: ' +
              str(srted.iloc[1]['kernel_type']))
        print('sorted SYRK execs, index 1, {}: '.format(
            pce_names[0]) + str(srted.iloc[1][pce_names[0]]))
        print('sorted SYRK execs, index 1, {}: '.format(
            pce_names[1]) + str(srted.iloc[1][pce_names[1]]))
        print('')

    print('Overall demo took {:.2f} seconds'.format(main.interval))

if __name__ == '__main__':
    import sys

    do_demo(sys.argv[1:], translate=False)

