#!/usr/bin/env python

from __future__ import print_function
import os
from parsec_profiling import *
import numpy as np
import time

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
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
            if len(filenames) == 1 and '.h5-' in filenames[0]:
                print('First, we load the HDFed profile...')
                profile = ParsecProfile.from_hdf(filenames[0])
            else:
                print('First, we read the binary profile and put it in pandas format.')
                profile = ParsecProfile.from_native(filenames, 
                                                    translate=translate, 
                                                    print_progress=True)
        print('')
        print('First, let\'s print some basic information about the run.\n')

        print('Most PaRSEC profiles are profiles of testing executables, and')
        print('these runs tend to have some basic linear algebra attributes, such as matrix size.')
        print('If the profile contains these sorts of attributes, they will print below:\n')
        try:
            print('N: {} M: {} NB: {} MB: {} gflops: {} time elapsed: {}\n'.format(
                profile.N, profile.M, profile.NB, profile.MB, profile.gflops, profile.time_elapsed))
        except:
            print('It appears that one or more of the basic attributes was not present,')
            print('so we\'ll just move on.\n')

        print('The bulk of the profile information is stored in a data structure called a DataFrame.')
        print('A DataFrame is a large matrix/table with labeled columns.\n')
        print('One of our profile\'s DataFrames contains all of the "events".')
        print('Each event in our profile is one row in the events DataFrame,')
        print('and some events have different pieces of information than others.\n')
        print('Here, we print some information about the shared columns of the events.')
        with Timer() as t:
            print(profile.events[profile.basic_columns].describe())
        print('There are ' + str(len(profile.events)) + ' events in this profile', end=' ')
        print('and they took {} seconds to describe.'.format(t.interval))

        print('')
        print('The columns (or data labels) and their datatypes are:')
        print(profile.events.dtypes)

        print('')
        print('Now, we will select only the PINS_L12_EXEC events via a simple operation.')
        onlyexec = profile.events[:][ (profile.events['key'] == profile.event_types.PINS_L12_EXEC['key'])]
        print('Notice how the description of this subset is very different:')
        print(onlyexec[profile.basic_columns].describe())
        print('')
        print('Now, we will select only the exec events from thread 26.')
        print('We will also pick only certain pieces of the statistics to show, using the same')
        print('syntax that is used to pick rows out of any regular DataFrame.\n')
        onlyexec = onlyexec[:][onlyexec.thread == 26]
        print('Again, our view of the dataframe has changed:')
        print(onlyexec[profile.basic_columns].describe().loc['count':'std',:])
        print('')
        print('It is also possible to perform both operations in one query, like so:')
        onlyexec = profile.events[:][ (profile.events['key'] == profile.event_types.PINS_L12_EXEC['key']) 
                                  & (profile.events.thread == 26)]
        print('Note that the description is the same as for the previous subset.')
        print(onlyexec[profile.basic_columns].describe().loc['count':'std',:])
        print('')
        print('Now, a simple sort of EXEC events from thread 26 by duration, in ascending order.')
        with Timer() as t:
            srted = onlyexec.sort_index(by=['duration'], ascending=[True])
        print('That sort only took ' + str(t.interval) + ' seconds.')

        # with Timer() as join_t:
        #     srted = srted.merge(profile.info_df, how='outer', on='unique_id')
        # print('join took ' + str(join_t.interval))
    #    print(srted.iloc[10]['info'][4][0])
        print('To show that we\'ve sorted the events, we print the first five,')
        print('middle five, and last five events in the dataframe:')
        print(srted.loc[:,['duration', 'begin', 'end', 'id']].iloc[:5])
        print('')
        print(srted.loc[:,['duration', 'begin', 'end', 'id']].iloc[len(srted)/2-3:len(srted)/2+2])
        print('')
        print(srted.loc[:,['duration', 'begin', 'end', 'id']].iloc[-5:])
        print('')
        print('Up until now, we have only been looking at certain columns of the DataFrame.')
        print('But now we will show that some of these events also have profiling info embedded into them.\n')

        print('For the sorted EXEC events from thread 26, the following profiling info data are available:\n')

        print(srted[ ['PAPI_L1', 'PAPI_L2', 'kernel_type',
                      'vp_id', 'thread_id'] ].describe().loc['mean':'std',:])
        print('')
        # print(srted[:][srted['kernel_name'] == ''].describe())

        print('We can select events by index and access their data piece by piece if we want.')
        print('First, we cut these events down to only those with the kernel name "SYRK"\n')
        srted = srted[:][srted.kernel_type == profile.event_types.SYRK['key']]
        print('Now we print the L1 and L2 misses for this second item in this set of events:\n')
        print('sorted SYRK execs, index 10, kernel type: ' + str(srted.iloc[1]['kernel_type']))
        print('sorted SYRK execs, index 10, L1 misses: ' + str(srted.iloc[1]['PAPI_L1']))
        print('sorted SYRK execs, index 10, L2 misses: ' + str(srted.iloc[1]['PAPI_L2']))
        print('')

        if not profile.store:
            print('Lastly, we want to show that the file can be written out')
            print('into a standard format. Our choice is HDF5, because it is fast.')
            with Timer() as t:
                store = profile.to_hdf(profile.store_name.replace('.prof-', '.h5-'))
                store.close()
            print('Storing the entire profile to HDF took only {} seconds.'.format(t.interval))

    print('Overall test took {:.2f} seconds'.format(main.interval))
    
if __name__ == '__main__':
    import sys

    do_demo(sys.argv[1:], translate=False)

