#!/usr/bin/env python

import py_dbpreader as dbpr
import cPickle
import os
from timer import Timer
from pandas import *
import numpy as np

def safe_unlink(files):
    for ufile in files:
        try:
            os.unlink(ufile) # no need to have them hanging around anymore
        except OSError:
            print('the file {} has apparently vanished.'.format(ufile))

def read_pickle_return(filenames, outfilename = None, delete=True):
    profile = None
    with Timer() as t:
        profile = dbpr.readProfile(filenames)
    print('Profile parse took ' + str(t.interval))
    print('')
    print('The DataFrame is a large matrix/table of the profile information')
    print('Here, we print some information about the standard (non-info-struct) pieces of the events.')
    print(profile.df[profile.event_columns].describe())
    print('There are ' + str(len(profile.df)) + ' events in this profile.')
    print('')
    print('Now, we will select only the PINS_L12_EXEC events via a simple operation.')
    onlyexec = profile.df[:][ (profile.df['key'] == profile.event_types['PINS_L12_EXEC'].key)]
    print('Notice how the description of this subset is very different:')
    print(onlyexec[profile.event_columns].describe())
    print('')
    print('Now, we will select only the exec events from thread 26.')
    onlyexec = profile.df[:][(profile.df['thread'] == 26)]
    print('Again, our view of the dataframe has changed:')
    print(onlyexec[profile.event_columns].describe())
    print('')
    print('It is also possible to perform both operations in one query, like so:')
    onlyexec = profile.df[:][ (profile.df['key'] == profile.event_types['PINS_L12_EXEC'].key) 
                              & (profile.df['thread'] == 26)]
    print('Note that the description is the same as for the previous subset.')
    print(onlyexec[profile.event_columns].describe())
    print('')
    print('Now, a simple sort of EXEC events from thread 26 by duration, in ascending order.')
    with Timer() as t:
        srted = onlyexec.sort_index(by=['duration'], ascending=[True])
    print('That sort only took ' + str(t.interval) + ' seconds.')

    # with Timer() as join_t:
    #     srted = srted.merge(profile.info_df, how='outer', on='unique_id')
    # print('join took ' + str(join_t.interval))
#    print(srted.iloc[10]['info'][4][0])
    print('To show that we\'ve sorted the events, we print the first ten events in the dataframe:')
    print(srted[profile.event_columns].describe())
    print('')
    print('Up until now, we have only been looking at certain columns of the DataFrame. But ' +
          'now we will show that some of these events also have profiling info embedded into them.')
    print('For the sorted EXEC events from thread 26, the following profiling info data are available:')
    print(srted[ ['PAPI_L1', 'PAPI_L2', 'kernel_type',
                  'kernel_name', 'vp_id', 'thread_id'] ].describe())
    print('')
    print('We can select events by index and access their data piece by piece if we want:')
    print('sorted execs, index 10, L1 misses: ' + str(srted.iloc[10]['PAPI_L1']))
    print('sorted execs, index 10, kernel name: ' + str(srted.iloc[10]['kernel_name']))
    print('')
#    profile.info_df['kernel_name'] = profile.info_df['kernel_name'].apply(lambda x: x.decode('utf-8'))
    
    if not outfilename:
        outfilename = filenames[0].replace('.profile', '') + '.pickle'
        pickle_name = outfilename
        hdf5_name = outfilename.replace('.pickle', '.h5')
        hdf5_info_name = outfilename.replace('.pickle', '_info.h5')

    # print(profile.info_df.dtypes) # this tells us if we'll be compatible with pure HDF5
    # # profile.df = profile.df.convert_objects() # this appears to be non-functional
    # profile.info_df = profile.info_df.convert_objects()
    
    # write dataframe to file as a test of speed
    print('')
    print('Part of the appeal of the pandas library is its speed.')
    print('We will now write the entire dataframe to disk in the HDF5 format.')
    with Timer() as t:
        # profile.df.to_hdf(hdf5_name, 'table', append=True) # this version is desired
        profile.df.to_hdf(hdf5_name, 'profile', mode='w')
        # profile.info_df.to_hdf(hdf5_info_name, 'info', mode='w')
    print('HDFing the DataFrame took ' + str(t.interval) + ' seconds.')

    print('')
    print('We can read the Dataframe back in (and even generate statistics on it)')
    print('in a very short amount of time also:')
    # read it back in for speed test
    with Timer() as t:
        # dataframe = read_hdf(hdf5_name, 'profile', where=['thread=26']) # this is the 'table' version
        dataframe = read_hdf(hdf5_name, 'profile')
        print(dataframe[profile.event_columns].describe())
        # info = read_hdf(hdf5_info_name, 'info')
    print('Reading and describing the HDF5 Dataframe took only ' + str(t.interval) + ' seconds.')

    # pickle for speed/size comparison
    # outfile = open(pickle_name, 'w')
    # with Timer() as t:
    #     cPickle.dump(profile, outfile)
    # print('Pickling took ' + str(t.interval))

    if delete:
        safe_unlink(filenames)
    
if __name__ == '__main__':
    import sys
    read_pickle_return(sys.argv[1:], delete=False)