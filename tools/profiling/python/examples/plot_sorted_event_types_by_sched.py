#!/usr/bin/env python

from __future__ import print_function
from parsec_profiling import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_prefs
import p3_group_profiles as p3_g
import os, sys
import itertools

# defaults
y_axis = 'PAPI_L2'
event_types = ['PINS_L12_EXEC']
event_subtypes = []
div_by = None

print('This script is designed to operate on PaRSEC profiles with papi_L123 PINS_L12_EXEC module events.')

if __name__ == '__main__':
    filenames = []
    name_tokens = []
    for index, arg in enumerate(sys.argv[1:]):
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            if arg.startswith('--div-by='):
                div_by = arg.replace('--div-by=', '')
                name_tokens.append('div_by_' + div_by)
            elif arg.startswith('--event-types='):
                event_types = arg.replace('--event-type=', '').split(',')
            elif arg.startswith('--yaxis='):
                y_axis = arg.replace('--yaxis=', '')
            else:
                event_subtypes.append(arg)

    # first, we group the profiles appropriately...
    profiles = p3_g.autoload_profiles(filenames, convert=True, unlink=False)
    profile_sets = find_profile_sets(profiles)
    for pset in profile_sets[1:]:
        if len(pset) != len(profile_sets[0]):
            print('A profile set has a different length than the first set,')
            print('which may cause your graph to look strange.')
    profiles = automerge_profile_sets(profile_sets)

    # then we pair up the selectors, if subtypes were specified...
    if len(event_subtypes) > 0:
        type_pairs = list(itertools.product(event_types, event_subtypes))
    else:
        type_pairs = [(event_type) for event_type in event_types]
    print(type_pairs)

    for type_pair in type_pairs:
        if len(type_pair) == 2: # it's a tuple
            main_type = type_pair[0]
            subtype = type_pair[1]
            type_name = main_type + '-' + subtype
        else:
            main_type = type_pair
            subtype = None
            type_name = main_type

        profile_name = ''
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for profile in profiles:
            profile_name = profile.exe.replace('./testing_', '').lower()
            extra_descrip = ''
            if div_by:
                extra_descrip = ' divided by ' + div_by

            if subtype:
                events = profile.events[:][(profile.events['type'] == profile.event_types[main_type]) &
                                           (profile.events['kernel_type'] == profile.event_types[subtype])]
            else:
                events = profile.events[:][(profile.events['type'] == profile.event_types[main_type])]

            sorted_events = events.sort(y_axis)

            # cut down to ignore noise
            sorted_events = sorted_events[int(len(sorted_events)*0.01):
                                          int(len(sorted_events)*0.99)]
            label = '{}: {} gflops/s'.format(profile.sched.upper(),
                                             int(profile.gflops))
            ax.plot(xrange(len(sorted_events)),
                    sorted_events[y_axis],
                    color=mpl_prefs.sched_colors[profile.sched.upper()],
                    label=label
                )
            print(label)

            ax.set_title(profile_name + ' ' + type_name +
                         ' tasks, sorted by ' + y_axis + extra_descrip + ', by scheduler\n' +
                         'N = {}, NB = {}, IB = {}, on {}'.format(profile.N, profile.NB,
                                                                  profile.IB, profile.hostname))
        ax.set_ylim(sorted_events.iloc[0][y_axis], sorted_events.iloc[-1][y_axis])
        ax.grid(True)
        ax.set_xlabel('{} kernels, sorted by '.format(type_name)
                      + y_axis + extra_descrip +
                      ', excluding top & bottom 1%')
        ax.set_ylabel(y_axis)
        ax.legend(loc='best', title='SCHED: perf')
        fig.set_size_inches(10, 5)
        fig.set_dpi(200)
        # TODO need a better naming scheme for these files...
        extra_name = '_'
        for name_token in name_tokens:
            extra_name += str(name_token) + '_'
        fig.savefig(profile_name + '_' +
                    type_name + '_' +
                    y_axis + extra_name +
                    'by_sched_hockey_stick.png', bbox_inches='tight')
