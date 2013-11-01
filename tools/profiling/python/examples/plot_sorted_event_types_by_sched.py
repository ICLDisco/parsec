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
event_types = ['PAPI_L12_EXEC']
event_subtypes = []
div_by = None
hi_cut = 99
lo_cut = 01
do_all = False

def print_help():
    print('')
    print(' This script was originally designed to perform hockey-stick-style plots')
    print(' of PaRSEC profiles with papi_L123 events. It now supports more general plots.')
    print(' The script plots the Y axis datum against the events, sorted by the Y axis datum.')
    print('')
    print(' It will accept sets of profiles as well, and will attempt to merge them if encountered.')
    print(' usage: <script_name> [PROFILE FILENAMES] [--event-types=TYPE1,TYPE2] [--event-subtypes=TYPE1,TYPE2] [--y-axis=Y_AXIS_DATUM]')
    print('')
    print(' --event-types    : Filters by event major type, e.g. GEMM, POTRF, PAPI_L12_EXEC')
    print(' --y-axis         : Y axis datum, e.g. duration, begin, end, PAPI_L2')
    print(' --event-subtypes : Filters by PAPI_L12 event kernel type, e.g. GEMM, POTRF, SYRK')
    print('')

def plot_profiles(profiles, main_type, subtype=None, shared_name=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # sort for the legend order
    profiles.sort(key=lambda x: x.gflops)

    if subtype:
        type_name = main_type + '-' + subtype
    else:
        type_name = main_type

    for profile in profiles:
        extra_descrip = ''
        if div_by:
            extra_descrip = ' divided by ' + div_by

        if subtype:
            events = profile.events[:][(profile.events.type == profile.event_types[main_type]) &
                                       (profile.events.kernel_type == profile.event_types[subtype])]
        else:
            events = profile.events[:][(profile.events.type == profile.event_types[main_type])]

        if len(events) == 0:
            print('skipping profile \'{}\' with type pair {}'.format(profile.descrip(), type_pair) +
                  'because it has no events of the selected type.')
            continue
        if y_axis not in events:
            print('skipping profile {} with type pair {}'.format(profile.descrip(), type_pair) +
                  'because it has no events of the selected type with the Y-axis variable {}.'.format(y_axis))
            continue

        sorted_events = events.sort(y_axis)

        # cut down to ignore noise
        # -- can do better than this
        sorted_events = sorted_events[int(len(sorted_events)*lo_cut * 0.01):
                                      int(len(sorted_events)*hi_cut * 0.01)]
        label = '{}: {} gflops/s'.format(profile.sched.upper(),
                                         int(profile.gflops))
        ax.plot(xrange(len(sorted_events)),
                sorted_events[y_axis],
                color=mpl_prefs.sched_colors[profile.sched.upper()],
                label=label
            )
        print(label)

        ax.set_title(shared_name.replace('_', ' ') + ' ' + type_name +
                     ' tasks, sorted by ' + y_axis + extra_descrip + ', by scheduler\n' +
                     'N = {}, NB = {}, IB = {}, on {}'.format(profile.N, profile.NB,
                                                              profile.IB, profile.hostname))
        ax.set_ylim(sorted_events.iloc[0][y_axis],
                    sorted_events.iloc[-1][y_axis])
    if not ax.has_data():
        print('This graph is empty, so it will not be created.')
        return -1
    ax.grid(True)
    ax.set_xlabel('{} kernels, sorted by '.format(type_name)
                  + y_axis + extra_descrip +
                  ', excl. below {}% & above {}%'.format(lo_cut, hi_cut))
    ax.set_ylabel(y_axis)
    ax.legend(loc='best', title='SCHED: perf')
    fig.set_size_inches(10, 5)
    fig.set_dpi(300)
    # TODO need a better naming scheme for these files...
    fig.savefig(shared_name.replace(' ', '_') + '_' +
                type_name + '_' +
                y_axis + '_{}-{}_'.format(lo_cut, hi_cut) +
                'by_sched_hockey_stick.pdf', bbox_inches='tight')

if __name__ == '__main__':
    filenames = []
    name_tokens = []
    for index, arg in enumerate(sys.argv[1:]):
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            if arg == '--help':
                print_help()
                sys.exit(0)
            elif arg.startswith('--div-by='):
                div_by = arg.replace('--div-by=', '')
                name_tokens.append('div_by_' + div_by)
            elif arg.startswith('--event-types='):
                event_types = arg.replace('--event-types=', '').split(',')
            elif arg.startswith('--y-axis='):
                y_axis = arg.replace('--y-axis=', '')
            elif arg.startswith('--event-subtypes='):
                event_subtypes = arg.replace('--event-subtypes=', '').split(',')
            elif arg.startswith('--cut='):
                cuts = arg.replace('--cut=', '').split(',')
                lo_cut = float(cuts[0])
                hi_cut = float(cuts[1])
            elif arg == '--do-all':
                do_all = True
            else:
                event_subtypes.append(arg)

    # then, we group the profiles appropriately...
    profiles = p3_g.autoload_profiles(filenames, convert=True, unlink=False)
    profile_sets = find_profile_sets(profiles)
    for pset in profile_sets.values()[1:]:
        if len(pset) != len(profile_sets.values()[0]):
            print('A profile set has a different length than the first set,')
            print('which may cause your graph to look strange.')
    profiles = automerge_profile_sets(profile_sets.values())
    profile_sets = find_profile_sets(profiles, on=[ 'exe', 'N', 'NB' ])

    for set_name, profiles in profile_sets.iteritems():
        # first we pair up the selectors, if subtypes were specified...
        if do_all:
            if len(event_types) > 0:
                event_subtypes = mpl_prefs.kernel_names[profiles[0].exe.replace('testing_', '')]
            else:
                event_types = mpl_prefs.kernel_names[profiles[0].exe.replace('testing_', '')]
        if len(event_subtypes) > 0:
            type_pairs = list(itertools.product(event_types, event_subtypes))
        else:
            type_pairs = [(event_type) for event_type in event_types]
        print('Now graphing the Y-axis datum \'{}\' against the '.format(y_axis) +
              ' event type pairs {}, where they can be found in the profile.'.format(type_pairs))

        for type_pair in type_pairs:
            if len(type_pair) == 2: # it's a tuple
                main_type = type_pair[0]
                subtype = type_pair[1]
            else:
                main_type = type_pair
                subtype = None

            plot_profiles(profiles, main_type, subtype=subtype, shared_name = set_name)
