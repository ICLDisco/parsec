#!/usr/bin/env python

from __future__ import print_function
from parsec_profiling import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_prefs
import binprof_utils as p3_utils
import os, sys
import itertools

y_axis = 'PAPI_L3'
lo_cut = 00
hi_cut = 100
ext = 'png'

event_types = ['PAPI_CORE_EXEC']
event_subtypes = ['GEMM']

def plot_Y_vs_duration(profiles, y_axis, main_type, subtype=None, shared_name='',
                       hi_cut=hi_cut, lo_cut=lo_cut, ext=ext):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    profiles.sort(key=lambda x: x.gflops)
    if subtype:
        type_name = main_type + '-' + subtype
    else:
        type_name = main_type

    for profile in profiles:
        if subtype:
            chosen_events = profile.events[:][(profile.event_types[main_type] == profile.events.type) &
                                          (profile.events.kernel_type == profile.event_types[subtype])]
        else:
            chosen_events = profile.events[:][(profile.event_types[main_type] == profile.events.type)]
        chosen_events = chosen_events.sort('duration')
        chosen_events = chosen_events[int(len(chosen_events) * lo_cut * 0.01):
                              int(len(chosen_events) * hi_cut * 0.01)]

        label = '{}: {:.1f} gflops/s'.format(profile.sched.upper(),
                                             profile.gflops)

        ax.plot(chosen_events['duration'], chosen_events[y_axis], linestyle='', marker='.',
                color=mpl_prefs.sched_colors[profile.sched.upper()],
                label=label)

        ax.set_title(y_axis + ' vs Duration of ' + type_name +
                     ' Tasks, By Scheduler\nfor {} where '.format(profile.exe) +
                     'N = {}, NB = {}, IB = {}, on {}'.format(profile.N, profile.NB,
                                                              profile.IB, profile.hostname))

    if not ax.has_data():
        print('Plot has no data.')
        return
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_ylabel(y_axis)
    ax.set_xlabel('duration of {} kernels, '.format(type_name) +
                  ', excl. below {}% & above {}%'.format(lo_cut, hi_cut))
    fig.set_size_inches(10, 5)
    fig.set_dpi(300)
    fig.savefig(shared_name.replace(' ', '_') + '_' + type_name +
                '{}_vs_dur_{}-{}_scatter.{}'.format(y-axis, lo_cut, hi_cut, ext),
                bbox_inches='tight')

if __name__ == '__main__':
    filenames = []
    for index, arg in enumerate(sys.argv[1:]):
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            if arg == '--help':
                print_help()
                sys.exit(0)
            elif arg.startswith('--event-types='):
                event_types = arg.replace('--event-types=', '').split(',')
            elif arg.startswith('--event-subtypes='):
                event_subtypes = arg.replace('--event-subtypes=', '').split(',')
            elif arg.startswith('--y-axis='):
                y_axis = arg.replace('--y-axis=', '')
            elif arg.startswith('--cut='):
                cuts = arg.replace('--cut=', '').split(',')
                lo_cut = float(cuts[0])
                hi_cut = float(cuts[1])
            else:
                event_subtypes.append(arg)

    profiles = p3_utils.autoload_profiles(filenames, convert=True, unlink=False)
    profile_sets = find_profile_sets(profiles)
    profiles = automerge_profile_sets(profile_sets.values())
    profile_sets = find_profile_sets(profiles, on=[ 'exe', 'N', 'NB' ])

    for set_name, profiles in profile_sets.iteritems():
        if len(event_subtypes) > 0:
            type_pairs = list(itertools.product(event_types, event_subtypes))
        else:
            event_subtypes = mpl_prefs.kernel_names[profiles[0].exe.replace('dplasma/testing/testing_', '')]
            type_pairs = list(itertools.product(event_types, event_subtypes))

        print('Now graphing the Y-axis datum \'{}\' against the '.format(y_axis) +
              ' event type pairs {}, where they can be found in the profile.'.format(type_pairs))

        for type_pair in type_pairs:
            if len(type_pair) == 2: # it's a tuple
                main_type = type_pair[0]
                subtype = type_pair[1]
            else:
                main_type = type_pair
                subtype = None

            plot_Y_vs_duration(profiles, y_axis, main_type, subtype=subtype,
                               hi_cut = hi_cut, lo_cut = lo_cut,
                               shared_name=set_name)
        # for event_type in event_types:
        #     plot_Y_vs_duration(profiles, event_type, shared_name=set_name)


