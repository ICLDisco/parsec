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

# defaults
y_axis = 'PAPI_L3'
x_axis = 'duration'
lo_cut = 00
hi_cut = 100
ext = 'png'
event_types = ['PAPI_CORE_EXEC']
event_subtypes = ['GEMM']

def plot_Y_vs_duration(profiles, x_axis, y_axis, main_type,
                       subtype=None, shared_name='',
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
        chosen_events = chosen_events.sort(x_axis)
        chosen_events = chosen_events[int(len(chosen_events) * lo_cut * 0.01):
                              int(len(chosen_events) * hi_cut * 0.01)]

        label = '{}: {:.1f} gflops/s'.format(profile.sched.upper(),
                                             profile.gflops)

        ax.plot(chosen_events[x_axis], chosen_events[y_axis], linestyle='', marker='.',
                color=mpl_prefs.sched_colors[profile.sched.upper()],
                label=label)

        ax.set_title('{} vs {} of {}'.format(y_axis, x_axis, type_name) +
                     ' Tasks, By Scheduler\nfor {} where '.format(profile.exe) +
                     'N = {}, NB = {}, IB = {}, on {}'.format(profile.N, profile.NB,
                                                              profile.IB, profile.hostname))

    if not ax.has_data():
        print('Plot has no data.')
        return None
    ax.legend(loc='best', title='SCHED: perf')
    ax.grid(True)
    ax.set_xlim(0, ax.get_xlim()[1]) # start from zero for scale
    ax.set_ylabel(y_axis)
    cut_label = ''
    if hi_cut < 100 or lo_cut > 0:
        cut_label = ', excl. below {}% & above {}%'.format(lo_cut, hi_cut)
    ax.set_xlabel('{} of {} kernels'.format(x_axis, type_name) + cut_label)
    fig.set_size_inches(10, 5)
    fig.set_dpi(300)
    fig.savefig(shared_name.replace(' ', '_') + '_' + type_name +
                '_{}_vs_{}_{}-{}_scatter.{}'.format(y_axis, x_axis, lo_cut, hi_cut, ext),
                bbox_inches='tight')

def print_help():
    print('')
    print(' The script plots the selected Y axis datum against the X axis datum.')
    print('')
    print(' It will accept sets of profiles as well, and will attempt to merge them if encountered.')
    print(' usage: <script_name> [PROFILE FILENAMES] [--event-types=TYPE1,TYPE2] [--event-subtypes=TYPE1,TYPE2] [--y-axis=Y_AXIS_DATUM]')
    print('')
    print(' --event-types    : Filters by event major type, e.g. GEMM, POTRF, PAPI_L12_EXEC')
    print(' --y-axis         : Y axis datum, e.g. duration, begin, end, PAPI_L2')
    print(' --x-axis         : X axis datum, e.g. duration, begin, end, PAPI_L2')
    print(' --event-subtypes : Filters by PAPI_L12 event kernel type, e.g. GEMM, POTRF, SYRK')
    print('')

if __name__ == '__main__':
    filenames = []
    slice_st_start = None
    slice_st_stop = None
    slice_t_start = None
    slice_t_stop = None
    for index, arg in enumerate(sys.argv[1:]):
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            if arg == '--help':
                print_help()
                sys.exit(0)
            elif arg.startswith('--x-axis='):
                x_axis = arg.replace('--x-axis=', '')
            elif arg.startswith('--y-axis='):
                y_axis = arg.replace('--y-axis=', '')

            elif arg.startswith('--event-types='):
                event_types = arg.replace('--event-types=', '').split(',')
            elif arg.startswith('--event-subtypes='):
                event_subtypes = arg.replace('--event-subtypes=', '').split(',')
            elif arg.startswith('--slice-subtypes='):
                arg = arg.replace('--slice-subtypes=', '')
                slice_st_start, slice_st_stop = [int(x) for x in arg.split(':')]
            elif arg.startswith('--slice-types='):
                arg = arg.replace('--slice-types=', '')
                slice_t_start, slice_t_stop = arg.split(':')

            elif arg.startswith('--cut='):
                cuts = arg.replace('--cut=', '').split(',')
                lo_cut = float(cuts[0])
                hi_cut = float(cuts[1])
            elif arg.startswith('--ext='):
                ext = arg.replace('--ext=', '')
            else:
                event_subtypes.append(arg)

    profiles = p3_utils.autoload_profiles(filenames, convert=True, unlink=False)
    profile_sets = find_profile_sets(profiles)
    for pset in profile_sets.values()[1:]:
        if len(pset) != len(profile_sets.values()[0]):
            print('A profile set has a different size ({}) than the first set ({}),'.format(
                len(pset), len(profile_sets.values()[0])))
            print('which may cause your graph to be unbalanced.')
    profiles = automerge_profile_sets(profile_sets.values())
    profile_sets = find_profile_sets(profiles, on=[ 'exe', 'N', 'NB' ])

    for set_name, profiles in profile_sets.iteritems():
        if slice_st_start != None or slice_st_stop != None:
            event_subtypes = mpl_prefs.kernel_names[profiles[0].exe][slice_st_start:slice_st_stop]
        if slice_t_start != None or slice_t_stop != None:
            event_types = mpl_prefs.kernel_names[profiles[0].exe][slice_t_start:slice_t_stop]
        # pair up the selectors, if subtypes were specified...
        if len(event_subtypes) > 0:
            type_pairs = list(itertools.product(event_types, event_subtypes))
        else:
            event_subtypes = mpl_prefs.kernel_names[profiles[0].exe.replace('dplasma/testing/testing_', '')]
            type_pairs = list(itertools.product(event_types, event_subtypes))

        print('Now graphing the Y-axis datum \'{}\' against the X-axis datum \'{}\''.format(y_axis, x_axis) +
              ' for the event type pairs {} (where they can be found in the profile).'.format(type_pairs))

        for type_pair in type_pairs:
            if len(type_pair) == 2: # it's a tuple
                main_type = type_pair[0]
                subtype = type_pair[1]
            else:
                main_type = type_pair
                subtype = None

            plot_Y_vs_duration(profiles, x_axis, y_axis, main_type, subtype=subtype,
                               hi_cut = hi_cut, lo_cut = lo_cut, ext=ext,
                               shared_name=set_name)
        # for event_type in event_types:
        #     plot_Y_vs_duration(profiles, event_type, shared_name=set_name)


