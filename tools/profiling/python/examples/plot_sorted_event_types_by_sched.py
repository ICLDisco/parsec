#!/usr/bin/env python

from __future__ import print_function
from parsec_trace_tables import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_prefs
import ptt_group_traces as ptt_g
import os, sys
import itertools

# defaults
y_axis = 'PAPI_L2'
event_types = ['PAPI_CORE_EXEC']
event_subtypes = ['GEMM']
div_by = None
lo_cut = 00
hi_cut = 100
ext = 'pdf'

def plot_traces(traces, y_axis, main_type, subtype=None, shared_name='',
                  hi_cut=hi_cut, lo_cut=lo_cut, ext=ext):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # sort for the legend order
    traces.sort(key=lambda x: x.gflops)

    if subtype:
        type_name = main_type + '-' + subtype
    else:
        type_name = main_type

    for trace in traces:
        if subtype:
            events = trace.events[:][(trace.events.type == trace.event_types[main_type]) &
                                       (trace.events.kernel_type == trace.event_types[subtype])]
        else:
            events = trace.events[:][(trace.events.type == trace.event_types[main_type])]

        if len(events) == 0:
            print('skipping trace \'{}\' with type pair {}'.format(trace.descrip(), type_pair) +
                  'because it has no events of the selected type.')
            continue
        if y_axis not in events:
            print('skipping trace {} with type pair {}'.format(trace.descrip(), type_pair) +
                  'because it has no events of the selected type with the Y-axis variable {}.'.format(y_axis))
            continue

        sorted_events = events.sort(y_axis)

        # cut down to ignore noise
        # -- can do better than this
        sorted_events = sorted_events[int(len(sorted_events)*lo_cut * 0.01):
                                      int(len(sorted_events)*hi_cut * 0.01)]
        label = '{}: {:.1f} gflops/s'.format(trace.sched.upper(),
                                            trace.gflops)
        ax.plot(xrange(len(sorted_events)),
                sorted_events[y_axis],
                color=mpl_prefs.sched_colors[trace.sched.upper()],
                label=label
            )

        ax.set_title(y_axis + ' for {} Tasks, Sorted, By Scheduler\n'.format(type_name) +
                     'for {} where N = {}, NB = {}, IB = {}, on {}'.format(trace.exe, trace.N, trace.NB,
                                                                       trace.IB, trace.hostname))
        ax.set_ylim(sorted_events.iloc[0][y_axis],
                    sorted_events.iloc[-1][y_axis])
    if not ax.has_data():
        print('This graph is empty, so it will not be created.')
        return -1
    ax.grid(True)
    cut_label = ''
    if hi_cut < 100 or lo_cut > 0:
        cut_label = ', excl. below {}% & above {}%'.format(lo_cut, hi_cut)
    ax.set_xlabel('{} kernels, sorted by {}'.format(type_name, y_axis) + cut_label)
    ax.set_ylabel(y_axis)
    ax.legend(loc='best', title='SCHED: perf')
    fig.set_size_inches(10, 5)
    fig.set_dpi(300)
    # TODO need a better naming scheme for these files...
    fig.savefig(shared_name.replace(' ', '_') + '_' +
                type_name + '_sorted_by_' + y_axis +
                '_{}-{}_'.format(lo_cut, hi_cut) +
                'by_sched_hockey_stick.' + ext,
                bbox_inches='tight')

def print_help():
    print('')
    print(' This script was originally designed to perform hockey-stick-style plots')
    print(' of PaRSEC traces with papi_L123 events. It now supports more general plots.')
    print(' The script plots the Y axis datum against the events, sorted by the Y axis datum.')
    print('')
    print(' It will accept sets of traces as well, and will attempt to merge them if encountered.')
    print(' usage: <script_name> [PROFILE FILENAMES] [--event-types=TYPE1,TYPE2] [--event-subtypes=TYPE1,TYPE2] [--y-axis=Y_AXIS_DATUM]')
    print('')
    print(' --event-types    : Filters by event major type, e.g. GEMM, POTRF, PAPI_L12_EXEC')
    print(' --y-axis         : Y axis datum, e.g. duration, begin, end, PAPI_L2')
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
            # elif arg.startswith('--div-by='):
            #     div_by = arg.replace('--div-by=', '')
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

    # then, we group the traces appropriately...
    traces = ptt_g.autoload_traces(filenames, convert=True, unlink=False)
    trace_sets = find_trace_sets(traces)
    for pset in trace_sets.values()[1:]:
        if len(pset) != len(trace_sets.values()[0]):
            print('A trace set has a different size ({}) than the first set ({}),'.format(
                len(pset), len(trace_sets.values()[0])))
            print('which may cause your graph to look strange.')
    traces = automerge_trace_sets(trace_sets.values())
    trace_sets = find_trace_sets(traces, on=[ 'exe', 'N', 'NB' ])

    for set_name, traces in trace_sets.iteritems():
        if slice_st_start != None or slice_st_stop != None:
            event_subtypes = mpl_prefs.kernel_names[traces[0].exe][slice_st_start:slice_st_stop]
        if slice_t_start != None or slice_t_stop != None:
            event_types = mpl_prefs.kernel_names[traces[0].exe][slice_t_start:slice_t_stop]
        # pair up the selectors, if subtypes were specified...
        if len(event_subtypes) > 0:
            type_pairs = list(itertools.product(event_types, event_subtypes))
        else:
            type_pairs = [(event_type) for event_type in event_types]
        print('Now graphing the Y-axis datum \'{}\' against the '.format(y_axis) +
              ' event type pairs {}, where they can be found in the trace.'.format(type_pairs))

        for type_pair in type_pairs:
            if len(type_pair) == 2: # it's a tuple
                main_type = type_pair[0]
                subtype = type_pair[1]
            else:
                main_type = type_pair
                subtype = None

            plot_traces(traces, y_axis, main_type, subtype=subtype, ext=ext,
                          shared_name = set_name, hi_cut = hi_cut, lo_cut=lo_cut)
        # for event_type in event_types:
        #     plot_traces(traces, event_type, shared_name = set_name)
