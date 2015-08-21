#!/usr/bin/env python

from __future__ import print_function
from parsec_trace_tables import *
import pbt2ptt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_prefs
import ptt_utils
import papi_core_utils
import os, sys
import itertools

# defaults
y_axes = ['PAPI_L3']
x_axis = 'duration'
event_types = ['PAPI_CORE_EXEC']
event_subtypes = ['GEMM']
std_x = 3
std_y = 3
ext = 'png'

def plot_Y_vs_X_scatter_and_sorted(traces, x_axis, y_axis, filters,
                                   trace_descrip='', filters_descrip='',
                                   std_x=std_x, std_y=std_y, ext=ext):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    traces.sort(key=lambda x: x.gflops)

    title = ''

    for trace in traces:
        events = trace.filter_events(filters)
        if x_axis == 'duration':
            events['duration'] = pandas.Series(events['end'] - events['begin'])

        if std_x:
            x_avg = events[x_axis].mean()
            events = events[:][events[x_axis] - x_avg  < events[x_axis].std() * std_x]
        if std_y:
            y_avg = events[y_axis].mean()
            events = events[:][events[y_axis] - y_avg  < events[y_axis].std() * std_y]

        label = '{}: {:.1f} gflops/s'.format(trace.sched.upper(),
                                             trace.gflops)

        ax.plot(events[x_axis], events[y_axis], linestyle='', marker='.',
                color=mpl_prefs.sched_colors[trace.sched],
                label=label)

        title = ('{} vs {} of {}'.format(y_axis, x_axis, filters_descrip) +
                 ' Tasks, By Scheduler\nfor {} where '.format(trace.exe) +
                 'N = {}, NB = {}, IB = {}, on {}'.format(trace.N, trace.NB,
                                                          trace.IB, trace.hostname))
    fig.text(0.5, 0.94, title,
             horizontalalignment='center',
             fontsize=12, #family='monospace',
             transform = ax.transAxes)

    if not ax.has_data():
        print('Plot has no data.')
        return None
    ax.legend(loc='upper center', title='SCHED: perf')
    ax.grid(True)
    ax.set_xlim(0, ax.get_xlim()[1]) # start from zero for scale
    ax.set_ylabel(y_axis)
    cut_label = ''
    if std_x:
        cut_label = ' (within {}SD)'.format(std_x)
    ax.set_xlabel('{} of {} kernels'.format(x_axis, filters_descrip) + cut_label)

    ax = ax.twiny()

    for trace in traces:
        events = trace.filter_events(filters)
        events = events.sort(y_axis)

        # cut down to ignore noise
        # -- can do better than this
        if std_x:
            x_avg = events[x_axis].mean()
            events = events[:][events[x_axis] - x_avg  < events[x_axis].std() * std_x]
        if std_y:
            y_avg = events[y_axis].mean()
            events = events[:][events[y_axis] - y_avg  < events[y_axis].std() * std_y]

        label = '{}: {:.1f} gflops/s'.format(trace.sched.upper(),
                                             trace.gflops)
        ax.plot(xrange(len(events)),
                events[y_axis],
                color=mpl_prefs.sched_colors[trace.sched],
                label=label
            )

    fig.set_size_inches(14, 7)
    fig.set_dpi(300)

    std_str = str(std_y)
    if std_y != std_x:
        str_str += '-{}'.format(std_x)
    filename = re.sub('[\(\)\' :]' , '',
                      ('{}_vs_{}_{}'.format(y_axis, x_axis, trace_descrip) +
                       '_{}_{}SD'.format(filters_descrip, std_str) +
                       '_scatter_w_hock.{}'.format(ext)))
    fig.savefig(filename, bbox_inches='tight')

def print_help():
    print('')
    print(' The script plots the selected Y axis datum against the X axis datum.')
    print('')
    print(' It will accept sets of traces as well, and will attempt to merge them if encountered.')
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
    papi_core_all = False
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
                y_axes = [arg.replace('--y-axis=', '')]

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

            elif arg.startswith('--stddev='):
                stddev = arg.replace('--stddev=', '')
                if ',' in stddev:
                    stddev = stddev.split(',')
                    std_x = float(stddev[0])
                    std_y = float(stddev[1])
                else:
                    std_x = float(stddev)
                    std_y = std_x

            elif arg.startswith('--ext='):
                ext = arg.replace('--ext=', '')
            elif arg.startswith('--papi-core-all'):
                papi_core_all = True
            else:
                event_subtypes.append(arg)

    traces = ptt_utils.autoload_traces(filenames, convert=True, unlink=False,
                                       enhance_filenames=True)
    trace_sets = find_trace_sets(traces)
    for pset in trace_sets.values()[1:]:
        if len(pset) != len(trace_sets.values()[0]):
            print('A trace set has a different size ({}) than the first set ({}),'.format(
                len(pset), len(trace_sets.values()[0])))
            print('which may cause your graph to be unbalanced.')
    traces = automerge_trace_sets(trace_sets.values())
    trace_sets = find_trace_sets(traces, on=[ 'exe', 'N', 'NB' ])

    for set_name, traces in trace_sets.iteritems():
        if slice_st_start != None or slice_st_stop != None:
            event_subtypes = mpl_prefs.kernel_names[traces[0].exe][slice_st_start:slice_st_stop]
        if slice_t_start != None or slice_t_stop != None:
            event_types = mpl_prefs.kernel_names[traces[0].exe][slice_t_start:slice_t_stop]

        if papi_core_all:
            event_types = []
            event_subtypes = []
            # find the PAPI_CORE_EXEC event(s)
            for event_name in traces[0].event_types.keys():
                if event_name.startswith('PAPI_CORE_EXEC_'):
                    event_types.append(event_name)
                    y_axes = papi_core_utils.PAPICoreEventValueLabelGetter()[event_name]
                    break
            event_subtypes = mpl_prefs.kernel_names[traces[0].exe][:1]

        # pair up the selectors, if subtypes were specified...
        if len(event_subtypes) > 0:
            type_pairs = list(itertools.product(event_types, event_subtypes))
        else:
            event_subtypes = mpl_prefs.kernel_names[traces[0].exe]
            type_pairs = list(itertools.product(event_types, event_subtypes))

        for y_axis in y_axes:
            print('Now graphing the Y-axis datum \'{}\' against the X-axis datum \'{}\''.format(y_axis, x_axis) +
                  ' for the event type pairs {} (where they can be found in the trace).'.format(type_pairs))
            for type_pair in type_pairs:
                filters = []
                if len(type_pair) == 2: # it's a tuple
                    filters.append('type==.event_types[\'' + type_pair[0] + '\']')
                    filters.append('kernel_type==.event_types[\''+type_pair[1]+'\']')
                else:
                    filters.append('type==.event_types[\'' + type_pair + '\']')

                plot_Y_vs_X_scatter_and_sorted(traces, x_axis, y_axis, filters,
                                               trace_descrip=set_name,
                                               filters_descrip=str(type_pair),
                                               std_x = std_x, std_y = std_y, ext=ext)
            # for event_type in event_types:
            #     plot_Y_vs_duration(traces, event_type, shared_name=set_name)


