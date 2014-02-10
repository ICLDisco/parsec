#!/usr/bin/env python

from __future__ import print_function
import os, sys
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import parsec_trace_tables as ptt
import ptt_utils
import mpl_prefs
from textwrap import wrap
import papi_core_utils
from common_utils import *

default_ext = 'pdf' # public - change before using

default_y_axis = 'GFLOPS'
default_x_axis = 'NB'
default_groupby = ['hostname', 'exe', 'load_dir', 'NCORES', 'sched', 'N', 'NB', 'POTRF_PRI_CHANGE']
default_color_group_name = 'sched'
default_colors = mpl_prefs.colors[default_color_group_name]
default_tag = None

y_min = None
y_max = None

def plot_performance(traces, x_axis, y_axis, groupby=default_groupby,
                     colors=default_colors, color_group_name=default_color_group_name,
                     ext=default_ext, sort_ascending=True, filename_tag=None):
    if isinstance(groupby, basestring):
        groupby = [groupby]
    # remove nonsensical info keys
    groupby = [gb for gb in groupby if gb != x_axis]

    # make DataFrame out of trace informations
    all_traces_info = pd.DataFrame([trace.information for trace in traces])
    all_traces_info.sort_index(by=x_axis, inplace=True)

    # now register the matching group IDs
    naive_groups = all_traces_info.groupby(groupby)
    title_info = set.intersection(*[set(gkey) for gkey, group in naive_groups])
    naive_keys = naive_groups.groups.keys()[0]

    groupby = [naive_groups.grouper.names[i] for i, key in enumerate(naive_keys)
               if key not in title_info]
    title_keys = [naive_groups.grouper.names[i] for i, key in enumerate(naive_keys)
                  if key in title_info]
    if len(groupby) == 0 and len(title_keys) > 0:
        groupby = [title_keys[0]]

    final_groups = all_traces_info.groupby(groupby)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for gkey, group in final_groups:
        if isinstance(gkey, basestring):
            gkey = [gkey]
        name_to_key = dict(zip(final_groups.grouper.names, gkey))
        color = colors[name_to_key[color_group_name]]
        linestyle = '-'
        marker = ''
        agg_groups = group.groupby(x_axis)[y_axis].mean()

        ax.plot(agg_groups.index, agg_groups.values, label=' '.join(gkey),
                color=color, linestyle=linestyle, marker=marker)

    ax.legend(loc='best', title=' '.join(groupby))

    title = '{} vs {} by {}'.format(y_axis, x_axis, ', '.join(groupby))
    title += '\nwith ' + '\n'.join(wrap(ptt.describe_dict(
        all_traces_info.iloc[0], keys=title_keys,
        key_val_sep=': ', sep=', '), 80))
    print('Plotted', title)

    ax.set_title(title)
    ax.grid(True)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    fig.set_size_inches(10, 7)

    ylims = list(ax.get_ylim())
    if y_max:
        ylims[1] = y_max
    if y_min:
        ylims[0] = y_min
    ax.set_ylim(ylims)

    # names = [trace.name(infos=title_keys) for trace in traces]
    filename = (str(y_axis) + '_vs_' + str(x_axis) +
                '_by_' + '_'.join(groupby) + '_with_' +
                '_'.join(
                    str(ptt.nice_val(all_traces_info.iloc[0], key)) for key in title_keys))
    # filename += longest_substr(names).strip('_')
    if filename_tag:
        filename += '_' + filename_tag

    plt.tight_layout()

    fig.savefig(filename + '.' + ext, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    filenames = []

    import argparse
    parser = argparse.ArgumentParser(description='Plots Y axis item against the chosen X axis item.')
    parser.add_argument('-x', '--x-axis', default=default_x_axis)
    parser.add_argument('-y', '--y-axis', default=default_y_axis)
    parser.add_argument('-f', '--file-ext', default=default_ext)
    parser.add_argument('-g', '--group-by', default=default_groupby)
    parser.add_argument('-s', '--second-group-by', default=None)
    parser.add_argument('-t', '--filename-tag', default=None)
    parser.add_argument('--ylim', default=None)

    args, filenames = parser.parse_known_args()

    # parse user-supplied ylimits
    if args.ylim:
        try:
            y_max = float(args.ylim)
        except:
            try:
                ylims = args.ylim.split(':')
                y_max = float(ylims[1])
                y_min = float(ylims[0])
            except:
                y_max = None
                y_min = None

    # load traces as skeletons
    traces = ptt_utils.autoload_traces(filenames, skeleton_only=True)

    if isinstance(args.group_by, basestring):
        args.group_by = args.group_by.split(',')

    plot_performance(traces, args.x_axis, args.y_axis, groupby=args.group_by,
                     ext=args.file_ext, filename_tag=args.filename_tag)
