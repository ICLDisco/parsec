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

default_ext = 'pdf' # public - may change before calling plot_y_vs_x_by

default_y_axis = 'GFLOPS'
default_x_axis = 'NB'
default_by = ['hostname', 'exe', 'load_dir', 'NCORES', 'sched', 'N', 'NB', 'POTRF_PRI_CHANGE']
default_color_group_name = 'sched'
default_colors = mpl_prefs.colors[default_color_group_name]
default_tag = None

x_max = None
x_min = None
y_min = None
y_max = None

def plot_y_vs_x_by(dataframe, x_axis, y_axis, by,
                   colors=default_colors, color_group_name=default_color_group_name,
                   ext=default_ext, filename_tag=None):
    # we need 'by' to be a list
    if isinstance(by, basestring):
        by = [by]

    dataframe.sort_index(by=x_axis, inplace=True)

    # remove nonsensical info keys
    by = [gb for gb in by if gb not in [x_axis, y_axis] and gb in dataframe]

    # determine which by names will help split the given dataset
    naive_groups = dataframe.groupby(by)
    naive_keys = naive_groups.groups.keys()[0] # keys() always returns a list of tuples
    naive_name_to_key = dict(zip(naive_groups.grouper.names, naive_keys))
    title_info = set.intersection(*[set(gkey) for gkey in naive_groups.groups.keys()])

    by = [naive_groups.grouper.names[i] for i, key in enumerate(naive_keys)
               if key not in title_info]
    title_names = [naive_groups.grouper.names[i] for i, key in enumerate(naive_keys)
                  if key in title_info]
    if len(by) == 0 and len(title_names) > 0:
        by = [title_names[0]]

    final_groups = dataframe.groupby(by)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    color_ct = 0
    for gkey, group in final_groups:
        # if gkey is a simple string, it won't work well with zip in the following line
        gkey = [gkey] if isinstance(gkey, basestring) else gkey
        gname_to_gkey = dict(zip(final_groups.grouper.names, gkey))
        if color_group_name:
            color = colors[gname_to_gkey[color_group_name]]
        else:
            color = colors[color_ct % len(colors)]
            color_ct += 1
        linestyle = '-'
        marker = ''

        data_mean = group.groupby(x_axis)[y_axis].mean()
        # data_std = group.groupby(x_axis)[y_axis].std()
        ax.plot(data_mean.index, data_mean.values, label=' '.join(gkey),
                color=color, linestyle=linestyle, marker=marker)

    title = '{} vs {} by {}'.format(y_axis, x_axis, ', '.join(by))
    title += '\nwith ' + '\n'.join(wrap(
        ptt.describe_dict(naive_name_to_key, keys=title_names, key_val_sep=': ', sep=', '), 80))
    print('Plotted', title)

    ax.set_title(title)
    ax.grid(True)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.legend(loc='best', title=' '.join(by))

    # adjust xy limits
    xlims = list(ax.get_xlim())
    if x_min:
        xlims[0] = x_min
    if x_max:
        xlims[1] = x_max
    ax.set_xlim(xlims)
    ylims = list(ax.get_ylim())
    if y_min:
        ylims[0] = y_min
    if y_max:
        ylims[1] = y_max
    ax.set_ylim(ylims)

    fig.set_size_inches(10, 7)
    plt.tight_layout()

    filename = (str(y_axis) + '_vs_' + str(x_axis) +
                '_by_' + '_'.join(by) + '_with_' +
                '_'.join(
                    str(ptt.nice_val(dataframe.iloc[0], key)) for key in title_names))
    if filename_tag:
        filename += '_' + filename_tag

    fig.savefig(filename + '.' + ext, dpi=300, bbox_inches='tight')


def parse_xylimits(limits_string):
    try:
        xy_max = float(limits_string)
        xy_min = None
    except:
        try:
            xylims = limits_string.split(':')
            xy_min = float(xylims[0])
            xy_max = float(xylims[1])
        except:
            xy_min = None
            xy_max = None
    return xy_min, xy_max

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plots Y axis param against the X axis param.')
    parser.add_argument('-x', '--x-axis', default=default_x_axis)
    parser.add_argument('-y', '--y-axis', default=default_y_axis)
    parser.add_argument('-b', '--by', default=default_by)
    parser.add_argument('-f', '--file-ext', default=default_ext)
    parser.add_argument('-t', '--filename-tag', default=None)
    parser.add_argument('--xlim', default=None)
    parser.add_argument('--ylim', default=None)

    args, filenames = parser.parse_known_args()

    if isinstance(args.by, basestring):
        args.by = args.by.split(',')

    # parse user-supplied xy limits
    if args.ylim:
        y_min, y_max = parse_xylimits(args.ylim)
    if args.xlim:
        x_min, x_max = parse_xylimits(args.xlim)

    # load traces as skeletons
    traces = ptt_utils.autoload_traces(filenames, skeleton_only=True)
    # make plottable DataFrame out of trace informations
    all_traces_info = pd.DataFrame([trace.information for trace in traces])

    plot_y_vs_x_by(all_traces_info, args.x_axis, args.y_axis, args.by,
                   ext=args.file_ext, filename_tag=args.filename_tag)
