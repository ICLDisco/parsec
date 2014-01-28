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
import papi_core_utils
from common_utils import *

default_ext = 'pdf' # public - change before using

default_x_axis = 'NB'
default_groupby = 'sched'
default_infos = ['hostname', 'exe', 'ncores', 'sched', 'N', 'NB', 'POTRF_PRI_CHANGE']
default_tag = None

def plot_performance(profiles, x_axis, groupby=default_groupby, infos=default_infos,
                     ext=default_ext, sort_ascending=True, groupby2=None, filename_tag=None):
    # remove nonsensical info keys
    infos = [x for x in infos if x != groupby and x != groupby2 and x != x_axis]
    x_axis_name = x_axis
    groupby_name = groupby
    groupby_name2 = ''
    if groupby2:
        groupby_name2 = str(groupby2)

    info_list = [profile.information for profile in profiles]
    for info in info_list: # temporary hack
        try:
            if info.exe.endswith('potrf'):
                ptt.nice_val(info, 'POTRF_PRI_CHANGE')
        except Exception as e:
            info['POTRF_PRI_CHANGE'] = 0

    matched_info = match_dicts(info_list)

    x_axis = ptt.raw_key(matched_info, x_axis)

    colors=mpl_prefs.colors[groupby]
    groupby = ptt.raw_key(matched_info, groupby)

        # groupby = [ptt.raw_key(matched_info, gb) for gb in groupby]
        # colors=mpl_prefs.colors[groupby]

    # make DataFrame out of profile informations
    all_profiles = pd.DataFrame(info_list)

    groups = all_profiles.groupby(groupby)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for group_id, group in groups:
        color = colors[group_id.upper()]
        if groupby2:
            ls = ['-', ':', '--', '_']
            groups2 = group.groupby(groupby2)
            for i, (group_id2, group2) in enumerate(groups2):
                do_plot(ax, group2, x_axis, 'GFLOPS', sort_ascending=sort_ascending,
                        label=group_id.upper() + ' : ' + str(group_id2), color=color,
                        linestyle=ls[i])
        else:
            do_plot(ax, group, x_axis, 'GFLOPS', sort_ascending=sort_ascending,
                    label=group_id.upper(), color=color)

    if groupby2:
        ax.legend(loc='best', title=groupby_name + ' : ' + str(groupby_name2))
    else:
        ax.legend(loc='best', title=groupby_name)

    title = ''
    try:
        title += ptt.nice_val(matched_info, 'exe').upper() + ' '
    except:
        pass
    title += 'Performance vs {}'.format(x_axis_name)
    try:
        title += ' on ' + matched_info.hostname
    except:
        pass
    title += '\n' + ptt.describe_dict(
        matched_info, keys=[x for x in infos if x not in ['hostname', 'exe']],
        key_val_sep=': ', sep=', ')
    print('Plotted', title)

    ax.set_title(title)
    ax.grid(True)
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel('Performance (GFLOPS/s)')
    fig.set_size_inches(10, 7)

    names = [profile.name(infos=infos) for profile in profiles]
    filename = 'perf_vs_' + str(x_axis_name) + '_'
    filename += longest_substr(names).strip('_')
    if filename_tag:
        filename += '_' + filename_tag

    fig.savefig(filename + '.' + ext, dpi=300, bbox_inches='tight')

def do_plot(axis, dataframe, x_axis, y_axis, sort_ascending=True,
            color=None, label=None, linestyle='-'):
    dataframe.sort_index(by=x_axis, ascending=sort_ascending, inplace=True)
    axis.plot(dataframe[x_axis], dataframe[y_axis], color=color, label=label,
              linestyle=linestyle)

if __name__ == '__main__':
    filenames = []

    import argparse
    parser = argparse.ArgumentParser(description='Plots performance against the chosen X axis item.')
    parser.add_argument('-x', '--x-axis', default=default_x_axis)
    parser.add_argument('-f', '--file-ext', default=default_ext)
    parser.add_argument('-g', '--group-by', default=default_groupby)
    parser.add_argument('-s', '--second-group-by', default=None)
    parser.add_argument('-t', '--filename-tag', default=None)

    args, filenames = parser.parse_known_args()

    # load profiles as skeletons
    profiles = ptt_utils.autoload_profiles(filenames, skeleton_only=True)

    plot_performance(profiles, args.x_axis, infos=default_infos, groupby=args.group_by,
                     ext=args.file_ext, groupby2=args.second_group_by,
                     filename_tag=args.filename_tag)
