#!/usr/bin/env python

from __future__ import print_function
import os, sys
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import parsec_profiling as p3
import binprof_utils as p3_utils
import mpl_prefs
import papi_core_utils
import p3_group_utils as p3_g

ext = 'pdf'

x_axis = 'NB'
infos = ['hostname', 'exe', 'ncores', 'N']

def plot_performance(profiles, x_axis, groupby='sched', infos=infos):
    x_axis_name = x_axis
    groupby_name = groupby

    info_list = [profile.information for profile in profiles]
    matched_info = p3.match_dicts(info_list)

    x_axis = p3.raw_key(matched_info, x_axis)
    if isinstance(groupby, str):
        colors=mpl_prefs.colors[groupby]
        groupby = p3.raw_key(matched_info, groupby)
    else:
        groupby = [p3.raw_key(matched_info, gb) for gb in groupby]

    # make DataFrame out of profile informations
    all_profiles = pd.DataFrame()
    all_profiles = all_profiles.append(info_list)

    groups = all_profiles.groupby(groupby)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for group_id, group in groups:
        group.sort_index(by=x_axis)
        ax.plot(group[x_axis], group['GFLOPS'], color=colors[group_id.upper()],
                label=group_id.upper())

    ax.grid(True)
    ax.legend(loc='best', title=groupby_name)
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel('Performance (GFLOPS/s)')

    title = ''
    try:
        title += p3.nice_val(matched_info, 'exe').upper() + ' '
    except:
        pass
    title += 'Performance vs {}'.format(x_axis_name)
    try:
        title += ' on ' + matched_info.hostname
    except:
        pass
    title += '; ' + p3.describe_dict(matched_info, keys=['N'], key_val_sep=': ', sep=', ')
    print('Plotted', title)

    ax.set_title(title)
    fig.set_size_inches(10, 7)

    names = [profile.name(infos=infos) for profile in profiles]

    fig.savefig(
        'perf_{}.{}'.format(p3_g.longest_substr(names), ext),
        dpi=300, bbox_inches='tight')

def print_help():
    print('Sorry, no help yet.')

if __name__ == '__main__':
    filenames = []
    for index, arg in enumerate(sys.argv[1:]):
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            if arg == '--help':
                print_help()
                sys.exit(0)
            elif arg.startswith('--x-axis='):
                x_axis = arg.replace('--x-axis=', '')
            elif arg.startswith('--ext='):
                ext = arg.replace('--ext=', '')
            else:
                print('Unknown argument', arg)

    # load profiles as skeletons
    profiles = p3_utils.autoload_profiles(filenames, skeleton_only=True)

    plot_performance(profiles, x_axis, infos=infos)
