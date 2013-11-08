#!/usr/bin/env python

from __future__ import print_function
from parsec_profiling import *
import parsec_binprof as p3_bin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_prefs
import binprof_utils as p3_utils
import os, sys
import itertools

# defaults
y_axes = ['PAPI_L3']
x_axis = 'duration'
lo_cut = 00
hi_cut = 100
y_lo_cut = 00
y_hi_cut = 99.1
ext = 'png'
event_types = ['PAPI_CORE_EXEC']
event_subtypes = ['GEMM']
bins = 500

def plot_Y_vs_X_colormap(profile, x_axis, y_axis, filters, profile_descrip='', filters_descrip='',
                         bins=bins, hi_cut=hi_cut, lo_cut=lo_cut, tiers=(0.85, 0.7),
                         ext=ext):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    events = profile.filter_events(filters)

    events = events.sort(x_axis)
    events = events[int(len(events) * lo_cut * 0.01):
                    int(len(events) * hi_cut * 0.01)]
    events = events.sort(y_axis)
    events = events[int(len(events) * y_lo_cut * 0.01):
                    int(len(events) * y_hi_cut * 0.01)]

    label = '{}: {:.1f} gflops/s'.format(profile.sched.upper(),
                                         profile.gflops)

    # calculate correlations
    p_corr = events[[y_axis, x_axis]].corr(method='pearson')
    s_corr = events[[y_axis, x_axis]].corr(method='spearman')
    p_corr_num = p_corr[y_axis].ix[1]
    s_corr_num = s_corr[y_axis].ix[1]

    if abs(s_corr_num) > tiers[0]:
        corr_cmap = cm.spectral
    elif abs(s_corr_num) > tiers[1]:
        corr_cmap = cm.jet
    else:
        corr_cmap = cm.Dark2

    heatmap, xedges, yedges = np.histogram2d(events[x_axis], events[y_axis], bins=bins,
                                             normed=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.imshow(heatmap.transpose()[::1], extent=extent, origin='lower',
              interpolation='none', cmap=corr_cmap)
    ax.set_aspect('auto', anchor='SW')
    # ax.hexbin(events[x_axis], events[y_axis], gridsize=gridsize,
    #           cmap=cm.jet, bins=None)

    ax.set_title('{} vs {} of {}\n'.format(y_axis, x_axis, filters_descrip) +
                 'for {} where '.format(profile.exe) +
                 'N = {}, NB = {}, IB = {}, sched = {}, on {}'.format(profile.N, profile.NB,
                                                                      profile.IB, profile.sched, profile.hostname))
    if not ax.has_data():
        print('Plot has no data.')
        return None
    ax.grid(True)
    ax.set_ylabel(y_axis)
    cut_label = ''
    if hi_cut < 100 or lo_cut > 0:
        cut_label = ', excl. below {}% & above {}%'.format(lo_cut, hi_cut)
    ax.set_xlabel('{} of {} kernels'.format(x_axis, filters_descrip) + cut_label)

    corr_text = 'Pearson: {:.3f}\nSpearman: {:.3f}'.format(p_corr_num, s_corr_num)

    fig.text(0.85, 0.15, corr_text,
             horizontalalignment='right',
             fontsize=40, color='white',
             alpha=0.5, family='monospace',
             transform=ax.transAxes)

    short_exe = profile.exe
    if profile.exe == 'dpotrf':
        short_exe = 'PO'
    elif profile.exe == 'dgetrf':
        short_exe = 'LU'
    elif profile.exe == 'dgeqrf':
        short_exe = 'QR'
    descrip_text = '{} {} {}'.format(y_axis, profile.sched, short_exe)

    fig.text(0.15, 0.8, descrip_text,
             horizontalalignment='left',
             fontsize=40, color='white',
             alpha=0.5,family='monospace',
             transform=ax.transAxes)

    fig.set_size_inches(12, 8)
    fig.set_dpi(300)
    filename = re.sub('[\(\)\' :]' , '',
                      ('{}_vs_{}_{}'.format(y_axis, x_axis, profile_descrip) +
                       '_{}_{}-{}'.format(filters_descrip, lo_cut, hi_cut) +
                       '_colormap.{}'.format(ext)))
    fig.savefig(filename, bbox_inches='tight')

def print_help():
    print('')
    print(' The script plots the selected Y axis datum against the X axis datum in a colormap.')
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

            elif arg.startswith('--cut='):
                cuts = arg.replace('--cut=', '').split(',')
                lo_cut = float(cuts[0])
                hi_cut = float(cuts[1])
            elif arg.startswith('--ext='):
                ext = arg.replace('--ext=', '')
            elif arg.startswith('--bins='):
                bins = int(arg.replace('--bins=', ''))
            elif arg.startswith('--papi-core-all'):
                papi_core_all = True
            else:
                event_subtypes.append(arg)

    profiles = p3_utils.autoload_profiles(filenames, convert=True, unlink=False)
    # then divide the profiles into sets based on equivalent command lines
    profile_sets = find_profile_sets(profiles, on=['exe', 'N', 'NB', 'IB', 'sched'])
    # then merge those sets so that multiple trials of the same params are aggregated
    profiles = automerge_profile_sets(profile_sets.values())

    for profile, name in zip(profiles, profile_sets.keys()):
        if slice_st_start != None or slice_st_stop != None:
            event_subtypes = mpl_prefs.kernel_names[profile.exe][slice_st_start:slice_st_stop]
        if slice_t_start != None or slice_t_stop != None:
            event_types = mpl_prefs.kernel_names[profile.exe][slice_t_start:slice_t_stop]

        if papi_core_all:
            event_types = []
            event_subtypes = []
            # find the PAPI_CORE_EXEC event(s)
            for event_name in profile.event_types.keys():
                if event_name.startswith('PAPI_CORE_EXEC_'):
                    event_types.append(event_name)
                    y_axes = p3_bin.papi_core_evt_value_lbls[event_name]
                    break
            event_subtypes = mpl_prefs.kernel_names[profile.exe][:1]

        # pair up the selectors, if subtypes were specified...
        if len(event_subtypes) > 0:
            type_pairs = list(itertools.product(event_types, event_subtypes))
        else:
            event_subtypes = mpl_prefs.kernel_names[profile.exe]
            type_pairs = list(itertools.product(event_types, event_subtypes))

        for y_axis in y_axes:
            for type_pair in type_pairs:
                filters = []
                if len(type_pair) == 2: # it's a tuple
                    filters.append('type==.event_types[\'' + type_pair[0] + '\']')
                    filters.append('kernel_type==.event_types[\''+type_pair[1]+'\']')
                else:
                    filters.append('type==.event_types[\'' + type_pair + '\']')

                print('plotting {} vs {} for {} in {}'.format(y_axis, x_axis, filters, profile))
                plot_Y_vs_X_colormap(profile, x_axis, y_axis, filters,
                                     profile_descrip=name, filters_descrip=str(type_pair),
                                     hi_cut = hi_cut, lo_cut = lo_cut, ext=ext, bins=bins)


