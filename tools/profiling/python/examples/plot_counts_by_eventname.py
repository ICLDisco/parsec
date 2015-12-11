#!/usr/bin/env python

from __future__ import print_function
import os
import parsec_trace_tables as ptt
import pbt2ptt
import numpy as np
import time
import pandas

import operator

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def safe_unlink(files):
    for ufile in files:
        try:
            os.unlink(ufile) # no need to have them hanging around anymore
        except OSError:
            print('the file {} has apparently vanished.'.format(ufile))

def convert_units(source, destination):
    if source == destination:
        return 1.0
    if source == 'ns':
        if destination == 's':
            return 1.0e-9
        if destination == 'ms':
            return 1.0e-6
        if destination == 'us':
            return 1.0e-3
    if source == 'us':
        if destination == 's':
            return 1.0e-6
        if destination == 'ms':
            return 1.0e-3
        if destination == 'ns':
            return 1.0e3
    return 1.0

def scatter_papi(filenames, units, unit_modify, args):
    with Timer() as main:
        # The import of matplotlib is timed because it takes a surprisingly long time.
        with Timer() as t:
            import matplotlib
            matplotlib.use('Agg') # For use with headless systems
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        #print('Importing matplotlib took {} seconds.\n'.format(t.interval))

        trace = None
        # This is for loading and possibly converting trace files.
        with Timer() as t:
            if len(filenames) == 1 and (ptt.is_ptt(filenames[0])):
                print('Loading the HDFed trace...')
            else:
                print('Converting binary trace to the ParSEC Trace Tables format...')
                filenames[0] = pbt2ptt.convert(filenames, report_progress=True)
                print('Loading the HDFed trace...')
            trace = ptt.from_hdf(filenames[0])

        print('The load took {} seconds.\n'.format(t.interval))

        i = 0
        if args.list:
            print('Available Counters:')
            for col_name in trace.events.columns.values:
                if col_name == 'begin':
                    break
                if '_start' in col_name:
                    continue
                print(str(i) + '\t' + col_name)
                i += 1
            print('\nAvailable Events:')
            for i in range(0,len(trace.event_names)-1):
                if not trace.event_names[i].startswith('PINS_PAPI'):
                    print(str(i) + '\t' + trace.event_names[i])
            exit()

        colors_needed = 0
        num_counters = 0
        event_names = []
        event_types = []
        column_names = []

        i = 0
        available = []
        # We only need to print these if the user hasn't already specified a list of counters
        if args.counters == None:
            print('/nPlease select the counter(s) you want to plot as a comma-separated list.  Enter \'-1\' for all.')
            print('You may also specify a range of events within the list.  For instance, 4-6 translates to 4,5,6.')
            print('Example: 0,1,4-6,8\n')
        for col_name in trace.events.columns.values:
            if col_name == 'begin':
                break
            if '_start' in col_name:
                continue
            if args.counters == None:
                print(str(i) + '\t' + col_name)
            available.append(col_name)
            i += 1
        if args.counters == None:
            selection = raw_input('Counter(s) to measure: ')
            selection = selection.replace(' ', '')
        else:
            selection = args.counters

        print('\nYour selected counter(s):')
        if selection != '-1':
            selection = selection.split(',')
            i = 0
            # Iterate through the selections and print the corresponding counters
            while True:
                # If this is a list of counters, break it up and add each to the list
                if '-' in selection[i]:
                    min_max = selection[i].split('-')
                    selection[i] = available[int(min_max[0])]
                    print(min_max[0] + '\t' + selection[i])
                    inc = 0
                    for n in range(int(min_max[0])+1,int(min_max[1])+1):
                        inc += 1
                        print(str(n) + '\t' + available[n])
                        selection.insert(i+inc, available[n])
                    i += inc + 1
                else:
                    print(selection[i] + '\t' + available[int(selection[i])])
                    selection[i] = available[int(selection[i])]
                    i += 1
                if i == len(selection):
                    break
        # Add all of the counters to the selection
        else:
            selection = []
            for i in range(0,len(available)):
                print(str(i) + '\t' + available[i])
                selection.append(available[i])
        column_names.extend(selection)
        num_counters = len(selection)

        # We only need to do this if the user hasn't specified a list of events already
        if args.events == None:
            print('\n\nPlease select the event(s) you want to plot as a comma-separated list.  Enter \'-1\' for all.')
            print('You may also specify a range of events within the list.  For instance, 4-6 translates to 4,5,6.')
            print('Example: 0,1,4-6,8\n')
            for i in range(0,len(trace.event_names)-1):
                if not trace.event_names[i].startswith('PINS_PAPI'):
                    print(str(i) + '\t' + trace.event_names[i])
            selection = raw_input('Event(s) to Measure: ')
            selection = selection.replace(' ', '')
        else:
            selection = args.events

        print('\nYour selected event(s):')
        if selection != '-1':
            selection = selection.split(',')
            i = 0
            # Iterate through the selections and print the corresponding events
            while True:
                # If this is a list of events, break it up and add each to the list
                if '-' in selection[i]:
                    min_max = selection[i].split('-')
                    selection[i] = trace.event_names[int(min_max[0])]
                    event_types.append(trace.event_types[selection[i]])
                    print(min_max[0] + '\t' + trace.event_names[int(min_max[0])])
                    inc = 0
                    for n in range(int(min_max[0])+1,int(min_max[1])+1):
                        inc += 1
                        print(str(n) + '\t' + trace.event_names[n])
                        selection.insert(i+inc, trace.event_names[n])
                        event_types.append(trace.event_types[selection[i+inc]])
                    i += inc + 1
                else:
                    print(selection[i] + '\t' + trace.event_names[int(selection[i])])
                    selection[i] = trace.event_names[int(selection[i])]
                    event_types.append(trace.event_types[selection[i]])
                    i += 1
                if i == len(selection):
                    break
        else:
            selection = []
            for i in range(0,len(trace.event_names)-1):
                if not trace.event_names[i].startswith('PINS_PAPI'):
                    print(str(i) + '\t' + trace.event_names[i])
                    selection.append(trace.event_names[i])
                    event_types.append(trace.event_types[selection[-1]])
        print('')
        event_names = selection

        colors_needed = num_counters * len(event_names)
        print('Colors Needed: ' + str(colors_needed))

        # The counter_data list will store subsets of the pandas dataframes for the
        # columns containing PAPI counter information.  The column_names list stores
        # the names corresponding to those columns.
        counter_data = []
        counter_dict = []
        event_data = []
        print('Populating user-defined lists...')
        with Timer() as t:
            # We start from the beginning, which is where the PAPI event columns will be.
            for i in range(0,len(trace.events.columns.values)):
                column_name = trace.events.columns.values[i]
                # If we hit the 'begin' column, there aren't any more PAPI event columns.
                if column_name == 'begin':
                    break
                if '_start' in column_name:
                    continue
                if column_name not in column_names:
                    continue
                # We only care about the data in this column for which there is data.
                # Note: counter_data actually stores all of the rows for which the column
                #       of interest is not NULL.
                counter_data.append(trace.events[:][trace.events[column_name].notnull()])
                prev = 0
                counter_dict.append(dict())

                end_count = counter_data[-1][column_name].values.tolist()
                start_count = counter_data[-1][column_name + '_start'].values.tolist()
                for j in range(0,len(end_count)):
                    counter_dict[-1][end_count[j]] = end_count[j] - start_count[j]
            for i in range(0,len(event_types)):
                event_data.append(trace.events[:][trace.events.type == event_types[i]])
        print('Populating the lists took {} seconds.\n'.format(t.interval))

        # Start the plot of the figure with a relatively large size.
        fig = plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        # Start the color iterator so we can plot each column in its own color.
        colors = iter(cm.rainbow(np.linspace(0, 1, colors_needed)))
        print('Plotting all selected counters and events together...')
        with Timer() as t:
            # Iterate through all of the selected counters
            for i in range(0, len(counter_data)):
                # For each counter, iterate through all of the selected event types
                for j in range(0, len(event_types)):
                    # The begins, ends, and streams lists are used for selecting correct records from pandas
                    begins = event_data[j].begin.tolist()
                    ends = event_data[j].end.tolist()
                    streams = event_data[j].stream_id.tolist()
                    counts = []
                    times = []

                    print('Processing and plotting \'' + event_names[j] + '_' + column_names[i] + '\'...')
                    # Iterate through all of the events within this event type
                    for k in range(0, len(event_data[j])):
                        # This command grabs all of the rows for which the counter data and the event data overlap in time on the same execution stream
                        temp_data = counter_data[i][:][(counter_data[i].begin < ends[k])
                                                       & (counter_data[i].end > begins[k])
                                                       & (counter_data[i].stream_id == streams[k])].loc[:,[column_names[i],'begin','end']].values.tolist()
                        # Iterate through all of the counter data rows for this event.
                        # Note: There will typically be only 1 overlapping counter row, but it is possible there could be more
                        for l in range(0,len(temp_data)):
                            temp_begin = 0
                            temp_end = 0
                            if temp_data[l][1] < begins[k]:
                                # The measurement started before the event
                                temp_begin = begins[k]
                            else:
                                # The measurement started after the event
                                temp_begin = temp_data[l][1]
                            if temp_data[l][2] > ends[k]:
                                # The measurement ended after the event
                                temp_end = ends[k]
                            else:
                                # The measurement ended before the event
                                temp_end = temp_data[l][2]
                            # This is the proportion of overlap between the counter and event data
                            overlap = (temp_end-temp_begin)/(temp_data[l][2]-temp_data[l][1])
                            # Only the proportion of the counter data that corresponds to this event is recorded.
                            # Note: This is an approximation, because events will inevitably accumulate counts
                            #       at different rates.
                            counts.append(int(overlap*counter_dict[i][temp_data[l][0]]))
                            times.append(temp_end * unit_modify)
                    # Plot the data for this event type
                    plt.scatter(times, counts, color = next(colors), label = event_names[j] + '_' + column_names[i])

                plt.title('Counts by Event Name')
                plt.ylim(ymin = 0)
                plt.xlim(xmin = 0)
                plt.ylabel('Count')
                if units != 'c':
                    plt.xlabel('Time (' + units + ')')
                else:
                    plt.xlabel('Cycles')

                ax = plt.subplot(111)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                plt.show()

        figure_name = 'counts_by_eventname.png'

        if args.output != None:
            figure_name = args.output

        print('Saving plot as ' + figure_name + '...')
        fig.savefig(figure_name)

        print('Plotting and saving took {} seconds.'.format(t.interval))

    print('Total Time: {} seconds\n'.format(main.interval))

if __name__ == '__main__':
    import sys

    import argparse
    parser = argparse.ArgumentParser(usage='plot_counts_by_eventname.py [filenames of profile files] -t [time_units] -u [display_units] -c [counters_list] -e [events_list]',
                                     description='Creates scatter plots of PAPI counter values by event for PaRSEC traces using the papi PINS module.')
    # This argument allows the user to supply the units that the trace values are in for times.
    parser.add_argument('-t', '--time-units', default=None,
                        help='(Optional) Units that PaRSEC used for timing.  Accepted values: ns (nanoseconds), us (microseconds), and c (cycles).')
    # This argument allows the user to supply the units that they want to be shown in the plots.
    parser.add_argument('-u', '--display-units', default=None,
                        help='(Optional) Units that you want the plots to use for time. Accepted values: ns (nanoseconds), us (microseconds), ms (miliseconds), and s (seconds).')
    # This argument allows the user to supply the list of counters they want to be shown in the plots.
    parser.add_argument('-c', '--counters', default=None,
                        help='(Optional) A comma-separated list of the indices of the counters you wish to plot. \'-1\' for all and #-# for a contiguous group.  Ex: 0,2,4-6')
    # This argument allows the user to supply the list of events they want to be shown in the plots.
    parser.add_argument('-e', '--events', default=None,
                        help='(Optional) A comma-separated list of the indices of the events you wish to plot. \'-1\' for all and #-# for a contiguous group.  Ex: 0,2,4-6')
    # This argument allows the user to specify the name of their output image.
    parser.add_argument('-o', '--output', default=None,
                        help='(Optional) The name of the output image.  The default name is counts_by_eventname.png')
    # This argument allows the user to list the available counters and events for their supplied trace.
    parser.add_argument('-l', '--list', action='store_true',
                        help='(Optional) This option lists the available counters and events for a given trace.')

    args, filenames = parser.parse_known_args()

    units = args.time_units
    if args.time_units == None:
        print('No units supplied.  Defaulting to ns.')
        units = 'ns'
    else:
        if args.time_units == 'ns':
            print('Found units: \'' + args.time_units + '\'')
        elif args.time_units == 'us':
            print('Found units: \'' + args.time_units + '\'')
        elif args.time_units == 'c':
            print('Found units: \'' + args.time_units + '\'.')
        else:
            print('Found units: \'' + args.time_units + '\', which is not an accepted unit.')
            print('Accepted units are ns (nanoseconds), us (microseconds), and c (cycles).')
            print('Defaulting to nanoseconds')
            units = 'ns'

    unit_modify = 1.0
    if args.display_units != None:
        if args.time_units == 'c':
            print('Cannot convert between cycles and time.')
        elif args.display_units == 's':
            print('Graphs will display time in seconds.')
            unit_modify = convert_units(units, args.display_units)
            units = args.display_units
        elif args.display_units == 'ms':
            print('Graphs will display time in miliseconds.')
            unit_modify = convert_units(units, args.display_units)
            units = args.display_units
        elif args.display_units == 'us':
            print('Graphs will display time in microseconds.')
            unit_modify = convert_units(units, args.display_units)
            units = args.display_units
        elif args.display_units == 'ns':
            print('Graphs will display time in nanoseconds.')
            unit_modify = convert_units(units, args.display_units)
            units = args.display_units
        else:
            print('\'' + args.display_units +'\' is not an accepted unit.')
            print('Accepted units are ns (nanoseconds), us (microseconds), ms (miliseconds), and s (seconds).')
            print('Units will remain \'' + units + '\'.')
    print('Conversion Factor: ' + str(unit_modify))

    print('')

    # Plot the PAPI counter columns from the files in 'filenames' that have 'units'
    # and use 'unit_modify' to change the units to the desired display units.
    scatter_papi(filenames, units, unit_modify, args)
