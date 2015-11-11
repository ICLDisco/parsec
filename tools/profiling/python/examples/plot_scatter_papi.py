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

def scatter_papi(filenames, units, unit_modify):
    with Timer() as main:
        # The import of matplotlib is timed because it takes a surprisingly long time.
        with Timer() as t:
            import matplotlib
            matplotlib.use('Agg') # For use with headless systems
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        print('Importing matplotlib took {} seconds.\n'.format(t.interval))

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

        print('The load took {} seconds.'.format(t.interval))
        print('')

        # The column_data list will store subsets of the pandas dataframes for the
        # columns containing PAPI counter information.  The column_names list stores
        # the names corresponding to those columns.
        column_data = []
        column_names = []
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
                column_names.append(column_name)
                # We only care about the data in this column for which there is data.
                # Note: column_data actually stores all of the rows for which the column
                #       of interest is not NULL.
                column_data.append(trace.events[:][trace.events[column_name].notnull()])
        print('Populating the lists took {} seconds.\n'.format(t.interval))

        # Determine the maximum number of colors that we would need for one of the graphs.
        colors_needed = 0
        for event_name in trace.event_names:
            if event_name.startswith('PINS_PAPI'):
                colors_needed += 1
        print('Colors Needed: ' + str(colors_needed))

        # Start the plot of the figure with a relatively large size.
        fig = plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
        # Start the color iterator so we can plot each column in its own color.
        colors = iter(cm.rainbow(np.linspace(0, 1, colors_needed)))
        print('Plotting all PAPI counters together...')
        with Timer() as t:
            for i in range(0, len(column_data)):
                # This is done in 4 lines instead of two due to an issue with multiplying unit_modify
                temp = column_data[i]['begin'] * unit_modify
                tempX = temp.values.tolist()
                temp = column_data[i]['end'] * unit_modify
                # We should now have all of the 'x' values (time)
                tempX.extend(temp.values.tolist())

                tempY = column_data[i][:][column_names[i] + '_start'].values.tolist()
                # We should now have all of the 'y' values (count)
                tempY.extend(column_data[i][:][column_names[i]].values.tolist())

                # Note: The values in tempX and tempY are stored with the first half of the array being
                #       the '_start' values and the second half being the 'end' values, so they match up
                #       properly, however a line plot would look very odd because these values should
                #       actually be interleaved.
                plt.scatter(tempX, tempY, color = next(colors), label = column_names[i])

                plt.title('All PAPI Counters')
                plt.ylim(ymin = 0)
                plt.xlim(xmin = 0)
                plt.ylabel('Count')
                if units != 'c':
                    plt.xlabel('Time (' + units + ')')
                else:
                    plt.xlabel('Cycles')

                plt.legend(loc='upper left')
                plt.show()

        print('Saving plot as all_papi_counters.png...')
        fig.savefig('all_papi_counters.png')
        print('Plotting and saving took {} seconds.'.format(t.interval))

        # Each iteration will plot a different individual counter as its own plot.
        for i in range(0, len(column_data)):
            with Timer() as t:
                print('Plotting data for: ' + column_names[i] + '...')
                fig = plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
                # Restart the colors iterator
                colors = iter(cm.rainbow(np.linspace(0, 1, colors_needed)))

                # Plot each non-empty subset of this counter by 'type'.  This typically means
                # the counters that occurred on each core are grouped together.
                for n in range(0, len(trace.event_names)-1):
                    if trace.event_names[n].startswith('PINS_PAPI'):
                        temp = column_data[i][:][column_data[i]['type'] == n]['begin'] * unit_modify

                        if len(temp) > 0:
                            tempX = temp.values.tolist()
                            temp = column_data[i][:][column_data[i]['type'] == n]['end'] * unit_modify
                            tempX.extend(temp.values.tolist())

                            tempY = column_data[i][:][column_data[i]['type'] == n][column_names[i] + '_start'].values.tolist()
                            tempY.extend(column_data[i][:][column_data[i]['type'] == n][column_names[i]].values.tolist())

                            plt.scatter(tempX, tempY, color = next(colors),\
                                        label = trace.event_names[n].replace('PINS_PAPI_', ''))

                plt.title(column_names[i])
                plt.ylim(ymin = 0)
                plt.xlim(xmin = 0)
                plt.ylabel('Count')
                if units != 'c':
                    plt.xlabel('Time (' + units + ')')
                else:
                    plt.xlabel('Cycles')

                plt.legend(loc='upper left')

                plt.show()
                fig.savefig(column_names[i] + '.png')
                print('Saving plot as ' + column_names[i] + '.png...')
            print('Plotting and saving took {} seconds.'.format(t.interval))

    print('Total Time: {} seconds\n'.format(main.interval))

if __name__ == '__main__':
    import sys

    import argparse
    parser = argparse.ArgumentParser(description='Creates scatter plots for traces using the papi PINS module.')
    # This argument allows the user to supply the units that the trace values are in for times.
    parser.add_argument('-t', '--time-units', default=None)
    # This argument allows the user to supply the units that they want to be shown in the plots.
    parser.add_argument('-u', '--display-units', default=None)

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
    scatter_papi(filenames, units, unit_modify)
