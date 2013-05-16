#! /usr/bin/env python

from profiling import *
from profiling_info import *
from pretty_print_profile_stats import *
import py_dbpreader as pread
import os, sys

if __name__ == '__main__':
    filenames = []
    task_focus = [] # print only these tasks
    for index, arg in enumerate(sys.argv[1:]):
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            task_focus.append(arg)

    profile = pread.readProfile(filenames)
    print(profile.handle_counts)

    event_stats = profile.event_type_stats
    printers = []

    total_stats = ExecSelectStats('TOTAL')
    for key, stats in event_stats.iteritems():
        if key == 'PINS_EXEC': # or key == 'PINS_SELECT':
            for pkey, pstats in stats.exec_stats.iteritems():
                if pkey in task_focus or 'all' in task_focus:
                    new_printer = LinePrinter()
                    new_printer.append(pstats)
                    new_printer.sorter = pstats
                    printers.append(new_printer)
                total_stats += pstats
    total_printer = LinePrinter()
    total_printer.append(total_stats)
    total_printer.sorter = total_stats
    printers.append(total_printer)

    total_stats = SocketStats()
    total_count = 0
    for key, stats in event_stats.iteritems():
        if key != 'PINS_EXEC' and key != 'PINS_SELECT':
            total_count += stats.count
    for key, stats in event_stats.iteritems():
        if key == 'PINS_SOCKET':
            for pkey, pstats in stats.socket_stats.iteritems():
                pstats.count = total_count
                total_stats += pstats
    total_printer = LinePrinter()
    total_printer.append(total_stats)
    total_printer.sorter = total_stats
    printers.append(total_printer)
    
    printers.sort(key = lambda x: (x.sorter.name))
    if len(printers) == 0:
        print('no events found matching your parameters')
    else:
        prev_printer = printers[0]
        print(prev_printer.row_header())
        print(prev_printer.row())
        for printer in printers[1:]:
            if printer.row_header() != prev_printer.row_header():
                print(prev_printer.row_header())
                print('')
                print(printer.row_header())
                prev_printer = printer
            print(printer.row())
        print(printers[-1].row_header())

