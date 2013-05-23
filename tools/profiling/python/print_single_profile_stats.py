#! /usr/bin/env python

from profiling import *
from profiling_info import *
from pretty_print_profile_stats import *
import py_dbpreader as pread
import os, sys

import time

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self
    
    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

if __name__ == '__main__':
    filenames = []
    task_focus = [] # print only these tasks
    for index, arg in enumerate(sys.argv[1:]):
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            task_focus.append(arg.lower())
            
    profile = None
    with Timer() as t:
        profile = pread.readProfile(filenames)
    print(t.interval)

    with Timer() as t:
        cPickle.dump(profile, open('raw.pickle', 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
    print(t.interval)
    
    print('handle counts: ' + str(profile.get_handle_counts()))

    printers = []

    total_stats = ExecSelectStats('TOTAL')
    for key, event_type in profile.dictionary.iteritems():
        if key == 'PINS_EXEC': # or key == 'PINS_SELECT':
            for pkey, pstats in event_type.stats.exec_stats.iteritems():
                if pkey.lower() in task_focus or 'all' in task_focus:
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
    for key, event_type in profile.dictionary.iteritems():
        if key != 'PINS_EXEC' and key != 'PINS_SELECT':
            total_count += event_type.stats.count
    for key, event_type in profile.dictionary.iteritems():
        if key == 'PINS_SOCKET':
            for pkey, pstats in event_type.stats.socket_stats.iteritems():
                pstats.count = total_count
                total_stats += pstats
    if total_stats.count > 0:
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

    with Timer() as t:
        profile.pickle('test.pickle')
    print(t.interval)