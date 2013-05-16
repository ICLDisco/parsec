#!/usr/bin/env python
# last updated 2013-05-15
import sys
import os
import copy
import cPickle
from parsec_trials import *
from profiling import *
from profiling_info import *
from pretty_print_profile_stats import *

if __name__ == '__main__':
    files = []
    task_focus = [] # print only these tasks
    for index, arg in enumerate(sys.argv[1:]):
        if os.path.exists(arg):
            files.append(arg)
        else:
            task_focus.append(arg)
            
    trial_sets = []
    totals = []
    for f in files:
        trial_set = TrialSet.unpickle(f, load_profile=False) # just load the stats
        trial_sets.append(trial_set)

    # sort trial sets by....
    trial_sets.sort(key = lambda x: (x.ex, x.N, x.sched))
    printers = []
    total_printers = []
    
    # assemble L1/L2 misses
    for trial_set in trial_sets:
        total_stats = ExecSelectStats('TOTAL')
        for trial in trial_set[:]:
            printer = LinePrinter()
            printer.append(ItemPrinter(trial.ex, 'EXEC', length=14))
            printer.append(ItemPrinter(trial.N, 'N', length=7))
            printer.append(ItemPrinter(trial.sched, 'SCHD', length=5))
            event_stats = trial.profile_event_stats
            for key, stats in event_stats.iteritems():
                if key == 'PINS_EXEC': # or key == 'PINS_SELECT':
                    for pkey, pstats in stats.exec_stats.iteritems():
                        if pkey in task_focus or 'all' in task_focus:
                            new_printer = copy.deepcopy(printer)
                            new_printer.append(pstats)
                            new_printer.append(ItemPrinter('{:>5.1f}'.format(trial.perf), 'PERF', length=6))
                            new_printer.sorter = pstats
                            printers.append(new_printer)
                        total_stats += pstats
        total_printer = LinePrinter()
        total_printer.append(ItemPrinter(trial_set.ex, 'EXEC', length=14))
        total_printer.append(ItemPrinter(trial_set.N, 'N', length=7))
        total_printer.append(ItemPrinter(trial_set.sched, 'SCHD', length=5))
        total_printer.append(total_stats)
        total_printer.sorter = total_stats
        total_printer.append(ItemPrinter('{:>5.1f}'.format(trial_set.perf_avg), 'PERF', length=6))
        total_printers.append(total_printer)

    printers.extend(total_printers)
    printers.sort(key = lambda x: (x.sorter.name))
    if len(printers) == 0:
        print('no kernels found matching your parameters')
    else:
        prev_printer = printers[0]
        print(prev_printer.row_header())
        print(prev_printer.row())
        for printer in printers[1:]:
            if printer.row_header() != prev_printer.row_header():
                print(printer.row_header())
            print(printer.row())
        print(printers[-1].row_header())
    print('')

    printers = []
    for trial_set in trial_sets:
        total_stats = SocketStats()
        for trial in trial_set[:]:
            total_count = 0
            printer = LinePrinter()
            printer.append(ItemPrinter(trial.ex, 'EXEC', length=14))
            printer.append(ItemPrinter(trial.N, 'N', length=7))
            printer.append(ItemPrinter(trial.sched, 'SCHD', length=5))
            event_stats = trial.profile_event_stats
            for key, stats in event_stats.iteritems():
                if key != 'PINS_EXEC' and key != 'PINS_SELECT':
                    total_count += stats.count
            for key, stats in event_stats.iteritems():
                if key == 'PINS_SOCKET':
                    for pkey, pstats in stats.socket_stats.iteritems():
                        pstats.count = total_count
                        # no need to print individual trials - just get total
                        # new_printer = copy.deepcopy(printer)
                        # new_printer.append(pstats)
                        # new_printer.append(ItemPrinter('{:>5.1f}'.format(trial.perf), 'PERF', length=6))
                        # printers.append(new_printer)
                        total_stats += pstats
        total_printer = LinePrinter()
        total_printer.append(ItemPrinter(trial_set.ex, 'EXEC', length=14))
        total_printer.append(ItemPrinter(trial_set.N, 'N', length=7))
        total_printer.append(ItemPrinter(trial_set.sched, 'SCHD', length=5))
        total_printer.append(total_stats)
        total_printer.append(ItemPrinter('{:>5.1f}'.format(trial_set.perf_avg), 'PERF', length=6))
        printers.append(total_printer)

    if len(printers) == 0:
        print('no L3 events found in the files')
    else:
        prev_printer = printers[0]
        print(prev_printer.row_header())
        print(prev_printer.row())
        for printer in printers[1:]:
            if printer.row_header() != prev_printer.row_header():
                print(printer.row_header())
            print(printer.row())
        print(printers[-1].row_header())
        
