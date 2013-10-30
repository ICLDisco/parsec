#! /usr/bin/env python

from parsec_profiling import *
# from profiling_info import *
from pretty_print_profile_stats import *
import parsec_binprof
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
        profile = parsec_binprof.read(filenames)
        # profile = Profile.unpickle('test.pickle', load_events=False)
    print(t.interval)

    printers = []

    # total_exec_stats = ExecSelectStats('TOTAL E')
    # total_exec_name = None
    # total_select_stats = ExecSelectStats('TOTAL S')
    # total_select_name = None
    # for name, event_type in profile.event_types.iteritems():
    #     if name[:4] == 'PINS' and 'EXEC' in name:
    #         total_exec_name = name
    #         for pkey, pstats in event_type.stats.exec_stats.iteritems():
    #             if pkey.lower() in task_focus or 'all' in task_focus:
    #                 new_printer = LinePrinter()
    #                 new_printer.append(ItemPrinter(name, 'EVENT', length=17))
    #                 new_printer.append(pstats)
    #                 new_printer.sorter = pstats
    #                 printers.append(new_printer)
    #             total_exec_stats += pstats
    #     if name[:4] == 'PINS' and 'SELECT' in name:
    #         total_select_name = name
    #         for pkey, pstats in event_type.stats.select_stats.iteritems():
    #             if pkey.lower() in task_focus or 'all' in task_focus:
    #                 new_printer = LinePrinter()
    #                 new_printer.append(ItemPrinter(name, 'EVENT', length=17))
    #                 new_printer.append(pstats)
    #                 new_printer.sorter = pstats
    #                 printers.append(new_printer)
    #             total_select_stats += pstats
    # if total_exec_name:
    #     total_printer = LinePrinter()
    #     total_printer.append(ItemPrinter(total_exec_name, 'EVENT', length=17))
    #     total_printer.append(total_exec_stats)
    #     total_printer.sorter = total_exec_stats
    #     printers.append(total_printer)
    # if total_select_name:
    #     total_printer = LinePrinter()
    #     total_printer.append(ItemPrinter(total_select_name, 'EVENT', length=17))
    #     total_printer.append(total_select_stats)
    #     total_printer.sorter = total_select_stats
    #     printers.append(total_printer)
    exec_keys = list()
    for name, value in profile.event_types.iteritems():
        if name[:4] == 'PINS' and 'EXEC' in name:
            exec_keys.append(value['key'])

    exec_df = profile.events[:][profile.events['key'].isin(exec_keys)]
    print(exec_df[ profile.basic_columns + ['PAPI_L2'] ].describe())
    # print((exec_df[ profile.event_columns + ['PAPI_L2'] ] / float(len(exec_df))).describe())

    socket_df = profile.events[:][profile.events['PAPI_L3'] > 0]
    print(socket_df[profile.basic_columns + ['PAPI_L1', 'PAPI_L2', 'PAPI_L3']].describe())

    socket_df = profile.events[:][profile.events['key'] == profile.event_types.PINS_L123['key']]
    print(socket_df[profile.basic_columns + ['PAPI_L1', 'PAPI_L2', 'PAPI_L3']
                ].describe().loc['count':'std',:] )

    # total_stats = SocketStats()
    # total_count = 0
    # total_name = None
    # for name, event_type in profile.event_types.iteritems():
    #     if name[:4] != 'PINS':
    #         total_count += event_type.stats.count
    # for name, event_type in profile.event_types.iteritems():
    #     if name in ['PINS_L123', 'PINS_SOCKET']:
    #         total_name = name
    #         for pkey, pstats in event_type.stats.socket_stats.iteritems():
    #             pstats.count = total_count
    #             total_stats += pstats
    # if total_stats.count > 0:
    #     total_printer = LinePrinter()
    #     total_printer.append(ItemPrinter(total_name, 'EVENT', length=17))
    #     total_printer.append(total_stats)
    #     total_printer.sorter = total_stats
    #     printers.append(total_printer)

    # printers.sort(key = lambda x: (x.sorter.name))
    # if len(printers) == 0:
    #     print('no events found matching your parameters')
    # else:
    #     prev_printer = printers[0]
    #     print(prev_printer.row_header())
    #     print(prev_printer.row())
    #     for printer in printers[1:]:
    #         if printer.row_header() != prev_printer.row_header():
    #             print(prev_printer.row_header())
    #             print('')
    #             print(printer.row_header())
    #             prev_printer = printer
    #         print(printer.row())
    #     print(printers[-1].row_header())
