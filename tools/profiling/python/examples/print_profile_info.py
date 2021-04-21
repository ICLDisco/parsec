#!/usr/bin/env python

from __future__ import print_function
from parsec_trace_tables import *
import pbt2ptt as ptt
import os, sys

if __name__ == '__main__':
    filenames = []
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            filenames.append(arg)

    for filename in filenames:
        if is_pbt(filename):
            trace = ptt.read(filename)
        else:
            trace = from_hdf(filename)
        if trace is not None:
            print(trace.information)
            print(trace.event_types)
