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
        trace = ptt.read(filename)
        print(trace.information)
        print(trace.event_types)
