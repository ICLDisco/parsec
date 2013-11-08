#!/usr/bin/env python

from __future__ import print_function
from parsec_profiling import *
import parsec_binprof as p3_bin
import p3_group_profiles as p3_g
import os, sys

if __name__ == '__main__':
    filenames = []
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            filenames.append(arg)
    for filename in filenames:
        profile = p3_bin.get_info(filename)
        print(profile.information)
        print(profile.event_types)
