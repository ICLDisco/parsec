#!/usr/bin/env python
from __future__ import print_function
import sys
import ptt_utils
import pbt2ptt

if __name__ == '__main__':
    if len(sys.argv[1:]) == 0:
        print("Usage: %s profile-file1 profile-file2 ..." % (sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    groups = ptt_utils.group_trace_filenames(sys.argv[1:])
    for f in groups:
        print("Processing %s" % f)
        name = pbt2ptt.convert(f)
        print("Generated: %s" % (name))
    sys.exit(0)
