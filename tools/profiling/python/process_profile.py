#!/usr/bin/python
# Peter Gaultney
# 2011.09.27
#
# meant to process .profile files into a format usable by gnuplot. e.g. calculate standard deviation, useful average, etc.

import sys
import os
import shutil
import re
import math
import glob
import online_math
import events

def processColumnarData(string, regex, sort=True):
    # get group count and initialize datasets
    datasets = []
    for i in range(regex.groups):
        datasets.append(dataSet())
    lines = string.split('\n')
    for line in lines:
        if len(line.lstrip()) != 0 and line.lstrip()[0] != '#':
            matchObj = regex.match(line)
            if matchObj:
                for i in range(1, regex.groups + 1):
                    datasets[i-1].append(float(matchObj.group(i)))
    # now we have our data....
    stats = []
    for set in datasets:
        # sort
        if sort:
            set.sort()
        # find std deviation
        variance, mean = online_variance_mean(set)
        set.stdDev = math.sqrt(variance)
        set.mean = mean
    # eliminate data points outside 2 std devs
    for set in datasets:
        set[:] = [x for x in set if (abs(x - set.mean) < (2 * set.stdDev))]
        # then recalculate mean
        set.mean = online_mean(set)
        print '# len = ' + str(len(set)) + '; stdDev = ' + str(set.stdDev) + '; mean = ' + str(set.mean)
        for i in xrange(len(set)):
            print str(i) + '\t' + str(set[i])

if __name__ == '__main__':
    sort = True
    if len(sys.argv) > 2 and sys.argv[2].startswith('s'):
        sort = True
    if os.path.isdir(sys.argv[1]):
        print 'isDIR!'
        profiles = glob.glob(sys.argv[1] + '/' + '*.dat')
        stdoutOrig = sys.stdout
        for profile in profiles:
            print profile
            sys.stdout = open(sys.argv[1] + '/' + os.path.basename(profile + '.data'), 'w')
            # get file from cmd line args
            linestring = open(profile, 'r').read()
            # read file into string
            processColumnarData(linestring, re.compile(r'\d+\s+\d+\s+GEMM\s+\d+\s+\d+\s+\d+\s+(\d+)'), sort)
            sys.stdout.close()
            sys.stdout = stdoutOrig
    elif os.path.isfile(sys.argv[1]):
        processColumnarData(open(sys.argv[1], 'r').read(), re.compile(r'\d+\s+\d+\s+GEMM\s+\d+\s+\d+\s+\d+\s+(\d+)'), sort)

