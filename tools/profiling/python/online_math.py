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

class dataSet(list):
    stdDev = 0.0
    mean = 0.0

# Welford's variance algorithm, via Knuth, via Wikipedia
def online_variance_mean(data):  
    n = 0
    mean = 0.0
    M2 = 0.0
    variance = 0.0
    for x in data:
        n = n + 1
        delta = x - mean
        mean = mean + delta/n
        M2 = M2 + delta*(x - mean)  # This expression uses the new value of mean
    if n > 0:
        variance_n = M2/n
	variance = M2/(n - 1)
    else:
        print ("empty data set")
        from traceback import print_tb
        print_tb(sys.last_traceback)
    return variance, mean

def online_mean(data):
    n = 0
    mean = 0
    for x in data:
        n += 1
        delta = x - mean
        mean = mean + delta/n
    return mean

