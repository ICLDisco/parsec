#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def cond_print(string, cond, **kwargs):
    if cond:
        print(string, **kwargs)

def longest_substr(strings):
    substr = ''
    if len(strings) > 1 and len(strings[0]) > 0:
        for i in range(len(strings[0])):
            for j in range(len(strings[0])-i+1):
                if j > len(substr) and is_substr(strings[0][i:i+j], strings):
                    substr = strings[0][i:i+j]
        return substr
    elif len(strings) == 1:
        return strings[0]
    else:
        return ''

def is_substr(find, strings):
    if len(strings) < 1 and len(find) < 1:
        return False
    for i in range(len(strings)):
        if find not in strings[i]:
            return False
    return True

def rreplace(s, old, new, count = 1):
    return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]

def safe_unlink(files, report_error = True):
    for ufile in files:
        try:
            print('unlinking', ufile)
            os.unlink(ufile) # no need to have them hanging around anymore
        except OSError:
            if report_error:
                print('the file {} has apparently vanished.'.format(ufile))

def smart_parse(arg, conv=int):
    if arg is None:
        return None
    if not isinstance(arg, str):
        if len(arg) > 1:
            return arg # already an interesting list
        else:
            arg = arg[0] #

    if ',' in arg:
        lst = [conv(x) for x in arg.split(',')]
    elif ':' in arg:
        splits = arg.split(':')
        if len(splits) == 2:
            start, stop = splits
            step = 1
        elif len(splits) == 3:
            start, stop, step = splits
        lst = xrange(int(start), int(stop), int(step))
    else:
        lst = [conv(arg)]
    return lst


def match_dicts(dicts):
    """ Returns the matching or compatible parts of multi-type dictionaries.

    Only matching keys and values will be retained, except:
    Matching keys with float values will be averaged.

    Retains the actual type of the items passed, assuming they are
    all the same type of dictionary-like object."""

    if len(dicts) == 0:
        return dict()

    matched_info = dicts[0]
    mult = 1.0 / len(dicts)
    for dict_ in dicts[1:]:
        for key, value in dict_.iteritems():
            if key not in matched_info: # not present
                matched_info.drop(key)
            elif value != matched_info[key]:
                try:
                    temp_fl = float(value)
                    if '.' in str(value): # if number was actually a float
                        # do average
                        if trace == p_set[1]:
                            matched_info[key] = matched_info[key] * mult
                        matched_info[key] += value * mult
                    else: # not float
                        matched_info.drop(key)
                except: # not equal and not float
                    matched_info.drop(key)
    return matched_info

