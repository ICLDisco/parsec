#!/usr/bin/env python
import parsec_profiling as p3
import sys, os, shutil, re # file utilities, etc.

def longest_substr(strings):
    substr = ''
    if len(strings) > 1 and len(strings[0]) > 0:
        for i in range(len(strings[0])):
            for j in range(len(strings[0])-i+1):
                if j > len(substr) and is_substr(strings[0][i:i+j], strings):
                    substr = strings[0][i:i+j]
    return substr

def is_substr(find, strings):
    if len(strings) < 1 and len(find) < 1:
        return False
    for i in range(len(strings)):
        if find not in strings[i]:
            return False
    return True

def rreplace(s, old, new, count = 1):
    return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]
