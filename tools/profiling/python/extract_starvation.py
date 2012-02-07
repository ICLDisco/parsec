#!/usr/bin/python
# USAGE:
# ./extract_starvation.py profiling.data
#
# requires Python 2.5 or greater because of multiple-attribute attrgetter
# as per http://docs.python.org/library/operator.html
import sys
import os
import re
from operator import attrgetter
import glob

defaultProfileRegex = r'(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*'

class Event(object):
    def __init__(self, procID, tID, key, ID, start, end, duration):
        self.procID = int(procID)
        self.tID = int(tID)
        self.key = key
        self.ID = int(ID)
        self.start = int(start)
        self.end = int(end)
        self.duration = int(duration)

# it is advisable to redirect sys.stdout before calling this method
# maybe use something like StringIO to make this work for big strings as well as files
def extractStarvation(file, regex):
    events = []
    for line in file:
        if line.startswith('#'):
            continue # skip
        match = regex.match(line)
        if match:
            ev = Event(match.group(1), match.group(2), match.group(3), match.group(4),
                       match.group(5), match.group(6), match.group(7))
            events.append(ev)
        else:
            print(line + ' not parsed')

    events.sort(key=attrgetter('procID', 'tID', 'start'))
            
    prevEv = None
    starvation = 0
    starve_sets = []
    for event in events:
        if prevEv is None:
            prevEv = event
            continue
        if (event.procID == prevEv.procID and event.tID == prevEv.tID):
            starvation += event.start - prevEv.end
        else:
            starve_sets.append((prevEv.procID, prevEv.tID, starvation))
            starvation = 0
        prevEv = event
    starve_sets.append((prevEv.procID, prevEv.tID, starvation))

    totalStarvation = 0
    for starve in starve_sets:
        procID, tID, time = starve
        print('procID: %6d tID: %6d starvation: %d' % (starve))    
        totalStarvation += time
    print('total starvation: {0} average thread starvation {1}'.format(
            totalStarvation, totalStarvation/len(starve_sets)))
    return totalStarvation/len(starve_sets)

if __name__ == '__main__':
    profile = re.compile(defaultProfileRegex)
    if os.path.isdir(sys.argv[1]):
        identifier = '*.dat'
        if len(sys.argv) > 2:
            identifier = sys.argv[2]
            print('Using identifier {0}'.format(identifier))
        dats = glob.glob(sys.argv[1] + os.sep + identifier)
        totalOfStarvationAverages = 0
        for dat in dats:
            totalOfStarvationAverages += extractStarvation(open(dat, 'r'), profile)
        if len(dats) > 0:
            print('Aggregate Average Starvation: {0}'.format(totalOfStarvationAverages / len(dats)))
        else:
            print('No files found to match identifier {0}'.format(identifier))
    else:
        extractStarvation(open(sys.argv[1], 'r'), profile)
