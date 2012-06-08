#!/usr/bin/python
import re

defaultProfileRegex = r'(\d+)\s+(\d+)\s+([\w\s]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*'
defaultRegex = re.compile(defaultProfileRegex)

class Event(object):
    def __init__(self, procID, tID, key, ID, start, end, duration):
        self.pid = int(procID)
        self.tid = int(tID)
        self.key = key
        self.id = int(ID)
        self.start = int(start)
        self.end = int(end)
        self.duration = int(duration)

def parse_events(file, regex = defaultRegex):
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
    return events

