#!python

import pandas as pd
import os, sys

filename = sys.argv[1]
assert os.path.isfile(filename)
t = pd.HDFStore(filename)

try:
    FULL_ASYNC = t.event_types['FULL_ASYNC']
    FULL_RESCHED = t.event_types['FULL_RESCHED']
    ASYNC = t.event_types['async::ASYNC']
    RESCHED = t.event_types['async::RESCHED']
    STARTUP = t.event_types['async::STARTUP']
except AttributeError:
        print("HDF5 file {} does not contains the event_types attribute. The file might be empty, corrupted or it was incorrectly generated".format(filename), file=sys.stderr)
        sys.exit(2)
except KeyError:
    print("One of keys FULL_ASYNC, FULL_RESCHED, ASYNC, RESCHED, or STARTUP is not defined in the trace",
          file=sys.stderr)
    sys.exit(1)

try:
    NB = t.information['NB']
except KeyError:
    print("Global information NB is not defined, the trace is malformed",
          file=sys.stderr)
    sys.exit(1)

error = 0

try:
    if len(t.events[t.events.type == STARTUP]) != 0:
        print("Error: there should be no STARTUP tasks." +
              "because they were marked with property \"[profiling=off]\"",
              file=sys.stderr)
        error += 1
    else:
        startup = t.events[t.events.type == STARTUP].iloc[0]
        print("There is one STARTUP task.")

    if len(t.events[t.events.type == FULL_RESCHED]) != 1:
        print("Error: there should be exactly one FULL_RESCHED event.",
              file=sys.stderr)
        error += 1
    else:
        print("There is one FULL_RESCHED event")
        full_resched = t.events[t.events.type == FULL_RESCHED].iloc[0]

    if len(t.events[t.events.type == FULL_ASYNC]) != NB:
        print("Error: there should be exactly {} FULL_ASYNC events, there are {}"
              .format(NB, t.events[t.events.type == FULL_ASYNC]),
              file=sys.stderr)
        error += 1
    else:
        print("There are exactly {} FULL_ASYNC events".format(NB))

    if len(t.events[t.events.type == ASYNC]) != 2*NB:
        print("Error: there should be exactly {} ASYNC events, there are {}"
              .format(2*NB, len(t.events[t.events.type == ASYNC])),
              file=sys.stderr)
        error += 1
    else:
        print("There are exactly {} ASYNC events".format(2*NB))

    if error > 0:
        print("Errors when counting tasks... Cannot continue",
              file=sys.stderr)
        print("Trace file fails the tests",
              file=sys.stderr)
        sys.exit(1)
except AttributeError:
        print("HDF5 file {} does not contains the events attribute. The file might be empty, corrupted or it was incorrectly generated".format(filename), file=sys.stderr)
        sys.exit(2)
except KeyError:
    print("One of keys FULL_ASYNC, FULL_RESCHED, ASYNC, RESCHED, or STARTUP is not defined in the trace",
          file=sys.stderr)
    sys.exit(1)


nb_begin_before = 0
nb_end_after = 0
for index, e in t.events[t.events.type == RESCHED].iterrows():
    if e.begin < full_resched.begin:
        if nb_begin_before > 0:
            error += 1
            print("Error: more than one RESCHED event begins before the FULL_RESCHED begins",
                  file=sys.stderr)
        nb_begin_before += 1
        if e.end < full_resched.begin or e.end > full_resched.end:
            error += 1
            print("Error: the RESCHED event that begins before the FULL_RESCHED "
                  "one should end within the FULL_RESCHED one",
                  file=sys.stderr)
    if e.end > full_resched.end:
        if nb_end_after > 0:
            error += 1
            print("Error: more than one RESCHED event ends after the FULL_RESCHED ends",
                  file=sys.stderr)
        nb_end_after += 1
        if e.begin < full_resched.begin or e.begin > full_resched.end:
            error += 1
            print("Error: the RESCHED event that ends after the FULL_RESCHED "
                  "one should begin within the FULL_RESCHED one",
                  file=sys.stderr)
    for index2, e2 in t.events[t.events.type == RESCHED].iterrows():
        if index != index2 and e.begin <= e2.end and e2.begin <= e.end:
            error += 1
            print("Error: two RESCHED events overlap: [{}, {}] and [{}. {}]"
                  .format(e.begin, e.end, e2.begin, e2.end),
                  file=sys.stderr)
if error == 0:
    print("All the RESCHED events are mutually exclusive, and they all "
          "occur within the FULL_RESCHED event (except extremities)")

for k in range(NB):
    fa = t.events[((t.events.type == FULL_ASYNC) & (t.events.id == k))]
    if len(fa) != 1:
        print("Error: There should be a single FULL_ASYNC({}) event".format(k), file=sys.stderr)
        error += 1
    else:
        aes = t.events[((t.events.type == ASYNC) & (t.events.k == k))].sort_values('begin')
        if len(aes) != 2:
            print("Error: there should be exactly two FULL_ASYNC({}) events".format(k), file=sys.stderr)
            error += 1
        else:
            first = aes.iloc[0]
            second = aes.iloc[1]
            ref = fa.iloc[0]
            if first.end < ref.begin or first.end > ref.end:
                print("Error: first occurrence of ASYNC({}) should end within FULL_ASYNC({})".format(k, k),
                      file=sys.stderr)
                error += 1
            if second.begin < ref.begin or second.begin > ref.end:
                print("Error: second occurrence of ASYNC({}) should begin within FULL_ASYNC({})".format(k, k),
                      file=sys.stderr)
                error += 1

t.close()

if error == 0:
    print("Each of the ASYNC(0-{}) events occur exactly twice, and are within the corresponding FULL_ASYNC event"
          .format(NB-1), file=sys.stderr)
    print("Trace file passes the tests", file=sys.stderr)
else:
    print("Trace file fails the tests")

sys.exit(error)
