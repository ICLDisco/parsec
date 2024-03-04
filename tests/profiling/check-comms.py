
import pandas as pd
import os, sys

filename = 'bw.h5'
assert os.path.isfile(filename)
t = pd.HDFStore(filename)

result = {
    'MPI_ACTIVATE': { 'nb': 100, 'lensum': 12000 },
    'MPI_DATA_CTL': { 'nb': 100, 'lensum': 209715200 },
    'MPI_DATA_PLD_SND': { 'nb': 100, 'lensum': 209715200 },
    'MPI_DATA_PLD_RCV': { 'nb': 100, 'lensum': 209715200 }
}

cmdline = 'mpirun -np 2 bw_test -n 10 -f 10 -l 2097152'

def fatal_error(msg):
    print(msg, file=sys.stderr)
    print("*** This test depends highly on the internal implementation of the communication system\n" +
          "*** and on the command line used to test. Did you use " + cmdline + "\n" +
          "*** Failures may indicate that control message sizes have changed, or the way message sizes are\n" +
          "*** accounted for in the communication engine has changed. If this test starts to fail after\n" +
          "*** a change in the communication system, the test should be updated to reflect the new numbers.",
          file=sys.sderr)
    sys.exit(1)

for mt in list(result.keys()):
    try:
        evs = t.events[ t.events.type == t.event_types[mt] ]
        if len(evs) != result[mt]['nb']:
            fatal_error(f"Found {len(evs)} events of type {mt}; expected 100.")
        else:
            print(f"Found {len(evs)} events of type {mt} -- correct")
        try:
            if evs['msg_size'].sum() != result[mt]['lensum']:
                fatal_error(f"Unexpected sum of length for events of type {mt}. Expected {result[mt]}, found {evs['msg_size'].sum()}.")
            else:
                print(f"Sum of msg_size of events of type {mt} is {result[mt]['lensum']} -- correct")
        except KeyError:
            print("Column 'msg_size' is not defined in this trace, something went wrong.")
            sys.exit(1)
    except AttributeError:
        print("HDF5 file {} does not contains the events or event_types attribute. The file might be empty, corrupted or it was incorrectly generated".format(filename), file=sys.stderr)
        sys.exit(2)
    except KeyError:
        fatal_error(f"Key {mt} is not present in the trace. You are using a different communication system or something went wrong")

print("Test passed for all communication event types")
sys.exit(0)
