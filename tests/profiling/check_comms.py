from __future__ import print_function
import pandas as pd
import sys

t = pd.HDFStore('bw.h5')

result = {
    'MPI_ACTIVATE': { 'nb': 100, 'lensum': 12000 },
    'MPI_DATA_CTL': { 'nb': 100, 'lensum': 2400 },
    'MPI_DATA_PLD_SND': { 'nb': 100, 'lensum': 209715200 },
    'MPI_DATA_PLD_RCV': { 'nb': 100, 'lensum': 209715200 }
}

cmdline = 'mpirun -np 2 bw_test -n 10 -f 10 -l 2097152'

for mt in result.keys():
    try:
        evs = t.events[ t.events.type == t.event_types[mt] ]
        if len(evs) != result[mt]['nb']:
            print("Found {} events of type {}; expected 100. Did you run {}?".
                  format(len(evs), mt, cmdline),
                  file=sys.stderr)
            sys.exit(1)
        else:
            print("Found {} events of type {} -- correct".format(len(evs), mt))
        try:
            if evs['msg_size'].sum() != result[mt]['lensum']:
                print("Unexpected sum of length for events of type {}. Expected {}, found {}."
                      "Did you run {}?".format(mt, result[mt]['lensum'], evs['msg_size'].sum(), cmdline))
                sys.exit(1)
            else:
                print("Sum of msg_size of events of type {} is {} -- correct".format(mt, result[mt]['lensum']))
        except KeyError:
            print("Column 'msg_size' is not defined in this trace, something went wrong.")
            sys.exit(1)
    except KeyError:
        print("Key {} is not present in the trace. "
              "You are using a different communication system or something went wrong".format(mt),
              file=sys.stderr)
        sys.exit(1)

print("Test passed for all communication event types")
sys.exit(0)
