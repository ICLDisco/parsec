#!/usr/bin/env python3

try:
    import os
    import numpy as np
    import time
    import pandas
    import sys
except ModuleNotFoundError:
    print("Did not find a system module, use pip to install it")

try:
    import parsec_trace_tables as ptt
    import pbt2ptt
except ModuleNotFoundError:
    print("Did not find pbt2ptt, you are likely using python version that does not match the version used to build PaRSEC profiling tools")
    print(sys.path)

import json
import re
import sys
import math

def bool(str):
    return str.lower() in ["true", "yes", "y", "1", "t"]

def h5_to_ctf(ptt_filename, ctf_filename, skip_parsec_events, skip_mpi_events):
    print(f"Converting {ptt_filename} into {ctf_filename}")

    ctf_data = {"traceEvents": []}

    trace = ptt.from_hdf(ptt_filename)

    for e in range(len(trace.events)):
        # print('id=',trace.events.id[e],' node_id=',trace.events.node_id[e],' stream_id=',trace.events.stream_id[e],'key=',trace.events.key[e],' type=',trace.events.type[e],' b=',trace.events.begin[e],' e=',trace.events.end[e])
        # print('\n')

        if(skip_parsec_events == True and trace.event_names[trace.events.type[e]].startswith("PARSEC")):
            continue
        if(skip_mpi_events == True and trace.event_names[trace.events.type[e]].startswith("MPI")):
            continue

        ctf_event = {}
        ctf_event["ph"] = "X"  # complete event type
        ctf_event["ts"] = 0.001 * trace.events.begin[e] # when we started, in ms
        ctf_event["dur"] = 0.001 * (trace.events.end[e] - trace.events.begin[e]) # when we started, in ms
        ctf_event["name"] = trace.event_names[trace.events.type[e]]

        if trace.events.key[e] is not None:
            ctf_event["args"] = trace.events.key[e].decode('utf-8').rstrip('\x00')
            ctf_event["name"] = trace.event_names[trace.events.type[e]]+"<"+ctf_event["args"]+">"

        ctf_event["pid"] = trace.events.node_id[e]
        tid = trace.streams.th_id[trace.events.stream_id[e]]
        ctf_event["tid"] = 111111 if math.isnan(tid) else int(tid)

        ctf_data["traceEvents"].append(ctf_event)

    with open(ctf_filename, "w") as chrome_trace:
        json.dump(ctf_data, chrome_trace)

if __name__ == "__main__":

    ptt_filename = sys.argv[1]
    ctf_file_name = sys.argv[2]
    skip_parsec_events = True
    skip_mpi_events = True
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        sys.exit("usage: h5toctf.py <ptt filename> <ctf filename> [skip PaRSEC events? default=1] [skip MPI events? default=1]")
    if len(sys.argv) >= 4:
        skip_parsec_events = bool(sys.argv[3])
    if len(sys.argv) >= 5:
        skip_mpi_events = bool(sys.argv[4])

    h5_to_ctf(ptt_filename, ctf_file_name, skip_parsec_events, skip_mpi_events)
