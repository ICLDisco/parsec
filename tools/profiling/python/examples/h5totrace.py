#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import pandas as pd
import paje
import argparse
import sys
import os
import re
import math
import time

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def print_event_names(store):
    maxlen = len("#rank")
    for en in store.event_names:
        if( maxlen < len(en) ):
            maxlen = len(en)

    line = ("{:<%s} "%(maxlen)).format("#rank")
    for n in range(0, store.information.nb_nodes):
        line += "%5d "%(n)
    print(line)

    for i in range(0, len(store.event_names)-1):
        en = store.event_names[i]
        line = ("{:>%d} "%(maxlen)).format(en)

        df = store.events[:][store.events.type==i].groupby('node_id').count()['begin']
        for n in range(0, len(df)):
            line += "%5d "%(df[n])
        print(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert an HDF5 profile file (and optionnaly a DOT file) into a PAJE trace')
    parser.add_argument('--h5', help='HDF5 input file', required=1, dest='input')
    parser.add_argument('--dot', help='DOT input file (should match the run of the HDF5 input file)', dest='dot')
    parser.add_argument('--out', help='Output file name', dest='output')
    parser.add_argument('--counter', help='Add a counter line per rank to represent the event names given as parameter (multiple allowed)',
                        action='append', dest='counters', nargs='+')
    parser.add_argument('--ignore-type', help='Set an event name to ignore (multiple allowed, can be a pattern with /pattern/)',
                        action='append', dest='type_ignore', nargs='+')
    parser.add_argument('--ignore-stream', help='Set a stream name to ignore (multiple allowed, can be a pattern with /pattern/)',
                        action='append', dest='stream_ignore', nargs='+')
    parser.add_argument('--dot-DAG', help='Include links of the DAG obtained from the DOT file', dest='DAG',
                        action='store_true', default=False)
    parser.add_argument('--COMM', help='Include links of communications', dest='COMM',
                        action='store_true', default=False)
    parser.add_argument('--list', help='List the events names in the hdf5 file', action='store_true', dest='list')
    args = parser.parse_args()

    store = pd.HDFStore(args.input)
    if( not 'events' in store or not 'streams' in store or not 'event_names' in store ):
        warning('This HDF5 store does not hold a PaRSEC profile')
        sys.exit(1)

    if( args.list ):
        print_event_names(store)
        sys.exit(0)

    if( None == args.output ):
        warning("Output file not specified, and --list not specified")
        args.print_help()
        sys.exit(1)

    try:
        os.unlink("%s.trace" % (args.output))
    except OSError:
        pass

    task_names = dict()
    if args.DAG:
        task_dot_id = dict()
        dot_links = dict()
    if args.dot:
        try:
            dotfile = open(args.dot, 'r')
            for line in dotfile:
                ls = line.find("label=\"")
                ts = line.find("tooltip=\"")
                ne = line.find(" ")
                if ls >= 0 and ts >= 0:
                    label=line[ls+7:-1]
                    le = label.find("\",")
                    assert le >= 1
                    label = label[0:le]

                    tooltip=line[ts+9:-1]
                    te = tooltip.find("\"")
                    assert te >= 1
                    tooltip = tooltip[0:te]

                    ttfields = tooltip.split(':')

                    task_uid = "%s:%s:%s"%(ttfields[0], ttfields[1], ttfields[3])
                    task_names[task_uid] = label
                    if args.DAG:
                        task_dot_id[ line[0:ne] ] = task_uid
                        dot_links[ line[0:ne] ] = list()
                elif args.DAG:
                    ts = line.find(" -> ")
                    ls = line.find("label")
                    if ts >= 0 and ls >= 0:
                        dot_links[ line[0:ne] ].append( line[ts + 4:ls - 2] )
            dotfile.close()
        except IOError as e:
            warning("Could not open %s: %s"% (args.dot, e))

    paje.startTrace(args.output)

    PajeContainerType = paje.PajeDef('PajeDefineContainerType')
    PajeStateType = paje.PajeDef('PajeDefineStateType')
    PajeContainerCreate = paje.PajeDef('PajeCreateContainer')
    PajeSetState = paje.PajeDef('PajeSetState', task_name='string')
    PajeEntityValue = paje.PajeDef('PajeDefineEntityValue')
    PajeDefineVariableType = paje.PajeDef('PajeDefineVariableType')
    PajeAddVariable = paje.PajeDef('PajeAddVariable')
    PajeSubVariable = paje.PajeDef('PajeSubVariable')
    PajeStartLink = paje.PajeDef('PajeStartLink')
    PajeEndLink   = paje.PajeDef('PajeEndLink')
    PajeLinkType  = paje.PajeDef('PajeDefineLinkType')

    paje_ct = PajeContainerType.PajeEvent(Name="Application", Type="0")
    paje_pt = PajeContainerType.PajeEvent(Name="Process", Type=paje_ct)
    paje_vt = PajeContainerType.PajeEvent(Name="VP", Type=paje_pt)
    paje_tt = PajeContainerType.PajeEvent(Name="Stream", Type=paje_vt)
    paje_st = PajeStateType.PajeEvent(Name="CT_ST", Type=paje_tt)
    paje_vt = PajeDefineVariableType.PajeEvent(Name="CT_VT", Type=paje_tt, Color="1.0,1.0,1.0")

    if args.DAG:
        paje_lslt = PajeLinkType.PajeEvent(Name="local_DAG_LINK", Type=paje_ct, StartContainerType=paje_tt, EndContainerType=paje_tt)
        paje_rslt = PajeLinkType.PajeEvent(Name="remote_DAG_LINK", Type=paje_ct, StartContainerType=paje_tt, EndContainerType=paje_tt)
    if args.COMM:
        paje_lcom = PajeLinkType.PajeEvent(Name="MPI_Comm", Type=paje_ct, StartContainerType=paje_tt, EndContainerType=paje_tt)
        paje_sndl = PajeLinkType.PajeEvent(Name="Task_to_MPI", Type=paje_ct, StartContainerType=paje_tt, EndContainerType=paje_tt)
        paje_rcvl = PajeLinkType.PajeEvent(Name="MPI_to_Task", Type=paje_ct, StartContainerType=paje_tt, EndContainerType=paje_tt)

    paje_entity_waiting = PajeEntityValue.PajeEvent(Name="Waiting", Type=paje_st, Color="0.2,0.2,0.2")

    paje_container_aliases = dict()
    container_endstate = dict()

    state_aliases = dict()
    for x in range(len(store.event_names)-1):
        color_code = int(store.event_attributes[x], 16)
        r = (color_code >> 16) & 0xFF
        g = (color_code >>  8) & 0xFF
        b = (color_code      ) & 0xFF
        state_aliases[x] = PajeEntityValue.PajeEvent(Name=store.event_names[x], Type=paje_st, Color="%g,%g,%g" % (r/255.0, g/255.0, b/255.0))

    paje_c_appli = PajeContainerCreate.PajeEvent(Name=store.information.exe, Time=0.000, Type=paje_ct, Container="0")

    types_to_count = list()
    counter_events = dict()
    if args.counters:
        args.counters = [item for sublist in args.counters for item in sublist]
        for c in args.counters:
            index=store.event_names[store.event_names.str.contains('^%s$'%(c))].index.tolist()
            if len(index)==1:
                i = store.event_names[index[0]]
                types_to_count.append(index[0])
                counter_events[i] = dict()
                counter_events[i]['events'] = list()
                counter_events[i]['error'] = False
                counter_events[i]['type_name'] = c
                for n in range(0, store.information.nb_nodes):
                    name = "M%d-%s"%(n,c)
                    if "M%d"%(n) not in paje_container_aliases:
                        paje_container_aliases["M%d"%(n)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%d" % (n),
                                                                                          Type=paje_ct, Container=paje_c_appli)
                    paje_container_aliases[name] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = name, Type=paje_ct,
                                                                                Container=paje_container_aliases["M%d"%(n)])
            else:
                warning('%s is not a valid event name (run with --list to get the valid event names).'%(c))

    types_to_ignore = list()
    if args.type_ignore:
        args.type_ignore = [item for sublist in args.type_ignore for item in sublist]
        for c in args.type_ignore:
            p = re.search('^\/(.*)\/$', c)
            if p:
                pattern = p.group(1)
            else:
                pattern = "^%s$"%(c)
            index=store.event_names[store.event_names.str.contains(pattern)].index.tolist()
            types_to_ignore += index

    streams_to_ignore = list()
    if args.stream_ignore:
        args.stream_ignore = [item for sublist in args.stream_ignore for item in sublist]
        for c in args.stream_ignore:
            p = re.search('^\/(.*)\/$', c)
            if p:
                pattern = p.group(1)
            else:
                pattern = "^%s$"%(c)
            index=store.streams[store.streams.description.str.contains(pattern)].index.tolist()
            streams_to_ignore += index
            print(streams_to_ignore)

    for trindex, t in store.streams.iterrows():
        if trindex in streams_to_ignore:
            continue
        if "M%d"%(t.node_id) not in paje_container_aliases:
            paje_container_aliases["M%d"%(t.node_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%d" % (t.node_id), Type=paje_ct, Container=paje_c_appli)
        if hasattr(t, 'vp_id'):
            if not math.isnan(t.vp_id):
                if "M%dV%d"%(t.node_id, t.vp_id) not in paje_container_aliases:
                    paje_container_aliases["M%dV%d"%(t.node_id,t.vp_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dV%d" % (t.node_id,t.vp_id), Type=paje_ct,
                                                                                                        Container=paje_container_aliases["M%d"%(t.node_id)])
        match = re.search(r'PaRSEC', t.description)
        if match is not None and "M%dT%d"%(t.node_id, t.stream_id) not in paje_container_aliases:
                if hasattr(t, 'vp_id') and isinstance(t.vp_id, int):
                        paje_container_aliases["M%dT%d"%(t.node_id,t.stream_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dT%d" % (t.node_id,t.stream_id), Type=paje_ct,
                                                                                                                Container=paje_container_aliases["M%dV%d"%(t.node_id,t.vp_id)])
                else:
                        paje_container_aliases["M%dT%d"%(t.node_id,t.stream_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dT%d" % (t.node_id,t.stream_id), Type=paje_ct,
                                                                                                                Container=paje_container_aliases["M%d"%(t.node_id)])
                container_endstate["M%dT%d"%(t.node_id,t.stream_id)] = 0.0
        else:
            match = re.search(r'GPU\ ([0-9]+)\-([0-9]+)', t.description)
            if match is not None:
                gpu_id = int(match.group(1))
                if "M%dGPU%d"%(t.node_id,gpu_id) not in paje_container_aliases:
                    paje_container_aliases["M%dGPU%d"%(t.node_id,gpu_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dGPU%d"%(t.node_id, gpu_id), Type=paje_ct,
                                                                                                          Container=paje_container_aliases["M%d"%(t.node_id)])
                if "M%dT%d"%(t.node_id, t.stream_id) not in paje_container_aliases:
                    paje_container_aliases["M%dT%d"%(t.node_id,t.stream_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dT%d"%(t.node_id,t.stream_id), Type=paje_ct,
                                                                                                             Container=paje_container_aliases["M%dGPU%d"%(t.node_id,gpu_id)])
                    container_endstate["M%dT%d"%(t.node_id,t.stream_id)] = 0.0
            else:
                match = re.search(r'MPI.*', t.description)
                if match is not None:
                    if "M%dMPI"%(t.node_id) not in paje_container_aliases:
                        paje_container_aliases["M%dMPI"%(t.node_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dMPI" % (t.node_id),
                                                                                                     Type=paje_ct, Container=paje_container_aliases["M%d"%(t.node_id)])
                        container_endstate["M%dMPI"%(t.node_id)] = 0.0
                else:
                    print("Found unknown stream description '%s' at stream_id %d"%(t.description, trindex))
                    if "M%dUnknown"%(t.node_id) not in paje_container_aliases:
                        paje_container_aliases["M%dUnknown"%(t.node_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dUnknown"%(t.node_id), Type=paje_ct,
                                                                                                        Container=paje_container_aliases["M%d"%(t.node_id)])
                        container_endstate["M%dUnknown"%(t.node_id)] = 0.0
                    if "M%dT%d"%(t.node_id, t.stream_id) not in paje_container_aliases:
                        paje_container_aliases["M%dT%d"%(t.node_id, t.stream_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dT%d" % (t.node_id,t.stream_id),
                                                                                                                Type=paje_ct, Container=paje_container_aliases["M%dUnknown"%(t.node_id)])
                        container_endstate["M%dT%d"%(t.node_id, t.stream_id)] = 0.0
                        PajeSetState.PajeEvent(Time=0.000, Type=paje_st, Container=paje_container_aliases["M%dT%d"%(t.node_id,t.stream_id)], Value="Waiting", task_name="")

    if args.DAG:
        dag_info = dict()

    progress = 0.0
    delta = 1.0 / len(store.events)
    for evr in store.events.iterrows():
        update_progress( progress )
        progress = progress + delta
        ev = evr[1]
        if (ev['type'] in types_to_ignore):
            continue
        if (ev['type'] in types_to_count):
            type_name = store.event_names[ ev['type'] ]
            info_name = '%s_start'%(type_name)
            if( (int(ev['flags']) & (1<<2)) != 0) and ('size' in ev[info_name]):
                counter_events[type_name]['events'].append( {'time': float(ev.begin), 'size':float(ev[info_name ]['size']), 'node_id': ev.node_id } )
                counter_events[type_name]['events'].append( {'time': float(ev.end), 'size': -1.0 * float(ev[info_name ]['size']), 'node_id': ev.node_id } )
            else:
                counter_events[type_name]['events'].append( {'time': float(ev.begin), 'size':1.0, 'node_id': ev.node_id } )
                counter_events[type_name]['events'].append( {'time': float(ev.end),  'size':-1.0, 'node_id': ev.node_id } )
#            elif not counter_events[ev['type']]['error']:
#                counter_events[type_name]['error'] = True
#                warning('You requested to use counters for events of type %s, but such events are not marked as counter-types'%(type_name))
        else:
            #Don't forget to check if that container was ignored by the user
            if ( (int(ev['flags']) & (1<<2)) == 0) and ("M%dT%d"%(ev.node_id,ev.stream_id) in paje_container_aliases):
                if ev['end'] <= container_endstate["M%dT%d"%(ev.node_id,ev.stream_id)]:
                    #This event is entirely under the current event on that stream: skip it
                    continue
                if ev['begin'] < container_endstate["M%dT%d"%(ev.node_id,ev.stream_id)]:
                    begin_date = container_endstate["M%dT%d"%(ev.node_id,ev.stream_id)]
                else:
                    begin_date = ev['begin']
                key = "tpid=%d:tcid=%d:tid=%d"%(ev.taskpool_id,ev.tcid,ev.id)
                if args.DAG:
                    dag_info[key] = { 'container': paje_container_aliases["M%dT%d"%(ev.node_id,ev.stream_id)],
                                      'start': float(ev.begin), 'end': float(ev.end), 'rank': ev.node_id }
                if key in task_names.keys():
                    PajeSetState.PajeEvent(Time=float(begin_date), Type=paje_st, Container=paje_container_aliases["M%dT%d"%(ev.node_id,ev.stream_id)],
                                           Value=state_aliases[ev.type], task_name=task_names[key])
                else:
                    PajeSetState.PajeEvent(Time=float(begin_date), Type=paje_st, Container=paje_container_aliases["M%dT%d"%(ev.node_id,ev.stream_id)],
                                           Value=state_aliases[ev.type], task_name=store.event_names[ev.type])
                PajeSetState.PajeEvent(Time=float(ev.end), Type=paje_st, Container=paje_container_aliases["M%dT%d"%(ev.node_id,ev.stream_id)],
                                       Value=paje_entity_waiting, task_name="Waiting")
                container_endstate["M%dT%d"%(ev.node_id,ev.stream_id)] = ev.end

    if args.COMM:
        try:
            snd_type = store.event_types['MPI_DATA_PLD_SND']
            rcv_type = store.event_types['MPI_DATA_PLD_RCV']
        except KeyError:
            warning("You requested to display communication, but no MPI event types were logged in this profile")
        else:
            sends = store.events[ store.events.type == snd_type ]
            nblink = 0
            for sendr in sends.iterrows():
                send = sendr[1]
                PajeSetState.PajeEvent(Time=float(send.begin), Type=paje_st, Container=paje_container_aliases["M%dMPI"%(send.node_id)],
                                           Value=state_aliases[send.type], task_name=store.event_names[send.type])
                PajeSetState.PajeEvent(Time=float(send.end), Type=paje_st, Container=paje_container_aliases["M%dMPI"%(send.node_id)],
                                           Value=paje_entity_waiting, task_name="Waiting")
                recvs = store.events[ ( (store.events.type == rcv_type) &
                                        (store.events.tcid  == send['tcid']) &
                                        (store.events.tpid  == send['tpid']) &
                                        (store.events.tid   == send['tid']) ) ]
                for rrecv in recvs.iterrows():
                    recv = rrecv[1]
                    PajeSetState.PajeEvent(Time=float(recv.begin), Type=paje_st, Container=paje_container_aliases["M%dMPI"%(recv.node_id)],
                                           Value=state_aliases[recv.type], task_name=store.event_names[recv.type])
                    PajeSetState.PajeEvent(Time=float(recv.end), Type=paje_st, Container=paje_container_aliases["M%dMPI"%(recv.node_id)],
                                           Value=paje_entity_waiting, task_name="Waiting")
                    PajeStartLink.PajeEvent(Time=float(send.begin), Type=paje_lcom, Container=paje_c_appli,
                                            StartContainer = paje_container_aliases["M%dMPI"%(send.src)],
                                            Value = "", Key="%d"%(nblink))
                    PajeEndLink.PajeEvent(Time=float(recv.end), Type=paje_lcom, Container=paje_c_appli,
                                          EndContainer = paje_container_aliases["M%dMPI"%(recv.dst)],
                                          Value = "", Key="%d"%(nblink))
                    nblink = nblink + 1

    if args.DAG:
        nblink = 0
        for src, dstlist in dot_links.iteritems():
            for dst in dstlist:
                try:
                    src_uid = task_dot_id[src]
                    dst_uid = task_dot_id[dst]
                except KeyError as e:
                    print("couldn't find %s in task_dot_id"%(e))
                    pass
                try:
                    src_info = dag_info[src_uid]
                    dst_info = dag_info[dst_uid]
                except KeyError as e:
                    print("couldn't find %s in dag_info"%(e))
                    pass
                if src_info['rank'] == dst_info['rank']:
                    link_type=paje_lslt
                else:
                    link_type=paje_rslt
                PajeStartLink.PajeEvent(Time=src_info['end'], Type=link_type, Container=paje_c_appli, StartContainer=src_info['container'], Value="", Key="%d"%(nblink))
                PajeEndLink.PajeEvent(Time=dst_info['start'], Type=link_type, Container=paje_c_appli, EndContainer=dst_info['container'], Value="", Key="%d"%(nblink))
                nblink = nblink+1

    for name,cer in counter_events.items():
        if not cer['error']:
            ce = cer['events']
            ce.sort(cmp = lambda x, y: cmp(x['time'], y['time']) )
            for e in ce:
                #Check if that container was ignored by the user
                if "M%d-%s"%(e['node_id'],cer['type_name']) in paje_container_aliases:
                    PajeAddVariable.PajeEvent(Time=float(e['time']), Type=paje_vt,
                                              Container=paje_container_aliases["M%d-%s"%(e['node_id'],cer['type_name'])],
                                              Value=e['size'])

    paje.endTrace();

    sys.exit(0)

