#!env python2.7

from __future__ import print_function

import numpy as np
import pandas as pd
import paje
import argparse
import sys
import os
import re

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
    parser.add_argument('--ignore-thread', help='Set a thread name to ignore (multiple allowed, can be a pattern with /pattern/)',
                        action='append', dest='thread_ignore', nargs='+')
    parser.add_argument('--list', help='List the events names in the hdf5 file', action='store_true', dest='list')
    args = parser.parse_args()

    store = pd.HDFStore(args.input)
    if( not 'events' in store or not 'threads' in store or not 'event_names' in store ):
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
    if args.dot:
        try:
            dotfile = open(args.dot, 'r')
            for line in dotfile:
                ls = line.find("label=\"")
                ts = line.find("tooltip=\"")
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
                    task_names["%s:%s:%s"%(ttfields[0], ttfields[1], ttfields[3])] = label
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

    paje_ct = PajeContainerType.PajeEvent(Name="Application", Type="0")
    paje_pt = PajeContainerType.PajeEvent(Name="Process", Type=paje_ct)
    paje_vt = PajeContainerType.PajeEvent(Name="VP", Type=paje_pt)
    paje_tt = PajeContainerType.PajeEvent(Name="Thread", Type=paje_vt)
    paje_st = PajeStateType.PajeEvent(Name="CT_ST", Type=paje_tt)
    paje_vt = PajeDefineVariableType.PajeEvent(Name="CT_VT", Type=paje_tt, Color="1.0,1.0,1.0")

    paje_entity_waiting = PajeEntityValue.PajeEvent(Name="Waiting", Type=paje_st, Color="0.2,0.2,0.2")

    paje_container_aliases = dict()
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

    threads_to_ignore = list()
    if args.thread_ignore:
        args.thread_ignore = [item for sublist in args.thread_ignore for item in sublist]
        for c in args.thread_ignore:
            p = re.search('^\/(.*)\/$', c)
            if p:
                pattern = p.group(1)
            else:
                pattern = "^%s$"%(c)
            index=store.threads[store.threads.description.str.contains(pattern)].index.tolist()
            threads_to_ignore += index
            print(threads_to_ignore)

    for trindex, t in store.threads.iterrows():
        if trindex in threads_to_ignore:
            continue
        if "M%d"%(t.node_id) not in paje_container_aliases:
            paje_container_aliases["M%d"%(t.node_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%d" % (t.node_id), Type=paje_ct, Container=paje_c_appli)
        if isinstance(t.vp_id, int):
            if "M%dV%d"%(t.node_id, t.vp_id) not in paje_container_aliases:
                paje_container_aliases["M%dV%d"%(t.node_id,t.vp_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dV%d" % (t.node_id,t.vp_id), Type=paje_ct,
                                                                                                     Container=paje_container_aliases["M%d"%(t.node_id)])
            if "M%dT%d"%(t.node_id, t.thread_id) not in paje_container_aliases:
                paje_container_aliases["M%dT%d"%(t.node_id,t.thread_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dT%d" % (t.node_id,t.thread_id), Type=paje_ct,
                                                                                                         Container=paje_container_aliases["M%dV%d"%(t.node_id,t.vp_id)])
        else:
            if "M%dT%d"%(t.node_id, t.thread_id) not in paje_container_aliases:
                paje_container_aliases["M%dT%d"%(t.node_id, t.thread_id)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dT%d" % (t.node_id,t.thread_id),
                                                                                                          Type=paje_ct, Container=paje_container_aliases["M%d"%(t.node_id)])
        PajeSetState.PajeEvent(Time=0.000, Type=paje_st, Container=paje_container_aliases["M%dT%d"%(t.node_id,t.thread_id)], Value="Waiting", task_name="")

    sev = store.events.sort(['node_id', 'thread_id', 'begin'])
    for evr in store.events.iterrows():
        ev = evr[1]
        if (ev['type'] in types_to_ignore):
            continue
        if (ev['type'] in types_to_count):
            type_name = store.event_names[ ev['type'] ]
            info_name = '%s_start'%(type_name)
            if(ev['flags'] & (1<<2) != 0) and ('size' in ev[info_name]):
                counter_events[type_name]['events'].append( {'time': float(ev.begin), 'size':float(ev[info_name ]['size']), 'node_id': ev.node_id } )
                counter_events[type_name]['events'].append( {'time': float(ev.end), 'size': -1.0 * float(ev[info_name ]['size']), 'node_id': ev.node_id } )
            elif not counter_events[ev['type']]['error']:
                counter_events[type_name]['error'] = True
                warning('You requested to use counters for events of type %s, but such events are not marked as counter-types'%(type_name))
        else:
            #Don't forget to check if that container was ignored by the user
            if (ev['flags'] & (1<<2) == 0) and ("M%dT%d"%(ev.node_id,ev.thread_id) in paje_container_aliases):
                key = "hid=%d:did=%d:tid=%d"%(ev.handle_id,ev.type,ev.id)
                if key in task_names.keys():
                    PajeSetState.PajeEvent(Time=float(ev.begin), Type=paje_st, Container=paje_container_aliases["M%dT%d"%(ev.node_id,ev.thread_id)],
                                           Value=state_aliases[ev.type], task_name=task_names[key])
                else:
                    PajeSetState.PajeEvent(Time=float(ev.begin), Type=paje_st, Container=paje_container_aliases["M%dT%d"%(ev.node_id,ev.thread_id)],
                                           Value=state_aliases[ev.type], task_name=store.event_names[ev.type])
                PajeSetState.PajeEvent(Time=float(ev.end), Type=paje_st, Container=paje_container_aliases["M%dT%d"%(ev.node_id,ev.thread_id)],
                                       Value=paje_entity_waiting, task_name="Waiting")

    for name,cer in counter_events.iteritems():
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

