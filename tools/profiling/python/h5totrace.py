#!env python2.7

from __future__ import print_function

import numpy as np
import pandas as pd
import paje
import argparse
import sys
import os

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert an HDF5 profile file (and optionnaly a DOT file) into a PAJE trace')
    parser.add_argument('--h5', help='HDF5 input file', required=1, dest='input')
    parser.add_argument('--dot', help='DOT input file (should match the run of the HDF5 input file)', dest='dot')
    parser.add_argument('--out', help='Output file name', required=1, dest='output')
    args = parser.parse_args()

    store = pd.HDFStore(args.input)
    if( not 'events' in store or not 'threads' in store or not 'event_names' in store ):
        warning('This HDF5 store does not hold a PaRSEC profile')
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

    paje_ct = PajeContainerType.PajeEvent(Name="Application", Type="0")
    paje_pt = PajeContainerType.PajeEvent(Name="Process", Type=paje_ct)
    paje_vt = PajeContainerType.PajeEvent(Name="VP", Type=paje_pt)
    paje_tt = PajeContainerType.PajeEvent(Name="Thread", Type=paje_vt)
    paje_st = PajeStateType.PajeEvent(Name="CT_ST", Type=paje_tt)

    paje_entity_waiting = PajeEntityValue.PajeEvent(Name="Waiting", Type=paje_st, Color="0.2,0.2,0.2")

    state_aliases = dict()
    for x in range(len(store.event_names)-1):
        color_code = int(store.event_attributes[x], 16)
        r = (color_code >> 16) & 0xFF
        g = (color_code >>  8) & 0xFF
        b = (color_code      ) & 0xFF
        state_aliases[x] = PajeEntityValue.PajeEvent(Name=store.event_names[x], Type=paje_st, Color="%g,%g,%g" % (r/255.0, g/255.0, b/255.0))

    paje_c_appli = PajeContainerCreate.PajeEvent(Name=store.information.exe, Time=0.000, Type=paje_ct, Container="0")
    paje_container_aliases = dict()

    for nid in store.nodes['id'][:]:
        paje_container_aliases["M%d"%(nid)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%d" % (nid), Type=paje_ct, Container=paje_c_appli)
        for vpid in store.threads['vp_id'][store.threads.node_id == nid].unique():
            paje_container_aliases["M%dV%d"%(nid,vpid)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dV%d" % (nid,vpid), Type=paje_ct,
                                                                                        Container=paje_container_aliases["M%d"%(nid)])
            for thid in store.threads[['thread_id','vp_id']][store.threads.node_id == nid]['thread_id'][store.threads.vp_id == vpid]:
                paje_container_aliases["M%dT%d"%(nid,thid)] = PajeContainerCreate.PajeEvent(Time = 0.0000, Name = "M%dV%dT%d" % (nid,vpid,thid), Type=paje_ct,
                                                                                            Container=paje_container_aliases["M%dV%d"%(nid,vpid)])
                PajeSetState.PajeEvent(Time=0.000, Type=paje_st, Container=paje_container_aliases["M%dT%d"%(nid,thid)], Value="Waiting", task_name="")

    sev = store.events.sort(['node_id', 'thread_id', 'begin'])
    for ev in store.events.iterrows():
        key = "hid=%d:did=%d:tid=%d"%(ev[1].handle_id,ev[1].type,ev[1].id)
        if key in task_names.keys():
            PajeSetState.PajeEvent(Time=float(ev[1].begin), Type=paje_st, Container=paje_container_aliases["M%dT%d"%(ev[1].node_id,ev[1].thread_id)],
                                   Value=state_aliases[ev[1].type], task_name=task_names[key])
        else:
            PajeSetState.PajeEvent(Time=float(ev[1].begin), Type=paje_st, Container=paje_container_aliases["M%dT%d"%(ev[1].node_id,ev[1].thread_id)],
                                   Value=state_aliases[ev[1].type], task_name=store.event_names[ev[1].type])
        PajeSetState.PajeEvent(Time=float(ev[1].end), Type=paje_st, Container=paje_container_aliases["M%dT%d"%(ev[1].node_id,ev[1].thread_id)],
                               Value=paje_entity_waiting, task_name="Waiting")

    paje.endTrace();

    sys.exit(0)

