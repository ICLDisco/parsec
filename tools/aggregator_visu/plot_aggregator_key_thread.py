#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script contains the source of the thread connecting the GUI
with the aggregator.

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import socket
from data_handler import *

def plot_aggregator_thread(conn, params, Data, sem, name):
    # send type;id
    message = '{0};{1}'.format(2, name)
    params.debug(1,' << --{0}--'.format(message))
    conn.send(message)
    # receiving keys list
    data = conn.recv(4096)
    params.debug(1,' >> --{0}--'.format(data))
    tmp = splitclean(data, ';')
    # K keys ; N mpi processes ; M streams ;
    K = int(tmp[0])
    tmp = tmp[1:]
    iK = 0
    while iK < K:
        N = int(tmp[0])
        M = int(tmp[1])
        P = int(tmp[2])
        Q = int(tmp[3])
        key = tmp[4]
        tmp = tmp[5:]
        Data.addKey(key, N, M, P, Q)

        if 'reduce' in key:
            tkey = key
            tkey = tkey.split('_')
            Data.setTitle(key, '')
            Data.setPlotTitles(key, '{0} per proc over time'.format(tkey[0]))
            Data.setLegends(key, 'MPI process ')
        elif 'diff' in key:
            tkey = key
            tkey = tkey.split('_')
            Data.setTitle(key, 'Evolution of {0}'.format(tkey[0]))
            Data.setPlotTitles(key, 'Mpi process ')
            Data.setLegends(key, 'ARGO stream ')
        else:
            Data.setTitle(key, 'Counters value for {0}'.format(key))
            Data.setPlotTitles(key, 'Mpi process ')
            Data.setLegends(key, 'ARGO stream ')

        params.addKey(key)
        iK += 1

    message = '{0}'.format(params.getKeyPlot())
    params.debug(1,' << --{0}--'.format(message))
    conn.send(message)

    K = 1
    sem.release()

    while not params.stop_event.is_set():
        data = conn.recv(4096)
        params.debug(1,' >> --{0}--'.format(data))
        tmp = splitclean(data, ';')
        k = 0
        while k < K:
            key = tmp[0]
            tmp = tmp[1:]
            iN = 0
            N = Data.getN(key)
            M = Data.getM(key)
            while iN < N:
                iM = 0
                while iM < M:
                    time_s, val_s = tmp[0].split(':')
                    tmp = tmp[1:]
                    time = float(time_s)
                    val = float(val_s)
                    Data.appendData(key, iN, iM, time, val)
                    iM += 1
                params.debug(9, 'Releasing {0}:{1}'.format(key, iN))
                Data.release(key, iN)
                iN += 1
            k += 1

        ack = 'ACK'
        conn.send(ack)

