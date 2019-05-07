#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script is the source code for the thread that receive data
from the simulation

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import socket
from threading import *
from data_handler import *
from aggregator_math_thread import *
import struct

def aggregator_simu_thread(conn, addr, Data, params, data):
    params.debug(1, '-HEADER-{0}-HEADER-'.format(data))
    name = data[0]
    iN = int(data[1])
    N = int(data[2])
    M = int(data[3])
    P = params.getP()
    Q = params.getQ()
    #if len(data) > 4:
        #P = int(data[4])
    #if len(data) > 5:
        #Q = int(data[5])
    keys = data[6:]

    for key in keys:
        #if key not in params.keys:
        key_diff = key+'_diff'
        key_reduce = key+'_reduce'
        Data.addKey(key, N, M, P, Q)
        Data.addKey(key_diff, N, M, P, Q)
        Data.addKey(key_reduce, 1, N, 1, 1)
        params.addKeys((key, key_diff, key_reduce,))

        if iN is 0:
            params.debug(1, 'Creating math thread for {0} {1} {2}'.format(key, 'fprime', key_diff))
            t = Thread(target=aggregator_math_thread, args=(Data, key, 'fprime', key_diff, params))
            t.start();
            params.appendMath(t)

            params.debug(1, 'Creating math thread for {0} {1} {2}'.format(key_diff, 'reduce', key_reduce))
            u = Thread(target=aggregator_math_thread, args=(Data, key_diff, 'reduce', key_reduce, params))
            u.start();
            params.appendMath(u)


    if iN == 0:
        print '  > {4}, {0} processes (grid {2}x{3}) of {1} streams.'.format(N, M, P, Q, name)
        print 'Available keys to plot: {0}'.format(params.keys)

    while not params.stop_event.is_set():
        try:
            keys = []
            data = conn.recv(4096)
            timestamp,nbstructs = struct.unpack_from("dH", data, offset=0);
            offset = 10
            nbvalues = ((len(data)-offset)/nbstructs - 8)/8
            for s in range(nbstructs):
                name, = struct.unpack_from("8s", data[offset:offset+8])
                key = name.replace('\x00', '')
                keys.append(key)
                offset += 8
                for iM in range(nbvalues):
                    value, = struct.unpack_from("Q", data, offset)
                    val = float(value) / 10**9
                    Data.appendData(key, iN, iM, timestamp, val)
                    offset += 8


            for key in keys:
                c = Data.release(key, iN)
                params.debug(9, 'Released {2} semaphores {0}:{1}'.format(key, iN, c))


        except struct.error as msg:
            print '-{0}--> disconnection'.format(msg)
            conn.close()
            return

        except Exception as msg:
            print '-{0}-'.format(msg)
            conn.close()
            return

    conn.close()
