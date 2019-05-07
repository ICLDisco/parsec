#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script is the source code for the thread in charge of
streaming data to the GUI application

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import socket
from threading import *
from data_handler import *

def aggregator_gui_thread(conn, addr, Data, params, data):
    id_s = data[0]
    # sending list of keys + parameters
    message = '{0};'.format(len(params.keys))

    for key in params.keys:
        M = Data.getM(key)
        N = Data.getN(key)
        P = Data.getP(key)
        Q = Data.getQ(key)
        message += '{0};{1};{2};{3};{4};'.format(N, M, P, Q, key)
        Data.addKey(key, N, M, P, Q)

    params.debug(1,' << --{0}--'.format(message))
    conn.send(message)
    # receiving GUI list of keys
    data = conn.recv(1024)
    params.debug(1,' >> --{0}--'.format(data))
    keys = splitclean(data, ';')
    print 'GUI {0} connected from {1}:{2} has requested the following keys: {3}'.format(id_s, addr[0], addr[1], keys)

    Sems = {}
    for key in keys:
        N = Data.getN(key)
        populateSemDict(Sems, key, N)
        Data.appendSemDict(Sems, key, N)

    while not params.stop_event.is_set():
        message = ''
        for key in keys:
            iN = 0
            message = message + key+';'
            N = Data.getN(key)
            M = Data.getM(key)
            while iN < N:
                params.debug(9,'Acquiring {0}:{1}'.format(key, iN))
                Sems[key][iN].acquire()
                iM = 0
                while iM < M:
                    (time, val) = Data.readHead(key, iN, iM)
                    #(time, val) = Quantities[key][iN*M+iM].readhead()
                    message += str(time) + ':' + str(val) + ';'
                    iM += 1
                iN += 1

        params.debug(9,' << --{0}--'.format(message))
        conn.send(message)
        data = conn.recv(32)
        data = data.rstrip("\n\r")
        if not data or 'EOC'  in data or data is '':
            break
        #else assume it's ACK
    conn.close()
