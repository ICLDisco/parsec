#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script is the main script for the gui application.

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import socket
import sys,getopt
import threading
from threading import *
import numpy

from iparam import iParam as iP
from plot_thread import movie_thread, plot_thread
from data_handler import *

gui_threads = []

def agregator_thread(conn, params, Data, sem):
    # send id
    message = '42'
    #print ' << --{0}--'.format(message)
    conn.send(message)
    # receiving keys list
    data = conn.recv(4096)
    print ' >> --{0}--'.format(data)
    tmp = splitclean(data, ';')
    # K keys ; N mpi processes ; M streams ;
    K = int(tmp[0])
    tmp = tmp[1:]
    iK = 0
    message = ''
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
            Data.setPlotTitles(key, '{0} accumulated per proc over time'.format(tkey[0]))
            Data.setLegends(key, 'MPI process ')
        elif 'diff' in key:
            tkey = key
            tkey = tkey.split('_')
            Data.setTitle(key, '{0} differenciated over time'.format(tkey[0]))
            Data.setPlotTitles(key, 'Mpi process ')
            Data.setLegends(key, 'ARGO stream ')
        else:
            Data.setTitle(key, '{0} counter over time'.format(key))
            Data.setPlotTitles(key, 'Mpi process ')
            Data.setLegends(key, 'ARGO stream ')

        params.addKey(key)
        message += key + ';'
        iK += 1
    # send keys list
    print ' << --{0}--'.format(message)
    conn.send(message)

    sem.release()

    while not params.stop_event.is_set():
        data = conn.recv(4096)
        #print ' >> --{0}--'.format(data)
        # gflops;1.0:3.0;1.2:5.3;
        #    key; time:val_p0s0 ; ... ; time:val_p0sM ; time:val_p1s0 ; ... ; time:val_pNsM
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
                Data.release(key, iN)
                iN += 1
            k += 1

        ack = 'ACK'
        conn.send(ack)



def main(argv):
    params = iP()
    params.parseArgv(argv)
    Data = HashData('', params.getN(), params.getM())
    sem = initSem(0)

    agregator_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    agregator_socket.connect((params.getAgregatorAddress(), params.getAgregatorPort()))
    agregator_t = threading.Thread(target=agregator_thread, args=(agregator_socket, params, Data, sem,))
    agregator_t.start();

    sem.acquire()

    while True:
        print 'EOC to quit, plot <key>'
        print 'Available keys: {0}'.format(params.keys)
        s = raw_input('> ')
        s = s.rstrip("\n\r")

        if 'EOC' in s:
            params.stop_event.set()
            break
        elif 'plot' in s:
            tmp = s.split(' ')
            keys = splitclean(tmp[1], ';')
            for key in keys:
                t = Thread(target=plot_thread, args=(params, Data, key,))
                t.start()
                gui_threads.append(t)

    agregator_t.join()
    agregator_socket.close()
    for t in gui_threads:
        t.join()


if __name__ == "__main__":
   main(sys.argv[:])
