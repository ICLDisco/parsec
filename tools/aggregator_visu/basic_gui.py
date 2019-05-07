#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script is the main script for the basic GUI that just prints
data in the terminal.

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import socket
import sys,getopt
import threading
from data_handler import *
from iparam import iParam as iP


def agregator_thread(conn, params, Data):
    # send id
    message = 'Dump_Gui'
    print ' << --{0}--'.format(message)
    conn.send(message)
    # receiving keys list
    data = conn.recv(1024)
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
        params.addKey(key)
        message += key + ';'
        iK += 1
    # send keys list
    print ' << --{0}--'.format(message)
    conn.send(message)

    while not params.stop_event.is_set():
        data = conn.recv(2048)
        print '--{0}--'.format(data)
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

    agregator_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    agregator_socket.connect((params.getAgregatorAddress(), params.getAgregatorPort()))
    agregator_t = threading.Thread(target=agregator_thread, args=(agregator_socket, params, Data))
    agregator_t.start();

    s = raw_input('EOC to stop > ')
    params.stop_event.set()

    agregator_t.join()
    agregator_socket.close()


if __name__ == "__main__":
   main(sys.argv[:])





