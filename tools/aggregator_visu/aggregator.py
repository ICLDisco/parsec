#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script is the main script for the aggregator application.

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import socket
import sys, getopt
from threading import *
from time import sleep
import numpy
from iparam import iParam as iP
from data_handler import *
from aggregator_math_thread import *
from aggregator_simu_thread import *
from aggregator_gui_thread import *
from aggregator_database_thread import *

C = 64

def socket_binding(s, port_number):
    try:
        s.bind((socket.gethostname(), port_number))
    except socket.error as msg:
        print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
        sys.exit()
    s.listen(C) # keeps C pending connections


def accepting(s, Data, params):
    s.settimeout(5.)
    while not params.stop_event.is_set():
        try:
            conn, addr = s.accept()
            data = conn.recv(2048)
            data = splitclean(data, ';')
            tmp = data[1:]
            params.debug(1,' >> --{0}--'.format(data))
            if '1' in data[0]:
                t = Thread(target=aggregator_simu_thread, args=(conn, addr, Data, params,tmp,))
                params.appendSimu(t)
                t.start()
            if '2' in data[0]:
                t = Thread(target=aggregator_gui_thread, args=(conn, addr, Data, params, tmp,))
                params.appendGui(t)
                t.start()

        except socket.timeout as msg:
            pass


def main(argv):
    params = iP()
    params.parseArgv(argv)

    Data = HashData(params)

    if params.db_name is not None:
        sem = initSem(0)
        t = Thread(target=aggregator_database_thread, args=(Data, params, sem))
        t.start()
        params.db = t
        sem.acquire()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_binding(s, params.getPort());
    print 'Listening on '+socket.gethostname()+' port ', params.getPort()
    t = Thread(target=accepting, args=(s, Data, params))
    params.accepting = t
    t.start()

    print 'hit a key to stop!'
    st = raw_input(' > ')

    params.stop()
    params.join()
    s.close()

if __name__ == "__main__":
   main(sys.argv[:])
