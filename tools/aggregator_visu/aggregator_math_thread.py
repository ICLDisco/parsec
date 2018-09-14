#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script is the source code for the mathematical thread in
charge of taking data and applying math operator to generate new data

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import threading
from threading import *
import numpy
from iparam import iParam as iP
from data_handler import *


def aggregator_math_thread(Data, key_i, op, key_o, params):
    params.debug(2,'Starting Math Thread: {0} {1} {2}'.format(key_i, op, key_o))
    P = Data.getP(key_i)
    Q = Data.getQ(key_i)
    N = Data.getN(key_i)
    M = Data.getM(key_i)

    Sems = {}
    populateSemDict(Sems, key_i, N)
    Data.appendSemDict(Sems, key_i, N)

    while not params.stop_event.is_set():
        if 'fprime' in op:
            #requires at least two points
            iN = 0
            while iN < N:
                params.debug(9,'Acquiring {0}:{1}'.format(key_i, iN))
                Sems[key_i][iN].acquire()
                iM = 0
                while iM < M:
                    x, y = Data.getXY(key_i, iN, iM)
                    if len(x) > 1:
                        if numpy.fabs(x[-1] - x[-2]) > 0.0001:
                            deriv = (y[-1] - y[-2])/(x[-1] - x[-2])
                            time = x[-1]
                            Data.appendData(key_o, iN, iM, time, deriv)
                    iM += 1
                c = Data.release(key_o, iN)
                params.debug(9,'Released {2} semaphores {0}:{1}'.format(key_o, iN, c))
                iN += 1
            Data.release(key_o)

        if 'reduce' in op:
            iN = 0
            while iN < N:
                params.debug(9,'Acquiring {0}:{1}'.format(key_i, iN))
                Sems[key_i][iN].acquire()
                val = 0.
                time = 0.
                iM = 0
                while iM < M:
                    x, y = Data.getXY(key_i, iN, iM)
                    if len(x) > 0:
                        val += y[-1]
                        time += x[-1]
                    iM += 1
                Data.appendData(key_o, 0, iN, time/M, val)
                iN += 1
            c = Data.release(key_o, 0)
            params.debug(9,'Released {2} semaphores {0}:{1}'.format(key_o, iN, c))

        c = Data.release(key_o)
        params.debug(9,'Released {1} semaphores {0}'.format(key_o, c))

    #unregister the semaphores
    Data.removeSemDict(Sems, (key_i,), N)

