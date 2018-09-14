#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script is the source code for the iParam class. This class
is used to parse and store command line parameters. It provides accessors
and mutators to the stored fileds.

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import sys, getopt
import sqlite3 as sql
from threading import *

def verbose(string):
    print '{0}'.format(string)

def noverbose(string):
    pass


class iParam(object):
    def __init__(self, N = 1, M = 1, P = 1, Q = 1, aa = '', ap = 30420, gp = 34200, sp = 30042):
        self.N = N
        self.M = M
        self.P = P
        self.Q = Q
        self.agregator_address = aa
        self.agregator_port = ap
        self.gui_port = gp
        self.simu_port = sp
        self.stop_event = Event()
        self.keys = []
        self.lock = Lock()
        self.gui_threads = []
        self.simu_threads = []
        self.math_threads = []
        self.accepting = None
        self.keyplot = None
        self.printfunc = noverbose
        self.debug_level = 0
        self.db = None
        self.db_active = None
        self.db_name = None
        self.db_event_queue = None
        self.db_key_queue = None

    def db_push_event(self, t):
        self.db_event_queue.append(t)

    def db_push_key(self, t):
        self.db_key_queue.append(t)

    def setDebugLevel(self, lvl):
        self.debug_level = lvl

    def debug(self, level_req, string):
        if self.debug_level >= level_req:
            self.printfunc(string)

    def appendGui(self, t):
        self.gui_threads.append(t)

    def appendSimu(self, t):
        self.simu_threads.append(t)

    def appendMath(self, t):
        self.math_threads.append(t)

    def join(self):
        for t in self.gui_threads:
            t.join()
        for t in self.simu_threads:
            t.join()
        for t in self.math_threads:
            t.join()
        if self.accepting is not None:
            self.accepting.join()
        if self.db is not None:
            self.db.join()

    def printself(self):
        print 'params = (M = {0}, N = {1}, P = {2}, Q = {3}, Agregator = ({4}:{5}), GUI port = {6}, Simulation port = {7}, keyplot = {8})'.format(self.M, self.N, self.P, self.Q, self.agregator_address, self.agregator_port, self.gui_port, self.simu_port, self.keyplot)

    def stop(self):
        self.stop_event.set()

    def setN(self, N):
        self.N = N

    def setM(self, M):
        self.M = M

    def setP(self, P):
        self.P = P

    def setQ(self, Q):
        self.Q = Q

    def setAgregatorAddress(self, aa):
        self.agregator_address = aa

    def setAgregatorPort(self, ap):
        self.agregator_port = ap

    def setGuiPort(self, gp):
        self.gui_port = gp

    def setSimuPort(self, sp):
        self.simu_port = sp

    def setPort(self, p):
        self.port = p

    def setDbName(self, n):
        self.db_name = n

    def addKey(self, k):
        self.lock.acquire()
        self.keys.extend((k,))
        self.lock.release()

    def addKeys(self, keys):
        self.lock.acquire()
        for key in keys:
            if key not in self.keys:
                self.keys.extend((key,))
        self.lock.release()

    def getN(self):
        return self.N

    def getM(self):
        return self.M

    def getP(self):
        return self.P

    def getQ(self):
        return self.Q

    def getAgregatorAddress(self):
        return self.agregator_address

    def getAgregatorPort(self):
        return self.agregator_port

    def getGuiPort(self):
        return self.gui_port

    def getPort(self):
        return self.port

    def getSimuPort(self):
        return self.simu_port

    def setKeyPlot(self, k):
        self.keyplot = k

    def getKeyPlot(self):
        return self.keyplot

    def printHelp(self, bin):
        if 'aggregator' in bin:
            print '{0} -p <listenning port number> -N <nb MPI processes> -M <nb Argo streams> -P <proc grid nb rows> -Q <proc grid nb cols> -v <debug level> -d <db name>'.format(bin)
        elif 'plot' in bin:
            print '{0} -g <aggregator address> -a <aggregator port number> -k <key to plot> -v <debug level>'.format(bin)
        else:
           # print '{0} -a <agregator port number> -s <simulation port number> -g <agregator address> -N <nb MPI processes> -M <nb Argo streams> -P <proc grid nb rows> -Q <proc grid nb cols>'.format(bin)
            pass

    def parseArgv(self, argv):
        bin = argv[0]
        argv = argv[1:]
        try:
            opts, args = getopt.getopt(argv,"v:ha:g:s:N:M:P:Q:p:k:d:",["--simulation-port=","--agregator-port=","--agregator-address=","--key-plot=","--verbose=","--db-name="])
        except getopt.GetoptError:
            self.printHelp(bin)
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                self.printHelp(bin)
                sys.exit(0)
            elif opt in ("-v", "--verbose"):
                self.printfunc = verbose
                self.setDebugLevel(int(arg))
            elif opt in ("-a", "--agregator-port"):
                self.setAgregatorPort(int(arg))
            elif opt in ("-d", "--db-name"):
                self.setDbName(arg)
            elif opt in ("-g", "--agregator-address"):
                self.setAgregatorAddress(arg)
            elif opt in ("-s", "--simu-port"):
                self.setSimuPort(int(arg))
            elif opt in ("-p", "--port"):
                self.setPort(int(arg))
            elif opt in ("-k", "--key-plot"):
                self.setKeyPlot(arg)
            elif opt in ("-N"):
                self.setN(int(arg))
            elif opt in ("-M"):
                self.setM(int(arg))
            elif opt in ("-P"):
                self.setP(int(arg))
            elif opt in ("-Q"):
                self.setQ(int(arg))

        if self.P * self.Q is not self.N:
            self.Q = 1
            self.P = self.N

