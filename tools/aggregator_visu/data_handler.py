#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script is the source code for the data hierarchy. File contains
code for Vector, GridData, HashData, and SemArray classes.
Vector is a sliding window implemented by swapping the two halves of the
array.


@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import sqlite3 as sql
from threading import *
from random import randrange as rr
import numpy

ref_colors = [ '#4169E1', '#B22222', '#228B22', '#FFD700', '#00BFFF', '#FFA07A', '#00FA9A', '#FFFFE0', '#F0FFFF', '#8B008B', '#808000', '#F5F5DC' ]

class Vector(object):
    def __init__(self, iS, s = 200):
        self.id = id(self)
        self.size = s
        self.points = 0
        self.half = s // 2
        self.start = rr(self.half, s) #desynchronize the swaps by randomly distributing them
        self.position = self.start
        self.x = numpy.zeros(self.size, dtype=float)
        self.y = numpy.zeros(self.size, dtype=float)
        self.lock = Lock()
        self.color = ref_colors[iS]
        self.legend = None

    def __getitem__(self, index):
        if index < self.size:
            return (self.x[index], self.y[index])
        else:
            raise Exception('__getitem__: Out of boundaries: [{0}:{1}], points = {2}'.format(self.start, self.position, self.points))

    def __setitem__(self, index, value):
        x, y = value
        if self.size < index:
            self.x[index] = x
            self.y[index] = y
        else:
            raise Exception('__setitem__: Out of boundaries: [{0}:{1}], points = {2}'.format(self.start, self.position, self.points))

    def swap(self):
        self.lock.acquire()
        self.position = self.half
        self.start -= self.half
        self.x[:self.half] = self.x[self.half:]
        self.y[:self.half] = self.y[self.half:]
        self.lock.release()


    def append(self, x, y):
        #print 'start = {1}, position = {0}, points = {2}'.format(self.position, self.start, self.points)

        if len(self.x) > 1:
            if self.x[self.position-1] > x:
                #print 'prev_x = {0}, x = {1} ; prev_y = {2}, y = {3}'.format(self.x[self.position-1], x, self.y[self.position-1], y)
                pass
        try:
            self.x[self.position] = x
            self.y[self.position] = y
        except IndexError as msg:
            raise Exception('V{3}, append: Out of boundaries: [{0}:{1}], points = {2}'.format(self.start, self.position, self.points, self.id%1000))


        self.position += 1
        self.points += 1
        if self.points > self.half:
            self.start += 1
        try:
            if self.position == self.size:
                self.swap()
        except:
            raise Exception('V{3}, swap: Out of boundaries: [{0}:{1}], points = {2}'.format(self.start, self.position, self.points, self.id%1000))


    def getLegend(self):
        return self.legend

    def setLegend(self, l):
        self.legend = l

    def getX(self):
        return self.x[self.start:self.position]

    def getY(self):
        return self.y[self.start:self.position]

    def getColor(self):
        return self.color

    def readHead(self):
        return (self.x[self.position-1], self.y[self.position-1])


class GridData(object):
    def __init__(self, k='', n = 1, m = 1, p = -1, q = -1):
        self.key = k
        self.sem = []
        self.proc_sem = {}
        self.N = n
        self.M = m
        self.P = p
        self.Q = q
        self.titles = {}
        self.plots = {}
        i = 0
        while i < self.N * self.M:
            self.plots[i] = Vector(i%self.M, 180)
            i += 1
        i = 0
        while i < self.N:
            self.titles[i] = None
            self.proc_sem[i] = []
            i += 1
        self.title = None
        self.id_key = None

    def getColor(self, iN, iM):
        if 0 > iN or iN >= self.N or 0 > iM or iM >= self.M:
            raise Exception('Out of boundaries, iN = {0}, N = {2}, iM = {1}, M = {3}'.format(iN, iM, self.N, self.M))
        return self.plots[iN*self.M+iM].getColor()

    def setTitle(self, t):
        self.title = t

    def getTitle(self):
        return self.title

    def setLegends(self, l):
        iN = 0
        while iN < self.N:
            iM = 0
            while iM < self.M:
                self.plots[iN*self.M+iM].setLegend('{0}{1}'.format(l, iM))
                iM += 1
            iN += 1

    def getLegend(self, iN, iM):
        if 0 > iN or iN >= self.N or 0 > iM or iM >= self.M:
            raise Exception('Out of boundaries, iN = {0}, N = {2}, iM = {1}, M = {3}'.format(iN, iM, self.N, self.M))
        return self.plots[iN*self.M+iM].getLegend()

    def setPlotTitles(self, t):
        if self.N == 1:
            self.titles[0] = '{0}'.format(t)
        else:
            iN = 0
            while iN < self.N:
                self.titles[iN] = '{0}{1}'.format(t, iN)
                iN += 1

    def getPlotTitle(self, iN):
        if 0 > iN or iN >= self.N:
            raise Exception('Out of boundaries, iN = {0}, N = {2}, iM = {1}, M = {3}'.format(iN, self.N))
        return self.titles[iN]

    def printself(self):
        print '     [ key = {}, N = {}, M = {}, P = {}, Q = {} ]'.format(self.key, self.N, self.M, self.P, self.Q)

    def __getitem__(self, idx):
        return self.plots[idx]

    def append(self, iN, iM, x, y):
        if 0 > iN or iN >= self.N or 0 > iM or iM >= self.M:
            raise Exception('Out of boundaries, iN = {0}, N = {2}, iM = {1}, M = {3}'.format(iN, iM, self.N, self.M))
        try:
            self.plots[iN*self.M+iM].append(x, y)
        except Exception as msg:
            print '{0}, key = {1}, iN = {2}, N = {3}, iM = {4}, M = {5}'.format(msg, self.key, iN, self.N, iM, self.M)

    def getXY(self, iN, iM):
        if 0 > iN or iN >= self.N or 0 > iM or iM >= self.M:
            raise Exception('Out of boundaries, iN = {0}, N = {2}, iM = {1}, M = {3}'.format(iN, iM, self.N, self.M))
        return (self.plots[iN*self.M+iM].getX(), self.plots[iN*self.M+iM].getY())

    def getN(self):
        return self.N

    def getM(self):
        return self.M

    def getP(self):
        return self.P

    def getQ(self):
        return self.Q

    def readHead(self, iN, iM):
        if 0 > iN or iN >= self.N or 0 > iM or iM >= self.M:
            raise Exception('Out of boundaries, iN = {0}, N = {2}, iM = {1}, M = {3}'.format(iN, iM, self.N, self.M))
        return self.plots[iN*self.M+iM].readHead()

    def release(self):
        if self.sem is not None:
            for sem in self.sem:
                sem.release()
        return len(self.sem)

    def releaseProc(self, iN):
        if self.proc_sem[iN] is not None:
            for sem in self.proc_sem[iN]:
                sem.release()
        return len(self.proc_sem[iN])

    def setIdKey(self, i):
        self.id_key = i

    def getIdKey(self):
        return self.id_key

class HashData(object):
    def __init__(self, params):
        self.params = params
        self.keys = []
        self.nb_keys = 0
        self.N = params.getN()
        self.M = params.getM()
        self.P = params.getP()
        self.Q = params.getQ()
        self.ht = dict()
        self.lock = Lock()

    def getColor(self, k, iN, iM):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].getColor(iN, iM)

    def setTitle(self, k, t):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        self.ht[k].setTitle(t)

    def getTitle(self, k):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].getTitle()

    def setLegends(self, k, l):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        self.ht[k].setLegends(l)

    def getLegend(self, k, iN, iM):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].getLegend(iN, iM)

    def setPlotTitles(self, k, t):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        self.ht[k].setPlotTitles(t)

    def getPlotTitle(self, k, iN):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].getPlotTitle(iN)

    def printself(self):
        print 'Data [ N = {0}, M = {1}, P = {2}, Q = {3} ]'.format(self.N, self.M, self.P, self.Q)
        print '     [ keys = {0} ]'.format(self.keys)
        for key in self.keys:
            self.ht[key].printself()

    def __getitem__(self, k):
        return self.ht[k]

    def __delitem__(self, k):
        if k in self.keys:
            del self.ht[k]

#size of the new key is either known or inherited
    def addKey(self, k, n = -1, m = -1, p = -1, q = -1):
        self.lock.acquire()
        if k in self.keys:
            self.lock.release()
            return
        self.params.debug(3,'Adding key {0} > N = {1}, M = {2}, P = {3}, Q = {4}'.format(k, n,m,p,q))
        self.keys.extend((k,))
        self.nb_keys += 1
        if n == -1:
            n = self.N
        if m == -1:
            m = self.M
        if p == -1:
            p = self.P
        if q == -1:
            q = self.Q
        self.ht[k] = GridData(k, n, m, p, q);
        if self.params.db_active is not None:
            sem = initSem(0)
            self.params.db_push_key( (sem,(k,n,m,p,q,'dpotrf')) )
            sem.acquire()
            self.params.debug(4, ' db new key > ({0}, {1}, {2}, {3}, {4}, {5}, {6})'.format(self.ht[k].getIdKey(),k,n,m,p,q,'dpotrf'))
        self.lock.release()

    def addKeys(self, ks):
        self.lock.acquire()
        for k in ks:
            if k in self.keys:
                self.lock.release()
                return
            self.keys.extend((k,))
            self.nb_keys += 1
            self.ht[k] = GridData(k, self.N, self.M);
        self.lock.release()

    def getXY(self, k, iN, iM):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].getXY(iN, iM)

    def getN(self, k):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].getN()

    def getM(self, k):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].getM()

    def getP(self, k):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].getP()

    def getQ(self, k):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].getQ()

    def readHead(self, k, iN, iM):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        return self.ht[k].readHead(iN, iM)

    def appendData(self, k, iN, iM, x, y):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        self.ht[k].append(iN, iM, x, y)
        if self.params.db_active is not None:
            self.params.db_push_event( (self.ht[k].getIdKey(), iN, iM, x, y) )
            self.params.debug(9, ' push event > ({0}, {1}, {2}, {3}, {4})'.format(self.ht[k].getIdKey(), iN, iM, x, y))

    def release(self, k, iN = -1):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        if iN is -1:
            return self.ht[k].release()
        else:
            return self.ht[k].releaseProc(iN)

    def appendSem(self, k, iN, sem):
        if k not in self.ht:
            raise Exception('Unknown key {0}'.format(k))
        self.ht[k].appendSem(iN, sem)

    def appendSemDict(self, Sems, key, N):
        c = 0
        iN = 0
        while iN < N:
            c += 1
            self.ht[key].proc_sem[iN].extend((Sems[key][iN],))
            iN += 1

    def removeSemDict(self, Sems, keys, N):
        for key in keys:
            iN = 0
            while iN < N:
                self.ht[key].proc_sem[iN].remove(Sems[key][iN])
                iN += 1



def splitclean(string, token):
    tmp = string.split(token)
    while len(tmp) > 0 and tmp[-1] is '':
        tmp = tmp[:-1]
    return tmp


def initSem(n):
    s = Semaphore(n)
    return s

def populateSemDict(d, key, N):
    i = 0
    d[key] = {}
    while i < N:
        d[key][i] = initSem(0)
        i += 1





class SemArray(object):
    def __init__(self, sz, n):
        self.sems = {}
        self.ready = {}
        self.done = {}
        self.current = 0
        self.size = sz
        i = 0
        while i < sz:
            self.sems[i] = initSem(n)
            self.ready[i] = False
            self.done[i] = False
            i += 1

    def __getitem__(self, k):
        if k in self.sems:
            return self.sems[k]

    def __delitem__(self, k):
        if k in self.sems:
            del self.sems[k]
            del self.bool[k]

    def nextGen(self):
        i = 0
        while i < sz:
            self.ready[i] = False
            self.done[i] = False
            i += 1
        self.current = 0

    def next(self, i):
        if self.current == self.size:
            self.nextGen()
            return -1

        i = 0
        while i < self.size:
            if not self.done[i]:
                self.ready[i] = self.sems[i].acquire(False)
                if self.ready[i]:
                    self.done[i] = True
                    self.current += 1
                    return i
            i += 1
            if i == self.size:
                i = 0
