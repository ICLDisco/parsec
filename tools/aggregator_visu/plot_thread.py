#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script contains the source code of the thread in charge of
updating the window plots

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation as FunkAnimation
from threading import *
from data_handler import *
from iparam import iParam as iP

lg = '#BBBBBB'
dg = '#777777'
gr = '#999999'

def initSem(n):
    s = Semaphore(n)
    return s


def movie_thread(params, data, rdy_event, keys):
    #generate a frame from the last update of data
    FFMpegWriter = manimation.writers['ffmpeg']


def plot_thread(params, Data, key):
    P = Data.getP(key)
    Q = Data.getQ(key)
    N = Data.getN(key)
    M = Data.getM(key)

    tkey = key.split('_')

    Sems = {}
    populateSemDict(Sems, key, N)
    Data.appendSemDict(Sems, key, N)

    fig = plt.figure()
    axes = {}
    lines = {}

    iP = 0
    while iP < P:
        iQ = 0
        while iQ < Q:
            iN = iP * Q + iQ
            axe = fig.add_subplot(P, Q, iN+1)
            if iP == P - 1:
                axe.set_xlabel('Time in seconds')
            if iQ == 0:
                axe.set_ylabel('{0} per second'.format(tkey[0]))
            ymax = 120#200+750*M #M avant d'inverser N et M
            axe.set_ylim(-10, ymax)
            axetitle = Data.getPlotTitle(key, iN)
            axe.set_title(axetitle)
            axes[iN] = axe

            lines[iN] = {}
            iM = 0
            xmin = 1000000000.
            xmax = 0.
            while iM < M:
                axes[iN].set_xlim(0,1)
                legend = Data.getLegend(key, iN, iM)
                lines[iN][iM] = Line2D([], [], color=Data.getColor(key,iN, iM), marker='.', linestyle='-', label=legend)
                axe.add_line(lines[iN][iM])
                iM += 1
            axe.legend()
            iQ += 1
        iP += 1

    plt.ion()
    plt.show()

    while not params.stop_event.is_set():
        iP = 0
        while iP < P:
            iQ = 0
            while iQ < Q:
                iN = iP * Q + iQ #process rank
                Sems[key][iN].acquire()
                iM = 0
                xmin = 1000000000.
                xmax = 0.
                while iM < M:
                    x, y = Data.getXY(key, iN, iM)
                    if xmin > x[0]:
                        xmin = x[0]
                    if xmax < x[-1]:
                        xmax = x[-1]
                    axes[iN].set_xlim(xmin, xmax)
                    lines[iN][iM].set_data(x,y)
                    iM += 1
                iQ += 1
            iP += 1
        fig.canvas.draw()

