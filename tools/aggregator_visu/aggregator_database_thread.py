#!/usr/bin/env python

"""
Copyright (c) 2015-2018 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved."

This python script is the source code of thread in charge of dumping
data into a sqlite database

@author Damien Genet
@email dgenet@icl.utk.edu
@date 2017-02-09
@version 0.1.0
"""

import socket
import time
from threading import *
from collections import deque
from data_handler import *
from aggregator_math_thread import *
import struct
import sqlite3 as sql
from itertools import *


def init_database(conn, cursor):
    try:
        cursor.execute("CREATE TABLE Version ( release_number CHAR(20), creation_time INTEGER );")
        cursor.execute("CREATE TABLE Keys ( id_key INTEGER PRIMARY KEY, key CHAR(32), N INTEGER, M INTEGER, P INTEGER, Q INTEGER, binary CHAR(32) );")
        cursor.execute("CREATE TABLE Events ( id_event INTEGER PRIMARY KEY, foreign_id_key INTEGER, process INTEGER, stream INTEGER, time REAL, val REAL );")
        conn.commit()
    except sql.Error as msg:
        print 'Error creating db {0}'.format(msg)

    try:
        cursor.execute("INSERT INTO Version VALUES ( '0.1.0', datetime() );")
        conn.commit()
    except sql.Error as msg:
        print 'Error inserting first entry {0}'.format(msg)

def connect_to_database(params):
    conn = sql.connect(params.db_name)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT count(*) FROM Version")
    except sql.Error as msg:
        init_database(conn, cursor)
        cursor.execute("SELECT count(*) FROM Version")
    row, = cursor.fetchone()
    if row is not 1:
        print 'weird, you have multiple entries in the control table'
        return None, None
    else:
        params.debug(1, '{0} loaded!'.format(params.db_name))
        return conn, cursor


def aggregator_database_thread(Data, params, sem):

    conn, cursor = connect_to_database(params)
    if conn is None:
        params.stop()
        return

    params.db_event_queue = deque()
    params.db_key_queue = deque()
    params.db_active = 1
    sem.release()

    i = 0
    L = None
    last_push = time.clock()

    while not params.stop_event.is_set() or len(params.db_event_queue) > 0:

        while len(params.db_key_queue) > 0:
            sem, L = params.db_key_queue.popleft()
            cursor.execute("INSERT INTO Keys (key, N, M, P, Q, binary) VALUES (?,?,?,?,?,?);", L)
            Data.ht[ L[0] ].setIdKey(cursor.lastrowid)
            conn.commit()
            sem.release()

        if len(params.db_event_queue) >= 10 or (time.clock() - last_push) > 0.25 and len(params.db_event_queue) > 0:
            size = len(params.db_event_queue)
            L = list(starmap(params.db_event_queue.popleft, repeat((), size)))
            cursor.executemany("INSERT INTO Events (foreign_id_key, process, stream, time, val) VALUES (?,?,?,?,?)", L)
            conn.commit()
            last_update = time.clock()

    conn.close()
