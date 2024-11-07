/*
 * Copyright (c) 2017-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _redistributeCOMMON_H
#define _redistributeCOMMON_H

#include "parsec.h"
#include "parsec/profiling.h"
#include "parsec/parsec_internal.h"
#include "parsec/parsec_config.h"
#include "tests/tests_timing.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Update PASTE_CODE_PROGRESS_KERNEL below if you change this list */
enum iparam_t {
  /* Source Matrix */
  IPARAM_P,            /* Rows in the process grid          */
  IPARAM_Q,            /* Columns in the process grid       */
  IPARAM_M,            /* Number of rows of the matrix      */
  IPARAM_N,            /* Number of columns of the matrix   */
  IPARAM_MB,           /* Number of rows in a tile          */
  IPARAM_NB,           /* Number of columns in a tile       */
  IPARAM_SMB,          /* Number of rows in a super-tile    */
  IPARAM_SNB,          /* Number of columns in a super-tile */
  IPARAM_DISI,         /* row start point                   */
  IPARAM_DISJ,         /* column start point                */
  /* Target/Redistributed Matrix */
  IPARAM_P_R,          /* Rows in the process grid          */
  IPARAM_Q_R,          /* Columns in the process grid       */
  IPARAM_M_R,          /* Redistributed M size              */
  IPARAM_N_R,          /* Redistributed N size              */
  IPARAM_MB_R,         /* Redistributed MB size             */
  IPARAM_NB_R,         /* Redistributed NB size             */
  IPARAM_SMB_R,        /* Number of rows in a super-tile    */
  IPARAM_SNB_R,        /* Number of columns in a super-tile */
  IPARAM_DISI_R,       /* row start point                   */
  IPARAM_DISJ_R,       /* column start point                */
  /* Matrix Common */
  IPARAM_RADIUS,       /* Radius of ghost region            */
  IPARAM_M_SUB,        /* Row size of submatrix             */
  IPARAM_N_SUB,        /* Column size of submatrix          */
  /* Others */
  IPARAM_RANK,         /* Rank                              */
  IPARAM_NNODES,       /* Number of nodes                   */
  IPARAM_NCORES,       /* Number of cores                   */
  IPARAM_CHECK,        /* Checking activated or not         */
  IPARAM_VERBOSE,      /* How much noise do we want?        */
  IPARAM_GETTIME,      /* Get time                          */
  IPARAM_NUM_RUNS,     /* Number of runs                    */
  IPARAM_THREAD_MULTIPLE,  /* MPI thread init type              */
  IPARAM_NO_OPTIMIZATION_VERSION, /* MPI thread init type              */
  IPARAM_SIZEOF
};

enum dparam_t {
  DPARAM_NETWORK_BANDWIDTH,    /* Network bandwidth                 */
  DPARAM_MEMCPY_BANDWIDTH,    /* Network bandwidth                 */
  DPARAM_SIZEOF
};

void print_usage(void);

parsec_context_t *setup_parsec(int argc, char* argv[], int *iparam, double *dparam);

void cleanup_parsec(parsec_context_t* parsec, int *iparam, double *dparam);

#endif /* _redistributeCOMMON_H */
